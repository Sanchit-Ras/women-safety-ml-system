"""Reproduce the original notebook's feature windows and prove the target leak.

The notebook (``WomenSafety_XGBoost.ipynb``) selected its "chronological"
columns with ``'Change in Crime Pct' in col``, which also matches every
``diff_Change in Crime Pct ...`` column. After sorting by year the sequence is
``c01, c02, d02, c03, d03, c04, d04, ...`` and a window of six followed by a
label lands on ``[c01, c02, d02, c03, d03, c04] -> d04`` where
``d04 = c04 - c03``: the label is arithmetic on two of the features.

This script rebuilds those exact windows and reports:
  1. what share of windows have a diff column as the label,
  2. that the label can be reconstructed *exactly* from features on those
     windows (max absolute error), and
  3. the R^2 an XGBoost model reaches on the leaky windows under the
     notebook's own district-grouped split, for comparison with the honest
     pipeline in ``ml/train.py``.

Run with ``python -m ml.diagnose_leak``.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor

from ml.features import ERA_A_PATH, ERA_A_CATEGORIES

WINDOW = 6
SEED = 42


def notebook_final_dataframe() -> pd.DataFrame:
    """Cells 2-3 of the notebook, reproduced faithfully (including the leaks)."""
    data = pd.read_csv(ERA_A_PATH)
    data["STATE/UT"] = data["STATE/UT"].str.lower()
    data["DISTRICT"] = data["DISTRICT"].str.lower()
    data = data[~data["DISTRICT"].str.contains("total", case=False)]
    data["Total Crime"] = data[ERA_A_CATEGORIES].sum(axis=1)
    pivot = data.pivot_table(index=["STATE/UT", "DISTRICT"], columns="Year", values="Total Crime")
    for start in range(2001, 2014):
        pivot[f"Change in Crime Pct {start}-{start + 1}"] = (pivot[start + 1] - pivot[start]) / pivot[start] * 100
    df = pivot.filter(like="Change in Crime Pct").reset_index().dropna()

    change_cols = sorted([c for c in df.columns if "Change in Crime Pct" in c],
                         key=lambda x: int(x.split()[-1].split("-")[0]))
    df["rolling_mean_3"] = df[change_cols].T.rolling(window=3).mean().T.iloc[:, -1]
    df["rolling_std_3"] = df[change_cols].T.rolling(window=3).std().T.iloc[:, -1]
    for prev, curr in zip(change_cols[:-1], change_cols[1:]):
        df[f"diff_{curr}"] = df[curr] - df[prev]
    df["zscore_mean"] = df[change_cols].mean(axis=1)
    df["zscore_std"] = df[change_cols].std(axis=1)
    df[change_cols] = df[change_cols].subtract(df["zscore_mean"], axis=0).divide(df["zscore_std"], axis=0)
    return df


def notebook_windows(df: pd.DataFrame):
    """Cell 7's column selection and sliding window, reproduced faithfully."""
    cols = [c for c in df.columns if "Change in Crime Pct" in c and "Predicted" not in c]
    cols = sorted(cols, key=lambda c: int(re.search(r"(\d{4})", c).group(1)))
    X, y, groups, leaked, f6_minus_f4, scale = [], [], [], [], [], []
    for _, row in df.iterrows():
        vals = row[cols].values.astype(float)
        for start in range(len(vals) - WINDOW):
            feats, label = vals[start:start + WINDOW], vals[start + WINDOW]
            if np.isnan(feats).any() or np.isnan(label) or np.isinf(feats).any() or np.isinf(label):
                continue
            X.append(feats)
            y.append(label)
            groups.append(f"{row['STATE/UT']}|{row['DISTRICT']}")
            leaked.append(cols[start + WINDOW].startswith("diff_"))
            f6_minus_f4.append(feats[5] - feats[3])
            scale.append(row["zscore_std"])
    return (np.array(X), np.array(y), np.array(groups), np.array(leaked),
            np.array(f6_minus_f4), np.array(scale), cols)


def main() -> None:
    df = notebook_final_dataframe()
    X, y, groups, leaked, f6_minus_f4, scale, cols = notebook_windows(df)
    print(f"districts after notebook dropna: {len(df)}")
    print(f"column sequence fed to the window ({len(cols)}): {cols[:7]} ...")
    print(f"windows: {len(y)}; label is a diff_ column in {leaked.sum()} ({100 * leaked.mean():.1f}%)")

    # On leaked windows the label is d_t = c_t - c_{t-1} in *raw* units, while the
    # features are z-scored per district: (f6 - f4) == d_t / zscore_std. So the
    # label is exactly (f6 - f4) * zscore_std.
    recon = f6_minus_f4[leaked] * scale[leaked]
    print(f"leaked windows: max |label - (f6 - f4) * district_std| = {np.max(np.abs(y[leaked] - recon)):.2e}")

    lr = LinearRegression().fit(f6_minus_f4[leaked].reshape(-1, 1), y[leaked])
    print(f"leaked windows: R^2 of a 1-feature linear fit on (f6 - f4) alone = "
          f"{r2_score(y[leaked], lr.predict(f6_minus_f4[leaked].reshape(-1, 1))):.3f}")

    # Reproduce the notebook's evaluation: district-grouped split, XGBoost.
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(X, y, groups))
    model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, random_state=SEED, n_jobs=-1)
    model.fit(X[tr], y[tr])
    pred = model.predict(X[te])
    print(f"XGBoost on leaky windows, grouped 80/20 split: test R^2 = {r2_score(y[te], pred):.3f}")
    print(f"   on leaked test windows only : R^2 = {r2_score(y[te][leaked[te]], pred[leaked[te]]):.3f}")
    print(f"   on clean  test windows only : R^2 = {r2_score(y[te][~leaked[te]], pred[~leaked[te]]):.3f}")


if __name__ == "__main__":
    main()
