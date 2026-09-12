"""Train the honest (leak-free) year-over-year change model and report it
against naive baselines.

Two evaluation splits are reported:

* **temporal** (primary): every window whose label year is the last available
  year is held out; the model trains on all earlier windows. This is what
  "forecast next year" actually means. A district-grouped split lets the model
  learn the held-out year's nationwide shock from the *other* districts, which
  is a subtle leak of its own.
* **grouped** (secondary): the notebook's 80/20 split by unit, kept only so
  the honest number can be placed next to the notebook's on the same footing.

Baselines are the bar the model must clear to be worth serving:
  zero         predict 0% change
  persistence  predict the previous change again
  window_mean  predict the mean of the six-year window

Run with ``python -m ml.train``. Writes ``ml/reports/<date>_era_a_baseline.md``
and ``ml/artifacts/xgb_era_a.joblib``.
"""
from __future__ import annotations

import json
from datetime import date

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor

from ml.features import FEATURE_NAMES, ROOT, load_long_era_a, make_windows, yearly_pct_change

SEED = 42
REPORTS = ROOT / "ml" / "reports"
ARTIFACTS = ROOT / "ml" / "artifacts"

# A fixed, modest configuration. Tuning is deliberately out of scope for the
# baseline: the number we want first is "what does a reasonable model do
# without leakage", not "what is the best we can squeeze out".
XGB_PARAMS = dict(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=5, random_state=SEED, n_jobs=-1)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "median_ae": float(median_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
    }


def baselines(X: np.ndarray) -> dict[str, np.ndarray]:
    f = {name: i for i, name in enumerate(FEATURE_NAMES)}
    return {
        "zero": np.zeros(len(X)),
        "persistence": X[:, f["lag_last"]],
        "window_mean": X[:, f["lag_mean"]],
    }


def evaluate(X, y, train_idx, test_idx) -> tuple[dict, XGBRegressor]:
    model = XGBRegressor(**XGB_PARAMS).fit(X[train_idx], y[train_idx])
    results = {"xgboost": metrics(y[test_idx], model.predict(X[test_idx]))}
    for name, pred in baselines(X[test_idx]).items():
        results[name] = metrics(y[test_idx], pred)
    return results, model


def fmt_table(results: dict) -> str:
    rows = ["| model | R2 | MAE | median AE | RMSE |", "|---|---|---|---|---|"]
    for name, m in results.items():
        rows.append(f"| {name} | {m['r2']:.3f} | {m['mae']:.1f} | {m['median_ae']:.1f} | {m['rmse']:.1f} |")
    return "\n".join(rows)


def main() -> None:
    long = load_long_era_a()
    changes = yearly_pct_change(long)
    X, y, meta = make_windows(changes)
    units = meta["state"] + "|" + meta["unit"]
    last_year = int(meta["label_year"].max())

    print(f"units with any data: {changes.shape[0]}; windows: {len(y)}; "
          f"label years {int(meta['label_year'].min())}-{last_year}; features: {len(FEATURE_NAMES)}")
    print(f"target: mean {y.mean():.1f}, median {np.median(y):.1f}, p1 {np.percentile(y, 1):.1f}, "
          f"p99 {np.percentile(y, 99):.1f}, max {y.max():.0f}")

    temporal_test = np.where(meta["label_year"] == last_year)[0]
    temporal_train = np.where(meta["label_year"] < last_year)[0]
    temporal, model = evaluate(X, y, temporal_train, temporal_test)

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(X, y, units))
    grouped, _ = evaluate(X, y, tr, te)

    print(f"\nTEMPORAL split (train label years < {last_year}, test = {last_year}; "
          f"{len(temporal_train)} / {len(temporal_test)} windows)")
    print(fmt_table(temporal))
    print(f"\nGROUPED split (80/20 by unit; {len(tr)} / {len(te)} windows)")
    print(fmt_table(grouped))

    REPORTS.mkdir(exist_ok=True)
    ARTIFACTS.mkdir(exist_ok=True)
    stamp = date.today().isoformat()
    report = REPORTS / f"{stamp}_era_a_baseline.md"
    report.write_text("\n".join([
        f"# Era A (2001-2014) honest baseline: {stamp}",
        "",
        "Leak-free pipeline (`ml/features.py`, `ml/train.py`) on the 2001-2014 file only.",
        f"Units: {changes.shape[0]}. Windows: {len(y)}. Features: {len(FEATURE_NAMES)} ({', '.join(FEATURE_NAMES)}).",
        f"Target distribution: median {np.median(y):.1f}%, p1 {np.percentile(y, 1):.1f}%, "
        f"p99 {np.percentile(y, 99):.1f}%, max {y.max():.0f}%.",
        "",
        f"## Temporal split: train on label years < {last_year}, test on {last_year} ({len(temporal_test)} windows)",
        "", fmt_table(temporal), "",
        f"## Grouped split: 80/20 by unit, seed {SEED} ({len(te)} test windows)",
        "", fmt_table(grouped), "",
        "## XGBoost params", "", "```json", json.dumps(XGB_PARAMS, indent=2), "```", "",
    ]), encoding="utf-8")
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES, "trained_on": f"era_a_label_years<{last_year}"},
                ARTIFACTS / "xgb_era_a.joblib")
    print(f"\nreport -> {report}\nmodel  -> {ARTIFACTS / 'xgb_era_a.joblib'}")


if __name__ == "__main__":
    main()
