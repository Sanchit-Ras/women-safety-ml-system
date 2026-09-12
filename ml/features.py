"""Leak-free feature construction for the year-over-year crime-change model.

The model's job: given a reporting unit's percent change in total crime for
six consecutive years, predict the percent change in the seventh. Everything
a window's features are computed from must lie *inside* that window, so that
no value from the label year, or any later year, can influence a feature.

Concretely, for a window of raw changes ``x[0..5]`` (oldest to newest) and
label ``x[6]``:

    lag_1..lag_6   the six raw changes, oldest first
    diff_1..diff_5 x[i+1] - x[i] for i in 0..4   (second derivative, in-window)
    lag_mean, lag_std, lag_last, lag_slope         summary stats, in-window

No column-wise or row-wise scaling is applied: tree models do not need it,
and the notebook's per-district z-score used every year of the row (including
future labels) to scale the features, which is temporal leakage.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_ingestion.normalize import norm, norm_state

ROOT = Path(__file__).resolve().parents[1]
ERA_A_PATH = ROOT / "data" / "raw" / "crimes_against_women_2001-2014.csv"

ERA_A_CATEGORIES = [
    "Rape",
    "Kidnapping and Abduction",
    "Dowry Deaths",
    "Assault on women with intent to outrage her modesty",
    "Insult to modesty of Women",
    "Cruelty by Husband or his Relatives",
    "Importation of Girls",
]

WINDOW = 6
FEATURE_NAMES: list[str] = (
    [f"lag_{i}" for i in range(1, WINDOW + 1)]
    + [f"diff_{i}" for i in range(1, WINDOW)]
    + ["lag_mean", "lag_std", "lag_last", "lag_slope"]
)


def load_long_era_a(path: Path = ERA_A_PATH) -> pd.DataFrame:
    """Era A as a long table: one row per (state, unit, year) with ``total``.

    - TOTAL rows (state aggregates) are dropped.
    - State spellings that vary across years are unified via ``norm_state``.
    - The single duplicated (state, unit, year) key is summed, on the grounds
      that two rows for the same police unit are two partial counts.
    """
    df = pd.read_csv(path)
    df["state"] = df["STATE/UT"].map(norm_state)
    df["unit"] = df["DISTRICT"].map(norm)
    df = df[~df["unit"].str.contains("total")]
    df["total"] = df[ERA_A_CATEGORIES].sum(axis=1)
    return (df.groupby(["state", "unit", "Year"], as_index=False)["total"].sum()
              .rename(columns={"Year": "year"}))


def yearly_pct_change(long: pd.DataFrame) -> pd.DataFrame:
    """Wide table: index (state, unit), one column per change year ``t`` holding
    the percent change from ``t-1`` to ``t``. A change from a zero base is
    undefined and stored as NaN, never as 0 or infinity."""
    wide = long.pivot(index=["state", "unit"], columns="year", values="total").sort_index(axis=1)
    years = list(wide.columns)
    out = {}
    for prev, curr in zip(years[:-1], years[1:]):
        base = wide[prev].where(wide[prev] > 0)          # zero base -> NaN
        out[curr] = (wide[curr] - base) / base * 100.0
    return pd.DataFrame(out, index=wide.index)


def window_features(x: np.ndarray) -> np.ndarray:
    """Features for one window of ``WINDOW`` raw changes, oldest first."""
    diffs = np.diff(x)
    slope = np.polyfit(np.arange(len(x)), x, 1)[0]
    return np.concatenate([x, diffs, [x.mean(), x.std(ddof=0), x[-1], slope]])


def make_windows(changes: pd.DataFrame, window: int = WINDOW):
    """Slide over each unit's change series. Returns ``X``, ``y`` and a meta
    frame with the unit key and the label year, so callers can split by time
    or by unit. Windows containing any NaN (missing year or zero base) are
    skipped rather than imputed."""
    years = list(changes.columns)
    X, y, meta = [], [], []
    for (state, unit), row in changes.iterrows():
        vals = row.values.astype(float)
        for start in range(len(vals) - window):
            chunk, label = vals[start:start + window], vals[start + window]
            if np.isnan(chunk).any() or np.isnan(label):
                continue
            X.append(window_features(chunk))
            y.append(label)
            meta.append({"state": state, "unit": unit, "label_year": years[start + window]})
    return np.array(X), np.array(y), pd.DataFrame(meta)


def latest_window_features(changes: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """The most recent complete window per unit, which is what the API serves from."""
    rows = []
    for (state, unit), row in changes.iterrows():
        vals = row.values.astype(float)[-window:]
        if len(vals) < window or np.isnan(vals).any():
            continue
        rows.append({"state": state, "unit": unit, **dict(zip(FEATURE_NAMES, window_features(vals)))})
    return pd.DataFrame(rows)
