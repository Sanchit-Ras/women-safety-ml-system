"""The forecasting bake-off on the re-based panel: can next year's change in a unit's crime be forecast, and by
which method, measured the honest way.

Series: stable units (ml/panel.py). Target: next year's log change of the unit's total, log(T[t+1] / T[t]),
scored in percent. Test: rolling origin, one round per target year, each trained only on windows whose label
year is earlier; settings are chosen on VALIDATION_YEARS alone, the headline is TEST_YEARS, and SHOCK_YEARS are
shown and also left out. Contenders: four baselines, credibility shrinkage, ridge, XGBoost on the log change,
Poisson XGBoost on counts, reconciliation of the better tree model with the state and national levels, the plain
average of the three, a stacked blend (reported only when it beats the average on validation), and a hotspot
classifier scored on precision at 10%. A win over the best baseline counts only when a paired bootstrap over
units puts the whole 95% interval of the MAE difference below zero.

    python -m ml.bakeoff [--out ml/reports]      # writes <date>_bakeoff.md and .json, saves the recommended model
"""
from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from agents.schemas import CORE_CATEGORIES
from ml import panel as unit_panel

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "ml" / "reports"
ARTIFACTS = ROOT / "ml" / "artifacts"
WINDOW = 6
VALIDATION_YEARS = (2016, 2017, 2018)
TEST_YEARS = (2019, 2020, 2021, 2022)
SHOCK_YEARS = (2020, 2021)
SEED = 42
BOOTSTRAPS = 1000
MIX = [f"mix_{c}" for c in CORE_CATEGORIES]
FEATURES = ([f"lag_{i}" for i in range(1, WINDOW + 1)] + [f"diff_{i}" for i in range(1, WINDOW)]
            + ["lag_mean", "lag_std", "lag_slope", "log_size", "state_last", "nat_last", "nbr_last", "hist_mean",
               "outside_share", *MIX])
XGB_GRID = [{"max_depth": d, "n_estimators": n} for d in (3, 4) for n in (300, 600)]
XGB_FIXED = {"learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5, "random_state": SEED, "n_jobs": -1}
CREDIBILITY_GRID = (10, 30, 100, 300, 1000)
RIDGE_GRID = (1.0, 10.0, 100.0)
BASELINES = ("zero", "persistence", "window_mean", "state_trend")
MAX_OUTSIDE = 0.5     # a window is used only if no year it touches had more than half its total estimated across the unit's boundary


# ---- windows ---------------------------------------------------------------------------------------------------

def log_changes(wide: pd.DataFrame) -> pd.DataFrame:
    """c[t] = log(T[t] / T[t-1]) along the columns (years); NaN where either total is missing or zero."""
    prev = wide.shift(1, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.log(wide / prev)
    return c.where((wide > 0) & (prev > 0))


def windows(panel: pd.DataFrame, neighbours: dict[str, list[str]], window: int = WINDOW) -> pd.DataFrame:
    """One row per unit and label year L: features from the years before L, and the target, the log change into L.
    lag_1 is the change into L-1 (the most recent) and lag_k the change into L-k; diff_k and the window stats are on
    the window oldest first. A row needs all ``window`` lags and the target defined (no zero or missing total)."""
    wide = panel.pivot(index="unit_id", columns="year", values="total").sort_index(axis=1)
    years = [int(y) for y in wide.columns]
    c = log_changes(wide)
    state_of = panel.drop_duplicates("unit_id").set_index("unit_id")["state_code"].to_dict()
    state_c = log_changes(panel.groupby(["state_code", "year"])["total"].sum().unstack("year").reindex(columns=wide.columns))
    nat_c = log_changes(panel.groupby("year")["total"].sum().to_frame("nat").T.reindex(columns=wide.columns)).iloc[0].to_numpy(dtype=float)
    carr = {u: c.loc[u].to_numpy(dtype=float) for u in wide.index}
    tarr = {u: wide.loc[u].to_numpy(dtype=float) for u in wide.index}
    sarr = {s: state_c.loc[s].to_numpy(dtype=float) for s in state_c.index}
    oarr = {u: v for u, v in panel.pivot(index="unit_id", columns="year", values="outside_share").reindex(columns=wide.columns).iterrows()}
    marr = {cat: panel.pivot(index="unit_id", columns="year", values=cat).reindex(columns=wide.columns) for cat in CORE_CATEGORIES}
    rows = []
    for u in wide.index:
        cu, tu, sc = carr[u], tarr[u], sarr[state_of[u]]
        nb = [carr[v] for v in neighbours.get(u, []) if v in carr]
        out_u = oarr[u].to_numpy(dtype=float)
        mix_u = {cat: marr[cat].loc[u].to_numpy(dtype=float) for cat in CORE_CATEGORIES}
        for i in range(window + 1, len(years)):                  # i: position of the label year; lags at i-window..i-1
            x = cu[i - window:i]                                 # oldest first
            if np.isnan(x).any() or np.isnan(cu[i]):
                continue
            prev = tu[i - 1]
            nbr = [v[i - 1] for v in nb if not np.isnan(v[i - 1])]
            row = {"unit_id": u, "state_code": state_of[u], "label_year": years[i], "count_prev": prev, "count_next": tu[i], "y": cu[i]}
            row.update({f"lag_{k}": x[-k] for k in range(1, window + 1)})
            row.update({f"diff_{k}": x[k] - x[k - 1] for k in range(1, window)})
            touched = out_u[i - window - 1:i + 1]                   # the totals behind the lags, and the target's two
            row.update({"lag_mean": float(x.mean()), "lag_std": float(x.std()), "lag_slope": float(np.polyfit(np.arange(window), x, 1)[0]),
                        "log_size": float(np.log1p(prev)), "state_last": float(sc[i - 1]), "nat_last": float(nat_c[i - 1]),
                        "nbr_last": float(np.mean(nbr)) if nbr else np.nan, "hist_mean": float(np.nanmean(cu[1:i])),
                        "outside_share": 0.0 if np.isnan(out_u[i - 1]) else float(out_u[i - 1]),
                        "outside_max": 0.0 if np.isnan(touched).all() else float(np.nanmax(touched))})
            row.update({f"mix_{cat}": float(mix_u[cat][i - 1]) / prev for cat in CORE_CATEGORIES})
            rows.append(row)
    return pd.DataFrame(rows)


def usable(win: pd.DataFrame, max_outside: float = MAX_OUTSIDE) -> pd.DataFrame:
    """Windows whose unit was mostly real in every year they touch. A unit that was not yet separate is a fixed share
    of its parent: its 'changes' are the parent's, so they would count twice and a forecast would be graded against an
    estimate. Such windows are left out of training and of the test."""
    return win[win["outside_max"] <= max_outside].reset_index(drop=True)


def rounds(win: pd.DataFrame, years: tuple[int, ...] = VALIDATION_YEARS + TEST_YEARS) -> dict[int, tuple[pd.DataFrame, pd.DataFrame]]:
    """Rolling origin: for each target year, train = windows whose label year is earlier, test = that year's."""
    return {L: (win[win["label_year"] < L].reset_index(drop=True), win[win["label_year"] == L].reset_index(drop=True)) for L in years}


# ---- contenders ---------------------------------------------------------------------------------------------------

def baselines(test: pd.DataFrame) -> dict[str, np.ndarray]:
    return {"zero": np.zeros(len(test)), "persistence": test["lag_1"].to_numpy(dtype=float),
            "window_mean": test["lag_mean"].to_numpy(dtype=float), "state_trend": test["state_last"].fillna(0.0).to_numpy(dtype=float)}


def credibility(test: pd.DataFrame, K: float) -> np.ndarray:
    """z * the unit's own window mean + (1 - z) * its state's last change, z = n / (n + K) with n the unit's size in
    cases: a big district trusts its own history, a small one leans on its state (Buhlmann credibility)."""
    n = test["count_prev"].to_numpy(dtype=float)
    z = n / (n + K)
    return z * test["lag_mean"].to_numpy(dtype=float) + (1 - z) * test["state_last"].fillna(0.0).to_numpy(dtype=float)


def ridge(train: pd.DataFrame, test: pd.DataFrame, alpha: float):
    model = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(), Ridge(alpha=alpha))
    return model.fit(train[FEATURES], train["y"]), model.predict(test[FEATURES])


def xgb(train: pd.DataFrame, test: pd.DataFrame, params: dict):
    model = XGBRegressor(**XGB_FIXED, **params).fit(train[FEATURES], train["y"])
    return model, model.predict(test[FEATURES])


def poisson(train: pd.DataFrame, test: pd.DataFrame, params: dict):
    """Next year's count with a Poisson loss, so a district's weight in training follows its size; the forecast is
    turned back into a log change for scoring."""
    cols = FEATURES + ["count_prev"]
    model = XGBRegressor(objective="count:poisson", **XGB_FIXED, **params).fit(train[cols], train["count_next"])
    pred = np.clip(model.predict(test[cols]), 0.5, None)
    return model, np.log(pred / test["count_prev"].to_numpy(dtype=float))


def top_decile(df: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    """True for the rows in the top 10% of ``values`` within their label year."""
    flag = np.zeros(len(df), dtype=bool)
    values = np.asarray(values, dtype=float)
    for idx in df.groupby("label_year").indices.values():
        k = max(1, int(np.ceil(0.1 * len(idx))))
        flag[idx[np.argsort(-values[idx], kind="stable")[:k]]] = True
    return flag


def hotspot(train: pd.DataFrame, test: pd.DataFrame, params: dict) -> np.ndarray:
    """A classifier for 'in the top 10% of growth next year'; its score is the probability, ranked for precision at 10%."""
    label = top_decile(train, train["y"].to_numpy()).astype(int)
    model = XGBClassifier(**XGB_FIXED, **params, eval_metric="logloss").fit(train[FEATURES], label)
    return model.predict_proba(test[FEATURES])[:, 1]


def reconcile(bottom: np.ndarray, groups: np.ndarray, group_pred: dict, nat_pred: float, var: dict[str, float]) -> np.ndarray:
    """Bottom forecasts (counts) adjusted so they add up to the state and national forecasts, each level weighted by
    its reliability: weighted least squares with a diagonal W whose entries are the level's relative error variance
    times the forecast squared (the MinT family with variance scaling)."""
    states = sorted(set(groups))
    nb, ns = len(bottom), len(states)
    S = np.zeros((1 + ns + nb, nb))
    S[0, :] = 1.0
    for j, s in enumerate(states):
        S[1 + j, groups == s] = 1.0
    S[1 + ns:, :] = np.eye(nb)
    yhat = np.concatenate([[nat_pred], [group_pred[s] for s in states], np.asarray(bottom, dtype=float)])
    rel = np.concatenate([[var["nation"]], np.full(ns, var["state"]), np.full(nb, var["unit"])])
    w_inv = 1.0 / np.maximum(rel * np.maximum(yhat, 1.0) ** 2, 1e-9)
    P = np.linalg.solve(S.T @ (w_inv[:, None] * S), S.T * w_inv[None, :])
    return P @ yhat


# ---- scores -------------------------------------------------------------------------------------------------------

def pct(c) -> np.ndarray:
    return (np.exp(np.asarray(c, dtype=float)) - 1.0) * 100.0


def precision_at_10(df: pd.DataFrame, scores: np.ndarray) -> float:
    flagged, actual = top_decile(df, scores), top_decile(df, df["y"].to_numpy())
    return float((flagged & actual).sum() / flagged.sum()) if flagged.sum() else float("nan")


def score(df: pd.DataFrame, pred: np.ndarray, mae_zero: float | None = None) -> dict:
    y, p = pct(df["y"]), pct(pred)
    err = np.abs(p - y)
    forecast = df["count_prev"].to_numpy(dtype=float) * np.exp(np.asarray(pred, dtype=float))
    nxt = df["count_next"].to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                        # a constant prediction has no rank correlation
        rho = float(spearmanr(p, y).statistic)
    out = {"n": int(len(df)), "mae": float(err.mean()), "median_ae": float(np.median(err)), "rmse": float(np.sqrt((err ** 2).mean())),
           "r2": float(r2_score(y, p)), "wape": float(np.abs(forecast - nxt).sum() / nxt.sum()), "spearman": rho,
           "precision_at_10": precision_at_10(df, np.asarray(pred))}
    out["skill"] = None if mae_zero is None else float(1 - out["mae"] / mae_zero)
    return out


def bootstrap_mae_diff(df: pd.DataFrame, pred: np.ndarray, base: np.ndarray, n_boot: int = BOOTSTRAPS, seed: int = SEED) -> list[float]:
    """95% interval of MAE(pred) - MAE(base) when units are resampled with replacement: a paired comparison on the
    same rows, so the noise that both share cancels. Below zero throughout = a real win."""
    y = pct(df["y"])
    e1, e0 = np.abs(pct(pred) - y), np.abs(pct(base) - y)
    _codes, inv = np.unique(df["unit_id"].to_numpy(), return_inverse=True)
    s1, s0, n = np.bincount(inv, weights=e1), np.bincount(inv, weights=e0), np.bincount(inv).astype(float)
    draws = np.random.default_rng(seed).integers(0, len(n), size=(n_boot, len(n)))
    diff = (s1[draws].sum(axis=1) - s0[draws].sum(axis=1)) / n[draws].sum(axis=1)
    return [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]


def bootstrap_precision(df: pd.DataFrame, scores: np.ndarray, n_boot: int = BOOTSTRAPS, seed: int = SEED) -> list[float]:
    """95% interval of precision at 10% when units are resampled with replacement (a unit's years travel together)."""
    scores = np.asarray(scores, dtype=float)
    codes, inv = np.unique(df["unit_id"].to_numpy(), return_inverse=True)
    rows_of = [np.flatnonzero(inv == k) for k in range(len(codes))]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        idx = np.concatenate([rows_of[k] for k in rng.integers(0, len(codes), size=len(codes))])
        out.append(precision_at_10(df.iloc[idx].reset_index(drop=True), scores[idx]))
    return [float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))]


# ---- the bake-off -------------------------------------------------------------------------------------------------

def _pooled(R: dict, preds: dict[int, np.ndarray], years) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.concat([R[L][1] for L in years], ignore_index=True)
    return df, np.concatenate([np.asarray(preds[L], dtype=float) for L in years])


def _mae(R, preds, years) -> float:
    df, p = _pooled(R, preds, years)
    return float(np.abs(pct(p) - pct(df["y"])).mean())


def level_series(panel: pd.DataFrame, by: str | None) -> pd.DataFrame:
    """The panel summed to states (``by = "state_code"``) or the nation (None), shaped like a unit panel."""
    keys = ([by] if by else []) + ["year"]
    g = panel.groupby(keys, as_index=False)[[*CORE_CATEGORIES, "total"]].sum()
    g["unit_id"] = g[by].map(lambda s: f"level_{s}") if by else "nation"
    g["state_code"] = g[by] if by else 0
    g["n_members"], g["outside_est"], g["outside_share"] = 1, 0.0, 0.0
    return g[unit_panel.COLUMNS]


def run(panel: pd.DataFrame, neighbours: dict[str, list[str]]) -> dict:
    every = windows(panel, neighbours)
    win = usable(every)
    R = rounds(win)
    years = VALIDATION_YEARS + TEST_YEARS
    preds: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    models: dict[str, dict[int, object]] = defaultdict(dict)
    for L, (tr, te) in R.items():
        for name, p in baselines(te).items():
            preds[name][L] = p
        for K in CREDIBILITY_GRID:
            preds[f"credibility K={K}"][L] = credibility(te, K)
        for a in RIDGE_GRID:
            models[f"ridge alpha={a:g}"][L], preds[f"ridge alpha={a:g}"][L] = ridge(tr, te, a)
        for g in XGB_GRID:
            tag = f"depth {g['max_depth']}, {g['n_estimators']} trees"
            models[f"xgb {tag}"][L], preds[f"xgb {tag}"][L] = xgb(tr, te, g)
            models[f"poisson {tag}"][L], preds[f"poisson {tag}"][L] = poisson(tr, te, g)
            preds[f"hotspot {tag}"][L] = hotspot(tr, te, g)

    # settings chosen on the validation years only
    def pick(prefix: str, key=lambda name: _mae(R, preds[name], VALIDATION_YEARS)) -> str:
        return min((n for n in preds if n.startswith(prefix)), key=key)

    chosen = {"credibility": pick("credibility"), "ridge": pick("ridge"), "xgb": pick("xgb"), "poisson": pick("poisson"),
              "hotspot": pick("hotspot", key=lambda name: -np.mean([precision_at_10(R[L][1], preds[name][L]) for L in VALIDATION_YEARS]))}
    best_baseline = min(BASELINES, key=lambda name: _mae(R, preds[name], VALIDATION_YEARS))
    for family, name in chosen.items():
        preds[family] = preds[name]
        if name in models:
            models[family] = models[name]
    tree = "poisson" if _mae(R, preds["poisson"], VALIDATION_YEARS) < _mae(R, preds["xgb"], VALIDATION_YEARS) else "xgb"

    # reconciliation of the better tree model with the state and national levels
    state_win, nat_win = windows(level_series(panel, "state_code"), {}), windows(level_series(panel, None), {})
    state_R, nat_R = rounds(state_win), rounds(nat_win)
    g = XGB_GRID[[f"depth {x['max_depth']}, {x['n_estimators']} trees" in chosen["xgb"] for x in XGB_GRID].index(True)]
    state_pred = {L: xgb(state_R[L][0], state_R[L][1], g)[1] for L in years}
    nat_pred = {L: nat_R[L][1]["lag_mean"].to_numpy(dtype=float) for L in years}

    def rel_var(frames, p) -> float:                        # mean squared relative error of the count forecast
        df = pd.concat([frames[L][1] for L in VALIDATION_YEARS], ignore_index=True)
        f = df["count_prev"].to_numpy() * np.exp(np.concatenate([p[L] for L in VALIDATION_YEARS]))
        return float(np.mean(((f - df["count_next"].to_numpy()) / np.maximum(df["count_next"].to_numpy(), 1.0)) ** 2))

    var = {"unit": rel_var(R, preds[tree]), "state": rel_var(state_R, state_pred), "nation": rel_var(nat_R, nat_pred)}
    for L in years:
        te, ste, nte = R[L][1], state_R[L][1], nat_R[L][1]
        bottom = te["count_prev"].to_numpy() * np.exp(preds[tree][L])
        sp = {int(s): float(c * np.exp(p)) for s, c, p in zip(ste["state_code"], ste["count_prev"], state_pred[L])}
        for s in set(te["state_code"]) - set(sp):              # a state without its own window: its units' sum, untouched
            m = te["state_code"] == s
            sp[int(s)] = float(bottom[m].sum())
        npred = float(nte["count_prev"].iloc[0] * np.exp(nat_pred[L][0])) if len(nte) else float(bottom.sum())
        rec = np.clip(reconcile(bottom, te["state_code"].to_numpy(), sp, npred, var), 0.5, None)
        preds["reconciled"][L] = np.log(rec / te["count_prev"].to_numpy())

    # ensembles
    parts = ("credibility", "xgb", "poisson")
    for L in years:
        preds["ensemble"][L] = np.mean([preds[n][L] for n in parts], axis=0)
    vdf, _ = _pooled(R, preds["zero"], VALIDATION_YEARS)
    Xv = np.column_stack([np.concatenate([preds[n][L] for L in VALIDATION_YEARS]) for n in parts])
    stack = Ridge(alpha=1.0, positive=True).fit(Xv, vdf["y"].to_numpy())
    for L in years:
        preds["stacked"][L] = stack.predict(np.column_stack([preds[n][L] for n in parts]))
    stacked_wins = _mae(R, preds["stacked"], VALIDATION_YEARS) < _mae(R, preds["ensemble"], VALIDATION_YEARS)

    # scores
    order = [*BASELINES, "credibility", "ridge", "xgb", "poisson", "reconciled", "ensemble", "stacked"]
    mae_zero = {}
    tables = {}
    for label, yrs in {"validation": VALIDATION_YEARS, "test": TEST_YEARS, "test_no_shock": tuple(y for y in TEST_YEARS if y not in SHOCK_YEARS)}.items():
        df, z = _pooled(R, preds["zero"], yrs)
        mae_zero[label] = float(np.abs(pct(z) - pct(df["y"])).mean())
        tables[label] = {name: score(df, _pooled(R, preds[name], yrs)[1], mae_zero[label]) for name in order}
    by_year = {name: {L: _mae(R, preds[name], (L,)) for L in years} for name in order}
    tdf, tbase = _pooled(R, preds[best_baseline], TEST_YEARS)
    intervals = {name: bootstrap_mae_diff(tdf, _pooled(R, preds[name], TEST_YEARS)[1], tbase) for name in order if name not in BASELINES}
    no_shock = tuple(y for y in TEST_YEARS if y not in SHOCK_YEARS)
    ndf, nbase = _pooled(R, preds[best_baseline], no_shock)
    intervals_no_shock = {name: bootstrap_mae_diff(ndf, _pooled(R, preds[name], no_shock)[1], nbase) for name in order if name not in BASELINES}
    rank_names = ("hotspot", "xgb", "poisson", "ridge", "reconciled", "ensemble", "stacked", "credibility", "window_mean", "persistence")
    vdf_all, _ = _pooled(R, preds["zero"], VALIDATION_YEARS)
    ranking = {name: {"validation": precision_at_10(vdf_all, _pooled(R, preds[name], VALIDATION_YEARS)[1]),
                      "test": precision_at_10(tdf, _pooled(R, preds[name], TEST_YEARS)[1]),
                      "test_ci95": bootstrap_precision(tdf, _pooled(R, preds[name], TEST_YEARS)[1])} for name in rank_names}
    candidates = [n for n in ("credibility", "ridge", "xgb", "poisson", "reconciled", "ensemble")] + (["stacked"] if stacked_wins else [])
    recommended = min(candidates, key=lambda n: _mae(R, preds[n], VALIDATION_YEARS))
    ranking_recommended = max((n for n in rank_names if n not in BASELINES), key=lambda n: ranking[n]["validation"])
    return {"date": date.today().isoformat(), "units": int(win["unit_id"].nunique()), "windows": int(len(win)),
            "windows_left_out_estimated": int(len(every) - len(win)), "max_outside": MAX_OUTSIDE,
            "label_years": [int(win["label_year"].min()), int(win["label_year"].max())],
            "windows_per_year": {int(k): int(v) for k, v in win.groupby("label_year").size().items()},
            "target_pct": {"median": float(np.median(pct(win["y"]))), "p1": float(np.percentile(pct(win["y"]), 1)),
                           "p99": float(np.percentile(pct(win["y"]), 99)), "max": float(pct(win["y"]).max())},
            "outside": {"windows_over_5pct": int((win["outside_share"] > 0.05).sum()), "windows_over_20pct": int((win["outside_share"] > 0.2).sum())},
            "features": FEATURES, "window": WINDOW, "validation_years": list(VALIDATION_YEARS), "test_years": list(TEST_YEARS),
            "shock_years": list(SHOCK_YEARS), "chosen": chosen, "best_baseline": best_baseline, "tree_for_reconciliation": tree,
            "reconciliation_variances": var, "stacked_weights": {n: float(w) for n, w in zip(parts, stack.coef_)},
            "stacked_beats_average_on_validation": bool(stacked_wins), "mae_zero": mae_zero, "scores": tables, "mae_by_year": by_year,
            "mae_diff_vs_best_baseline_ci95": intervals, "mae_diff_vs_best_baseline_ci95_no_shock": intervals_no_shock,
            "ranking": ranking, "recommended": recommended, "ranking_recommended": ranking_recommended,
            "xgb_fixed": XGB_FIXED, "bootstraps": BOOTSTRAPS, "_models": models, "_windows": win}


# ---- report -----------------------------------------------------------------------------------------------------

def _f(x, nd=1) -> str:
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def render(m: dict) -> str:
    years = m["validation_years"] + m["test_years"]
    t, v, ci = m["scores"]["test"], m["scores"]["validation"], m["mae_diff_vs_best_baseline_ci95"]
    L = [f"# Forecasting bake-off ({m['date']})", "",
         f"{m['units']} stable units, {m['windows']:,} windows with label years {m['label_years'][0]}-{m['label_years'][1]}, "
         f"{len(m['features'])} features from the {m['window']} changes before each label year. Target: next year's change of the "
         f"unit's total, scored in percent (median {_f(m['target_pct']['median'])}%, p1 {_f(m['target_pct']['p1'])}%, "
         f"p99 {_f(m['target_pct']['p99'])}%). {m['windows_left_out_estimated']:,} further windows were left out because a year "
         f"they touch had more than {m['max_outside']:.0%} of the unit's total estimated across its boundary (a unit not yet separate "
         f"is a fixed share of its parent: its changes are the parent's, and a forecast would be graded against an estimate). "
         f"Among the windows kept, the unit carried an estimate across its boundary in {m['outside']['windows_over_5pct']:,} "
         f"(over 5% of the total) and {m['outside']['windows_over_20pct']:,} (over 20%).", "",
         f"Rolling origin: each target year is forecast by a model trained only on earlier label years. Settings were chosen on "
         f"{', '.join(map(str, m['validation_years']))} alone; the headline is {', '.join(map(str, m['test_years']))}; "
         f"{' and '.join(map(str, m['shock_years']))} are nationwide shock years (lockdown and rebound), shown and also left out.", "",
         "## Settings chosen on the validation years", "", "| contender | setting | validation MAE |", "|---|---|---|"]
    for family, name in m["chosen"].items():
        L.append(f"| {family} | {name} | {_f(v[family]['mae']) if family in v else 'precision at 10%'} |")
    L += [f"| best baseline | {m['best_baseline']} | {_f(v[m['best_baseline']]['mae'])} |",
          f"| reconciliation | on {m['tree_for_reconciliation']}; relative error variances unit {m['reconciliation_variances']['unit']:.3f}, "
          f"state {m['reconciliation_variances']['state']:.3f}, nation {m['reconciliation_variances']['nation']:.3f} | {_f(v['reconciled']['mae'])} |",
          f"| stacked blend | weights {', '.join(f'{k} {w:.2f}' for k, w in m['stacked_weights'].items())}; "
          f"{'beats' if m['stacked_beats_average_on_validation'] else 'does not beat'} the plain average on validation | {_f(v['stacked']['mae'])} |",
          "", f"## Headline: test years {', '.join(map(str, m['test_years']))} ({t['zero']['n']:,} forecasts)", "",
          "| contender | MAE (pct points) | skill vs no change | R2 | WAPE on counts | Spearman | precision at 10% | MAE minus best baseline, 95% interval | real win? |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, s in t.items():
        iv = ci.get(name)
        verdict = "-" if iv is None else ("**yes**" if iv[1] < 0 else "no")
        L.append(f"| {name} | {_f(s['mae'])} | {_f(s['skill'], 3)} | {_f(s['r2'], 3)} | {_f(s['wape'], 3)} | {_f(s['spearman'], 3)} | "
                 f"{_f(s['precision_at_10'], 3)} | {'-' if iv is None else f'{iv[0]:+.1f} to {iv[1]:+.1f}'} | {verdict} |")
    ns = m["scores"]["test_no_shock"]
    L += ["", f"Skill = 1 - MAE / MAE of 'no change' ({_f(m['mae_zero']['test'])} on the test years). A real win needs the whole "
              "interval below zero: the interval is from 1,000 resamples of units, paired on the same forecasts.",
          "", "## MAE by year", "", "| contender | " + " | ".join(f"{y}{'*' if y in m['shock_years'] else ''}" for y in years) + " | test without shocks |",
          "|---|" + "---|" * (len(years) + 1)]
    for name, row in m["mae_by_year"].items():
        L.append(f"| {name} | " + " | ".join(_f(row[y]) for y in years) + f" | {_f(ns[name]['mae'])} |")
    cins = m["mae_diff_vs_best_baseline_ci95_no_shock"]
    L += ["", "\\* shock year.", "", f"Without the shock years, MAE minus the best baseline ({m['best_baseline']}, {_f(ns[m['best_baseline']]['mae'])}), "
          "95% interval: " + "; ".join(f"{name} {iv[0]:+.1f} to {iv[1]:+.1f}{' (real)' if iv[1] < 0 else ''}" for name, iv in cins.items()) + ".",
          "", "## Hotspots: precision at 10%", "",
          "The app shows tiers, so this asks: of the units a method puts in the top 10% of growth, how many were? Picking at random "
          "scores 0.100. The ranking method is chosen on the validation years; the interval resamples units on the test years.", "",
          "| ranked by | validation | test | test 95% interval | beats chance? |", "|---|---|---|---|---|"]
    for name, r in m["ranking"].items():
        L.append(f"| {name} | {_f(r['validation'], 3)} | {_f(r['test'], 3)} | {_f(r['test_ci95'][0], 3)} to {_f(r['test_ci95'][1], 3)} | "
                 f"{'**yes**' if r['test_ci95'][0] > 0.1 else 'no'} |")
    rec, s = m["recommended"], t[m["recommended"]]
    iv, rr = ci[rec], m["ranking_recommended"]
    rk = m["ranking"][rr]
    L += ["", "## Reading", "",
          f"**Size of next year's change.** By the rule (lowest validation MAE) the point-forecast contender is **{rec}**. On the test "
          f"years it scores MAE {_f(s['mae'])} against {_f(t[m['best_baseline']]['mae'])} for the best baseline ({m['best_baseline']}): "
          f"skill {_f(s['skill'], 3)}, R2 {_f(s['r2'], 3)}, MAE difference {iv[0]:+.1f} to {iv[1]:+.1f} points, "
          + ("a real win." if iv[1] < 0 else "not a win beyond the noise."),
          f"**Which units will be in the top tenth of growth.** By the rule (highest validation precision at 10%) the ranking contender "
          f"is **{rr}**: on the test years {_f(rk['test'], 3)} of the units it flags were in the top tenth, interval "
          f"{_f(rk['test_ci95'][0], 3)} to {_f(rk['test_ci95'][1], 3)}, against 0.100 by chance: "
          + ("a real signal." if rk["test_ci95"][0] > 0.1 else "not beyond chance."),
          "", "## Fixed settings", "", "```json", json.dumps({"xgb_fixed": m["xgb_fixed"], "xgb_grid": XGB_GRID, "credibility_K_grid": list(CREDIBILITY_GRID),
                                                            "ridge_alpha_grid": list(RIDGE_GRID), "bootstraps": m["bootstraps"]}, indent=1), "```", ""]
    return "\n".join(L)


def write(m: dict, out_dir: Path = REPORTS) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md, js = out_dir / f"{m['date']}_bakeoff.md", out_dir / f"{m['date']}_bakeoff.json"
    md.write_text(render(m), encoding="utf-8")
    js.write_text(json.dumps({k: v for k, v in m.items() if not k.startswith("_")}, indent=1, default=str), encoding="utf-8")
    return md, js


def save_recommended(m: dict, path: Path = ARTIFACTS / "bakeoff_recommended.joblib") -> Path | None:
    """The recommended contender refit on every window (label years up to the last), for Part 3 to serve from.
    Credibility and the ensembles have no single fitted object; they are recorded by name and settings."""
    rec, win = m["recommended"], m["_windows"]
    meta = {"contender": rec, "ranking_contender": m["ranking_recommended"], "chosen": m["chosen"], "features": FEATURES,
            "report": m["date"], "stacked_weights": m["stacked_weights"]}
    fit = {"xgb": lambda g: xgb(win, win.head(1), g)[0], "poisson": lambda g: poisson(win, win.head(1), g)[0]}
    objects = {}
    for family in ("xgb", "poisson"):
        tag = m["chosen"][family]
        g = XGB_GRID[[f"depth {x['max_depth']}, {x['n_estimators']} trees" in tag for x in XGB_GRID].index(True)]
        objects[family] = fit[family](g)
    objects["ridge"] = ridge(win, win.head(1), float(m["chosen"]["ridge"].split("=")[1]))[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"meta": meta, "models": objects}, path)
    return path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run the forecasting bake-off on the re-based panel and write the report.")
    ap.add_argument("--out", default=str(REPORTS))
    args = ap.parse_args(argv)
    panel, neighbours = unit_panel.build()
    m = run(panel, neighbours)
    md, js = write(m, Path(args.out))
    saved = save_recommended(m)
    print(render(m))
    print(f"report -> {md}\njson   -> {js}\nmodels -> {saved}")


if __name__ == "__main__":
    main()
