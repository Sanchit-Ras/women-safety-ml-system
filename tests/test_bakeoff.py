"""The bake-off's machinery on hand-built series: leak-free windows, the rolling split, the credibility formula,
reconciliation adding up, precision at 10%, the scores and the paired bootstrap."""
import numpy as np
import pandas as pd
import pytest

from agents.schemas import CORE_CATEGORIES
from ml import bakeoff

LOG11 = float(np.log(1.1))


def unit_rows(unit_id, state, totals, start=2001):
    rows = []
    for k, t in enumerate(totals):
        cats = {c: 0.0 for c in CORE_CATEGORIES}
        cats[CORE_CATEGORIES[0]] = float(t)
        rows.append({"unit_id": unit_id, "state_code": state, "year": start + k, **cats, "total": float(t), "n_members": 1,
                     "outside_est": 0.0, "outside_share": 0.0})
    return rows


def growth(n, rate=1.1, base=100.0):
    return [base * rate ** k for k in range(n)]


def test_windows_use_only_the_years_before_the_label_year():
    p = pd.DataFrame(unit_rows("a", 9, growth(12)))                        # 2001-2012, +10% every year
    w = bakeoff.windows(p, {})
    assert w["label_year"].tolist() == [2008, 2009, 2010, 2011, 2012]       # six lags need seven earlier totals
    r = w[w["label_year"] == 2010].iloc[0]
    assert all(r[f"lag_{k}"] == pytest.approx(LOG11) for k in range(1, 7)) and all(r[f"diff_{k}"] == pytest.approx(0.0) for k in range(1, 6))
    assert r["y"] == pytest.approx(LOG11) and r["count_prev"] == pytest.approx(100 * 1.1 ** 8) and r["count_next"] == pytest.approx(100 * 1.1 ** 9)
    assert (r["lag_mean"], r["lag_std"], r["lag_slope"]) == (pytest.approx(LOG11), pytest.approx(0.0), pytest.approx(0.0))
    assert r["log_size"] == pytest.approx(np.log1p(100 * 1.1 ** 8)) and r["state_last"] == pytest.approx(LOG11) and r["nat_last"] == pytest.approx(LOG11)
    assert np.isnan(r["nbr_last"]) and r["hist_mean"] == pytest.approx(LOG11) and r[f"mix_{CORE_CATEGORIES[0]}"] == 1.0 and r["outside_share"] == 0.0
    later = p.copy()
    later.loc[later["year"] >= 2010, ["total", CORE_CATEGORIES[0]]] *= 3     # the future changes: features before it must not
    w2 = bakeoff.windows(later, {})
    for L in (2008, 2009, 2010):
        a, b = w[w["label_year"] == L].iloc[0], w2[w2["label_year"] == L].iloc[0]
        assert a[bakeoff.FEATURES].astype(float).fillna(-9).tolist() == b[bakeoff.FEATURES].astype(float).fillna(-9).tolist()
    assert w2[w2["label_year"] == 2010].iloc[0]["y"] == pytest.approx(np.log(3 * 1.1))


def test_a_zero_total_breaks_the_windows_around_it_and_neighbours_feed_nbr_last():
    totals = growth(16)
    totals[4] = 0.0                                                          # 2005: no change defined into or out of it
    p = pd.DataFrame(unit_rows("a", 9, totals) + unit_rows("b", 9, growth(16, 1.2)))
    w = bakeoff.windows(p, {"a": ["b"], "b": ["a"]})
    assert w[w["unit_id"] == "a"]["label_year"].tolist() == [2013, 2014, 2015, 2016]
    assert w[w["unit_id"] == "b"]["label_year"].tolist() == list(range(2008, 2017))
    b = w[w["unit_id"] == "b"].set_index("label_year")
    assert b.loc[2014, "nbr_last"] == pytest.approx(LOG11) and b.loc[2008, "nbr_last"] == pytest.approx(LOG11)   # a's change into 2007 is defined
    assert w[w["unit_id"] == "a"]["nbr_last"].tolist() == pytest.approx([np.log(1.2)] * 4)
    r = bakeoff.rounds(w, (2015, 2016))
    assert r[2015][0]["label_year"].max() == 2014 and r[2015][1]["label_year"].unique().tolist() == [2015] and len(r[2016][0]) == len(w) - len(r[2016][1])


def test_windows_mostly_estimated_in_any_year_they_touch_are_left_out():
    rows = unit_rows("a", 9, growth(14))                                   # 2001-2014
    for r in rows:
        if r["year"] <= 2004:
            r["outside_est"], r["outside_share"] = r["total"] * 0.9, 0.9    # not yet separate until 2005
    w = bakeoff.windows(pd.DataFrame(rows), {})
    assert w["label_year"].tolist() == list(range(2008, 2015))
    assert w.set_index("label_year")["outside_max"].to_dict() == pytest.approx({2008: 0.9, 2009: 0.9, 2010: 0.9, 2011: 0.9, 2012: 0.0, 2013: 0.0, 2014: 0.0})
    assert bakeoff.usable(w)["label_year"].tolist() == [2012, 2013, 2014]     # 2011's window still touches 2004


def test_credibility_moves_from_the_state_trend_to_the_units_own_mean_as_it_grows():
    df = pd.DataFrame({"count_prev": [10.0, 1000.0], "lag_mean": [0.3, 0.3], "state_last": [0.1, 0.1]})
    small, big = bakeoff.credibility(df, K=100)
    assert 0.1 < small < big < 0.3 and small == pytest.approx(0.1 + (10 / 110) * 0.2)
    assert bakeoff.credibility(df, K=1e-9).tolist() == pytest.approx([0.3, 0.3]) and bakeoff.credibility(df, K=1e12).tolist() == pytest.approx([0.1, 0.1])


def test_reconciliation_makes_the_levels_add_up_and_leaves_coherent_forecasts_alone():
    groups = np.array([1, 1, 2, 2])
    var = {"unit": 0.1, "state": 0.05, "nation": 0.02}
    coherent = bakeoff.reconcile(np.array([12.0, 12.0, 8.0, 8.0]), groups, {1: 24.0, 2: 16.0}, 40.0, var)
    assert coherent.tolist() == pytest.approx([12.0, 12.0, 8.0, 8.0])
    rec = bakeoff.reconcile(np.array([10.0, 10.0, 10.0, 10.0]), groups, {1: 24.0, 2: 16.0}, 40.0, var)
    assert rec[:2].sum() > 20 > rec[2:].sum() and rec[0] == pytest.approx(rec[1]) and abs(rec.sum() - 40) < 1   # a compromise between levels
    pinned = bakeoff.reconcile(np.array([10.0, 10.0, 10.0, 10.0]), groups, {1: 24.0, 2: 16.0}, 40.0, {**var, "nation": 1e-12})
    assert pinned.sum() == pytest.approx(40.0, rel=1e-6)                     # a level trusted completely is matched exactly
    states = bakeoff.reconcile(np.array([10.0, 10.0, 10.0, 10.0]), groups, {1: 24.0, 2: 16.0}, 40.0, {**var, "state": 1e-12})
    assert states[:2].sum() == pytest.approx(24.0, rel=1e-6) and states[2:].sum() == pytest.approx(16.0, rel=1e-6)


def frame(y, label_years=None, unit_ids=None):
    n = len(y)
    return pd.DataFrame({"unit_id": unit_ids or [f"u{i}" for i in range(n)], "label_year": label_years or [2019] * n,
                         "count_prev": np.full(n, 100.0), "count_next": 100.0 * np.exp(y), "y": np.asarray(y, dtype=float)})


def test_precision_at_10_flags_the_top_tenth_within_each_year():
    y = np.linspace(-0.5, 0.5, 20)
    df = frame(y)
    assert bakeoff.top_decile(df, y).sum() == 2 and bakeoff.precision_at_10(df, y) == 1.0 and bakeoff.precision_at_10(df, -y) == 0.0
    two_years = frame(np.concatenate([y, y]), label_years=[2019] * 20 + [2020] * 20)
    assert bakeoff.top_decile(two_years, two_years["y"].to_numpy()).sum() == 4
    assert bakeoff.bootstrap_precision(df, y, n_boot=50) == [1.0, 1.0]       # a perfect ranking stays perfect in every resample
    lo, hi = bakeoff.bootstrap_precision(df, -y, n_boot=50)
    assert 0.0 <= lo <= hi < 1.0


def test_scores_and_the_paired_bootstrap():
    y = np.array([0.1, -0.2, 0.3, 0.0, 0.05, -0.1, 0.2, -0.05])
    df = frame(y, unit_ids=["a", "a", "b", "b", "c", "c", "d", "d"])
    zero = np.zeros(len(y))
    s = bakeoff.score(df, zero, mae_zero=bakeoff.score(df, zero)["mae"])
    assert s["skill"] == pytest.approx(0.0) and s["n"] == 8 and s["wape"] == pytest.approx(np.abs(100 - 100 * np.exp(y)).sum() / (100 * np.exp(y)).sum())
    perfect = bakeoff.score(df, y, mae_zero=s["mae"])
    assert perfect["mae"] == 0.0 and perfect["skill"] == 1.0 and perfect["r2"] == 1.0 and perfect["spearman"] == 1.0
    assert bakeoff.bootstrap_mae_diff(df, zero, zero, n_boot=50) == [0.0, 0.0]
    lo, hi = bakeoff.bootstrap_mae_diff(df, y, zero, n_boot=200)
    assert hi < 0 and lo <= hi                                              # a perfect forecast beats 'no change' in every resample
