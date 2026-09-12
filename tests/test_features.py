"""Guards for the properties that make ``ml.features`` leak-free.

These are cheap and deliberately literal: if someone later "improves" the
feature builder in a way that lets a label or a later year influence a
feature, ``test_features_never_depend_on_label_or_future`` fails.
"""
import numpy as np
import pandas as pd

from ml.features import FEATURE_NAMES, WINDOW, make_windows, window_features, yearly_pct_change


def _one_unit(values_by_year: dict[int, float]) -> pd.DataFrame:
    idx = pd.MultiIndex.from_tuples([("s", "u")], names=["state", "unit"])
    return pd.DataFrame({y: [v] for y, v in values_by_year.items()}, index=idx)


def test_feature_count_matches_names():
    assert len(window_features(np.arange(WINDOW, dtype=float))) == len(FEATURE_NAMES)


def test_diffs_and_summaries_are_computed_inside_the_window():
    x = np.array([1.0, 4.0, 2.0, 8.0, 5.0, 7.0])
    f = dict(zip(FEATURE_NAMES, window_features(x)))
    assert f["lag_1"] == 1.0 and f["lag_6"] == 7.0 and f["lag_last"] == 7.0
    assert f["diff_1"] == 3.0 and f["diff_2"] == -2.0 and f["diff_5"] == 2.0
    assert f["lag_mean"] == x.mean()


def test_features_never_depend_on_label_or_future():
    base = {2002: 1.0, 2003: 2.0, 2004: 3.0, 2005: 4.0, 2006: 5.0, 2007: 6.0, 2008: 100.0, 2009: -50.0}
    X1, y1, meta = make_windows(_one_unit(base))
    changed = {**base, 2008: 999.0, 2009: 999.0}          # only the label year and later change
    X2, y2, _ = make_windows(_one_unit(changed))
    assert int(meta.loc[0, "label_year"]) == 2008
    np.testing.assert_array_equal(X1[0], X2[0])           # features identical
    assert y1[0] == 100.0 and y2[0] == 999.0              # only the label moved


def test_windows_with_missing_values_are_skipped_not_imputed():
    X, y, _ = make_windows(_one_unit({2002: 1.0, 2003: np.nan, 2004: 3.0, 2005: 4.0, 2006: 5.0, 2007: 6.0, 2008: 7.0}))
    assert len(y) == 0


def test_zero_base_is_nan_not_inf():
    long = pd.DataFrame({"state": ["s"] * 3, "unit": ["u"] * 3, "year": [2001, 2002, 2003], "total": [0, 5, 10]})
    ch = yearly_pct_change(long)
    assert np.isnan(ch.loc[("s", "u"), 2002])
    assert ch.loc[("s", "u"), 2003] == 100.0
