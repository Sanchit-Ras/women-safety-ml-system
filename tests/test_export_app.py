"""Batch scoring for the app: next-year windows, the watch list, the parameters read from the report, and the
committed app data and map agreeing with the bake-off report and the panel (so the site cannot drift from the evidence)."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from ml import bakeoff, export_app
from ml import panel as unit_panel
from test_bakeoff import growth, unit_rows


def test_include_next_adds_one_unlabelled_row_per_unit_after_the_last_year():
    p = pd.DataFrame(unit_rows("a", 9, growth(12)))                         # 2001-2012
    w = bakeoff.windows(p, {}, include_next=True)
    nxt = w[w["label_year"] == 2013].iloc[0]
    assert len(w) == len(bakeoff.windows(p, {})) + 1 and np.isnan(nxt["y"]) and np.isnan(nxt["count_next"])
    assert nxt["lag_1"] == pytest.approx(np.log(1.1)) and nxt["count_prev"] == pytest.approx(100 * 1.1 ** 11)


def test_score_next_watches_the_top_tenth_of_scored_units():
    rng = np.random.default_rng(0)
    rows = []
    for k in range(25):
        totals = 100 * np.exp(np.cumsum(rng.normal(0.05, 0.2, 14)))
        rows += unit_rows(f"u{k:02d}", 9 if k % 2 else 10, totals.tolist())
    scored, facts = export_app.score_next(pd.DataFrame(rows), {}, {"max_depth": 2, "n_estimators": 20})
    assert facts["units_scored"] == len(scored) == 25 and facts["units_watched"] == math.ceil(0.1 * 25) == 3
    assert scored["rank"].tolist() == list(range(1, 26)) and scored["watch"].sum() == 3 and scored["watch"].iloc[:3].all()
    assert scored["score"].is_monotonic_decreasing and (scored["label_year"] == 2015).all()
    two, facts2 = export_app.score_next(pd.DataFrame(rows), {}, {"max_depth": 2, "n_estimators": 20}, groups=2)
    assert sorted(two["group"].unique()) == [1, 2] and facts2["size_groups"] == 2
    for _g, part in two.groupby("group"):                                     # each size group: its own ranks and its own top tenth
        assert sorted(part["rank"]) == list(range(1, len(part) + 1)) and part["watch"].sum() == math.ceil(0.1 * len(part))
    small, large = two[two["group"] == 1], two[two["group"] == 2]
    assert small["count_prev"].max() <= large["count_prev"].min() and (small["group_max"] == small["count_prev"].max()).all()


def test_params_of_reads_the_grid_entry_a_report_names():
    assert export_app.params_of("hotspot depth 3, 600 trees") == {"max_depth": 3, "n_estimators": 600}
    with pytest.raises(ValueError):
        export_app.params_of("hotspot depth 9, 1 trees")


def test_kind_of_tells_filed_adjusted_and_estimated_apart():
    assert [export_app.kind_of(b) for b in ("observed", "remainder", "crime:3y", "area")] == [0, 1, 2, 2]


# ---- the committed files ------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def app_data():
    return json.loads(export_app.APP_DATA.read_text(encoding="utf-8"))


def test_the_app_shows_the_numbers_of_the_latest_bakeoff_report(app_data):
    path, report = export_app.latest_bakeoff()
    m, rec, g = app_data["meta"], report["ranking_recommended"], report["size_groups"]
    assert m["evidence"]["report"] == f"ml/reports/{path.stem}.md"
    assert m["evidence"]["size_groups"] == g["chosen"] == m["model"]["size_groups"]
    assert m["evidence"]["precision_at_10"] == round(g["test"]["model"], 3)
    assert m["evidence"]["ci95"] == [round(x, 3) for x in g["test"]["ci95"]]
    assert m["evidence"]["size_only"] == round(max(g["test"]["smallest_first"], g["test"]["largest_first"]), 3)
    assert m["model"]["setting"] == report["chosen"][rec] and m["size_of_change"]["model"] == report["recommended"]
    peers = {d["unit_id"]: d["peer"] for d in app_data["districts"].values() if d["peer"]}
    sizes = {p["group"]: p["size"] for p in peers.values()}
    assert m["model"]["units_watched"] == sum(math.ceil(export_app.WATCH_SHARE * n) for n in sizes.values())


def test_every_district_of_the_panel_is_in_the_app_and_on_the_map(app_data):
    rebased = pd.read_parquet(unit_panel.REBASED_PATH)
    panel_codes = {int(c) for c in rebased["district_code"].dropna().unique()}
    app_codes = {int(c) for c in app_data["districts"]}
    geo = json.loads(export_app.MAP_OUT.read_text(encoding="utf-8"))
    map_codes = [f["properties"]["code"] for f in geo["features"]]
    assert panel_codes <= app_codes and app_codes <= set(map_codes) and len(map_codes) == len(set(map_codes))
    assert all(str(d["state_code"]) in app_data["states"] for d in app_data["districts"].values())
    watched_units = {d["unit_id"] for d in app_data["districts"].values() if d["status"] == "watch"}
    assert len(watched_units) == app_data["meta"]["model"]["units_watched"]


def test_the_research_page_quotes_the_numbers_the_app_data_carries(app_data):
    from pathlib import Path
    src = Path(export_app.ROOT) / "frontend" / "src" / "components"
    e = app_data["meta"]["evidence"]
    pct, lift = f"{round(e['precision_at_10'] * 100)}%", f"{e['precision_at_10'] / e['chance']:.1f}×"
    assert lift in (src / "Stats" / "constants.js").read_text(encoding="utf-8")
    for f in ("Findings/constants.js", "MapSection.jsx"):
        assert pct in (src / f).read_text(encoding="utf-8"), f
    assert f"value={{{e['precision_at_10']}}}" in (src / "Performance.jsx").read_text(encoding="utf-8")


def test_every_map_shape_is_wound_the_way_d3_draws_it():
    from shapely.geometry import shape
    geo = json.loads(export_app.MAP_OUT.read_text(encoding="utf-8"))
    for f in geo["features"]:
        g = shape(f["geometry"])
        for p in getattr(g, "geoms", [g]):
            assert not p.exterior.is_ccw and all(r.is_ccw for r in p.interiors), f["properties"]["code"]
