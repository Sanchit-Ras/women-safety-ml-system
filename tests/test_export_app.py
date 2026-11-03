"""Batch scoring for the app: next-year windows, the watch list, the parameters read from the report, the plain
summaries and the download table, and the committed files agreeing with the reports and the panel (so the site cannot
drift from the evidence)."""
import json
import math
import re

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


def test_bands_split_the_change_at_their_bounds_and_few_cases_overrule_them():
    band = export_app.band_of
    assert [band(c, 100) for c in (-0.5, -0.1, 0.0, 0.0999, 0.1, 0.4999, 0.5, 3.0)] == \
        ["fell", "steady", "steady", "steady", "rose", "rose", "rose_sharply", "rose_sharply"]
    assert band(1.0, 29) == band(None, 100) == "few_cases" and band(1.0, 30) == "rose_sharply"
    assert export_app.change_of(0, 5) is None and export_app.change_of(40, 30) == -0.25


YEARS = list(range(2017, 2023))


def district(code, state, totals, bases, cat="cruelty_by_husband"):
    """One district's rows, 2017-2022: every case in one category; bases name how each year's number came about."""
    return [{"unit_id": f"d{code}", "state_code": state, "district_code": code, "year": y,
             **{c: 0.0 for c in export_app.CORE_CATEGORIES}, cat: float(t), "total": float(t), "share_basis": b}
            for y, t, b in zip(YEARS, totals, bases)]


def small_panel():
    obs, rem, est = "observed", "remainder", "crime:3y"
    rows = (district(1, 1, [100, 110, 120, 130, 140, 150], [obs] * 6)               # filed in both years
            + district(2, 1, [60, 58, 56, 54, 52, 50], [rem] * 5 + [obs])            # 2 and 3: one unit, split in 2022
            + district(3, 1, [40, 41, 42, 43, 44, 45], [est] * 5 + [obs])
            + district(4, 1, [35, 38, 40, 36, 33, 30], [est] * 2 + [obs] * 4)        # alone, counted as today from 2019
            + district(5, 2, [10, 12, 14, 16, 18, 20], [obs] * 6)                    # too few cases for a percentage
            + district(6, 2, [50, 50, 50, 50, 50, 70], [est] * 5 + [obs])            # alone, borders changed in 2022
            + [{"unit_id": "s2u", "state_code": 2, "district_code": None, "year": 2017, "total": 5.0, "share_basis": obs,
                **{c: 0.0 for c in export_app.CORE_CATEGORIES}}])                    # a state-level unit, no district
    p = pd.DataFrame(rows).astype({"district_code": "Int64"})
    tie = (p["district_code"] == 1) & (p["year"] == 2022)                            # 2022: rape and dowry deaths equal
    p.loc[tie, ["cruelty_by_husband", "rape", "dowry_deaths"]] = [0.0, 75.0, 75.0]
    return p


def test_summaries_compare_like_with_like():
    facts, meta = export_app.summaries(small_panel(), {"u1": [1], "u23": [2, 3], "u4": [4], "u5": [5], "u6": [6]})
    assert meta["india"] == {"from_total": 300.0, "to_total": 365.0, "change": 0.2167}       # the state unit counts here
    assert meta["states"]["1"]["change"] == 0.1702 and meta["states"]["2"]["change"] == 0.3846
    t = {c: f["trend"] for c, f in facts.items()}
    assert t[1] == {"scope": "district", "from_year": 2017, "to_year": 2022, "from_total": 100.0, "to_total": 150.0,
                    "change": 0.5, "band": "rose_sharply", "state_change": 0.1702, "india_change": 0.2167}
    assert t[2]["scope"] == t[3]["scope"] == "area" and (t[2]["from_total"], t[2]["to_total"], t[2]["band"]) == (100.0, 95.0, "steady")
    assert t[2] == t[3]                                                                    # one unit, one comparison
    assert (t[4]["from_year"], t[4]["change"], t[4]["band"]) == (2019, -0.25, "fell")      # its first year as today
    assert (t[4]["state_change"], t[4]["india_change"]) == (0.0659, 0.1335)                # over the same, shorter years
    assert (t[5]["change"], t[5]["band"]) == (1.0, "few_cases") and t[6] is None
    assert [facts[c]["rank"]["rank"] for c in (1, 2, 3, 4)] == [1, 2, 3, 4] and facts[4]["rank"]["of"] == 4
    assert (facts[6]["rank"], facts[5]["rank"]["rank"]) == ({"year": 2022, "rank": 1, "of": 2}, 2)
    assert facts[1]["top"] == {"category": "rape", "share": 0.5, "state_share": 0.2727}   # the first of equals
    assert facts[6]["top"]["category"] == "cruelty_by_husband" and None not in facts


def test_the_download_table_has_a_row_per_district_and_year_and_says_how_each_was_counted():
    table = export_app.dataset(small_panel(), {1: {"name": "Alpha"}, 6: {"name": "Zeta"}}, {1: "Beta State", 2: "Alpha State"})
    assert len(table) == 36 and table["district_code"].notna().all()                       # the state unit is left out
    assert table["state"].iloc[0] == "Alpha State" and table.loc[table["district_code"] == 6, "district"].iloc[0] == "Zeta"
    counted = table.set_index(["district_code", "year"])["how_counted"]
    assert (counted[(2, 2017)], counted[(3, 2017)], counted[(3, 2022)]) == ("adjusted", "estimated", "filed")
    assert list(table.columns[-2:]) == ["total", "how_counted"]


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


def test_the_download_table_is_the_rebased_panel(app_data):
    table = pd.read_csv(export_app.DATASET)
    rebased = pd.read_parquet(unit_panel.REBASED_PATH)
    d = rebased[rebased["district_code"].notna()]
    shape = app_data["meta"]["dataset"]
    assert list(table.columns) == shape["columns"] and len(table) == len(d) == shape["rows"]
    assert table["district_code"].nunique() == shape["districts"] and not table.duplicated(["district_code", "year"]).any()
    assert table["total"].sum() == pytest.approx(d["total"].sum(), abs=0.05 * len(d))       # rounded to one decimal
    assert (table["how_counted"] == "filed").sum() == (d["share_basis"] == "observed").sum()
    assert shape["path"] == "/" + export_app.DATASET.relative_to(export_app.ROOT / "frontend" / "public").as_posix()


def test_every_summary_agrees_with_the_districts_own_numbers(app_data):
    m = app_data["meta"]["trend"]
    for d in app_data["districts"].values():
        s, years = d["summary"], {p[0]: p for p in d["series"]}
        t = s["trend"]
        if t:
            assert t["band"] == export_app.band_of(t["change"], t["from_total"], m["min_cases"]) and t["to_year"] == m["to_year"]
            if t["from_year"] == m["from_year"]:
                assert t["india_change"] == m["india"]["change"] and t["state_change"] == m["states"][str(d["state_code"])]["change"]
        if t and t["scope"] == "district":                                                # both years as filed, its own
            assert years[t["from_year"]][2] == years[t["to_year"]][2] == export_app.FILED
            assert (years[t["from_year"]][1], years[t["to_year"]][1]) == (t["from_total"], t["to_total"])
        if t and t["scope"] == "area":
            assert len(d["unit_members"]) > 1
        if s["rank"]:
            assert 1 <= s["rank"]["rank"] <= s["rank"]["of"] and s["rank"]["year"] == d["latest_year"]
        if s["top"]:
            assert d["latest"][s["top"]["category"]] == max(d["latest"].values())
    assert sum(d["summary"]["trend"] is not None for d in app_data["districts"].values()) > 700


def test_the_site_quotes_the_numbers_of_the_reports(app_data):
    """Every number on the "How it was built" page (frontend/src/content/story.js) and the data page, against its source."""
    root = export_app.ROOT
    src = root / "frontend" / "src"
    story = (src / "content" / "story.js").read_text(encoding="utf-8")
    pages = story + (src / "pages" / "About.jsx").read_text(encoding="utf-8") + (src / "pages" / "Data.jsx").read_text(encoding="utf-8")
    r = json.loads((root / "evals" / "reports" / "2026-10-18.json").read_text(encoding="utf-8"))
    nb, h, esc, cov = r["notebook_coverage"], r["hallucination"], r["escalation"]["2001_2014"], r["coverage"]["all"]
    assert f"{nb['pipeline_share']:.1%} of {nb['units']:,}" in pages and f"{nb['share']:.1%}" in pages
    assert f"{h['resolver_codes_outside_candidates']} of {h['resolver_answers']}" in story
    assert f"{esc['needed_review']} → {esc['with_cut']}" in story and f"−{round(100 * (1 - esc['with_cut'] / esc['needed_review']))}%" in story
    assert f"{esc['wrong_accepted']} errors" in story and f"from {esc['needed_review']} to {esc['with_cut']}" in story
    assert f"{cov['share']:.1%}" in story and f"{cov['records']:,} records" in story
    e = app_data["meta"]["evidence"]
    assert f"{round(e['precision_at_10'] * 100)}%" in story and f"{round(e['size_only'] * 100)}% by size alone" in story
    assert f"tests on {e['test_years'][0]}–{e['test_years'][-1]}" in story and f"{round(e['chance'] * 100)}% by guessing" in story
    f = json.loads((root / "evals" / "reports" / "2026-11-01_frameworks.json").read_text(encoding="utf-8"))
    lines, pooled = f["static"]["control_code_lines"], f["pooled"]
    assert pooled["A"]["precision"] == pooled["B"]["precision"] and f"Same {round(pooled['A']['precision'] * 100)}%" in story
    assert f"same {f['names']} names" in story and f"{lines['B']} lines of control code instead of {lines['A']}" in story
    assert f"{f['static']['footprint']['count']} more packages" in story
    baseline = (root / "ml" / "reports" / "2026-09-13_era_a_baseline.md").read_text(encoding="utf-8")
    assert "| xgboost | 0.061 |" in baseline and "scored 0.06" in story and "R² 0.88" in (root / "README.md").read_text(encoding="utf-8")
    assert "R² 0.88" in story
    for path in re.findall(r'"((?:evals|ml)/reports/[^"]+\.md)"', story):
        assert (root / path).exists(), path


def test_the_link_preview_image_is_1200_by_630():
    head = (export_app.ROOT / "frontend" / "public" / "og.png").read_bytes()[:24]
    assert head[:8] == b"\x89PNG\r\n\x1a\n" and (int.from_bytes(head[16:20]), int.from_bytes(head[20:24])) == (1200, 630)
    html = (export_app.ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert '/og.png"' in html and 'content="1200"' in html and 'content="630"' in html


def test_every_map_shape_is_wound_the_way_d3_draws_it():
    from shapely.geometry import shape
    geo = json.loads(export_app.MAP_OUT.read_text(encoding="utf-8"))
    for f in geo["features"]:
        g = shape(f["geometry"])
        for p in getattr(g, "geoms", [g]):
            assert not p.exterior.is_ccw and all(r.is_ccw for r in p.interiors), f["properties"]["code"]
