"""Re-basing on hand-built worlds with known shares: pairs, chains, several parents, late poolings, map-only
districts, missing years, missing evidence, the grouping rule, and the store round trip."""
import json

import numpy as np
import pandas as pd
import pytest

from agents import rebase, store
from agents.rebase import JOIN, SPLIT, Pooling, Rebaser
from agents.schemas import CORE_CATEGORIES

STATE = 9
NAMES = {1: "A", 2: "B", 3: "C", 4: "D", 5: "M"}


def districts(codes=(1, 2, 3, 4, 5)):
    return {c: {"state_code": STATE, "name": NAMES[c], "aliases": [], "in_files": 1} for c in codes}


def panel(filed: dict, split=None):
    """``filed``: (code, year) -> total. The total sits in the first category unless ``split`` gives a
    (first, second) pair for every row."""
    rows = []
    for (code, year), total in sorted(filed.items()):
        counts = {c: 0 for c in CORE_CATEGORIES}
        if split:
            counts[CORE_CATEGORIES[0]], counts[CORE_CATEGORIES[1]] = split(total)
        else:
            counts[CORE_CATEGORIES[0]] = total
        rows.append({"unit_id": f"d{code}" if code else f"s{STATE}u", "state_code": STATE, "district_code": code or None,
                     "year": year, "era": "x", **counts, "total": total, "n_records": 1})
    df = pd.DataFrame(rows)
    df["district_code"] = df["district_code"].astype("Int64")
    return df


def area(rows):
    """rows: (old_name, new_code, share, new_share)."""
    return pd.DataFrame([{"old_state": "s", "old_name": o, "old_norm": o.lower(), "old_state_norm": "s", "new_code": c,
                          "new_name": NAMES[c], "new_state_code": STATE, "share": sh, "new_share": ns} for o, c, sh, ns in rows])


def row(df, code, year):
    hit = df[(df["district_code"] == code) & (df["year"] == year)]
    assert len(hit) == 1, (code, year, len(hit))
    return hit.iloc[0]


def test_a_pair_takes_its_share_from_the_first_three_separate_years():
    filed = {(1, y): t for y, t in zip(range(2001, 2007), (100, 100, 100, 200, 300, 400))}
    filed |= {(2, 2004): 100, (2, 2005): 100, (2, 2006): 100}
    rb = Rebaser(panel(filed), [Pooling(2, 1, SPLIT, 2004)], districts((1, 2)))
    df = rb.run()
    share = rb.shares[Pooling(2, 1, SPLIT, 2004)]
    assert share.basis == "crime:3y" and share.years == [2004, 2005, 2006]
    assert share.share == pytest.approx((1 / 3 + 1 / 4 + 1 / 5) / 3)              # the mean of the three ratios, 0.261
    b, a = row(df, 2, 2001), row(df, 1, 2001)
    assert b["total"] == pytest.approx(100 * share.share) and a["total"] == pytest.approx(100 * (1 - share.share))
    assert (b["is_estimated"], b["share_basis"], b["holder_code"], b["share"]) == (1, "crime:3y", 1, pytest.approx(share.share))
    assert (a["is_estimated"], a["share_basis"], a["share"]) == (1, "remainder", pytest.approx(1 - share.share)) and pd.isna(a["holder_code"])
    assert row(df, 1, 2004)["share_basis"] == "observed" and row(df, 2, 2004)["share_basis"] == "observed"
    assert len(df) == 12 and df["total"].sum() == pytest.approx(sum(filed.values()))    # 6 + 3 filed, 3 estimated; nothing invented
    assert rb.coverage[0] == {"year": 2001, "observed": 0, "adjusted": 1, "estimated": 1, "missing": 0, "missing_codes": []}


def test_one_separate_year_is_a_weaker_basis_and_each_category_keeps_its_proportion():
    filed = {(1, 2001): 100, (1, 2004): 300, (2, 2004): 100}
    rb = Rebaser(panel(filed, split=lambda t: (0.75 * t, 0.25 * t)), [Pooling(2, 1, SPLIT, 2004)], districts((1, 2)))
    df = rb.run()
    assert rb.shares[Pooling(2, 1, SPLIT, 2004)].basis == "crime:1y"
    b = row(df, 2, 2001)
    assert b["share_basis"] == "crime:1y" and b[CORE_CATEGORIES[0]] == pytest.approx(18.75) and b[CORE_CATEGORIES[1]] == pytest.approx(6.25)


def test_a_chain_is_peeled_outer_edge_first():
    filed = {(1, y): 100 for y in range(2001, 2009)}                      # A holds B (to 2003), B holds C (to 2005)
    filed |= {(2, 2004): 100, (2, 2005): 100} | {(2, y): 50 for y in range(2006, 2009)} | {(3, y): 50 for y in range(2006, 2009)}
    edges = [Pooling(2, 1, SPLIT, 2004), Pooling(3, 2, SPLIT, 2006)]
    rb = Rebaser(panel(filed), edges, districts((1, 2, 3)))
    df = rb.run()
    assert rb.shares[edges[0]].share == pytest.approx(0.5) and rb.shares[edges[0]].years == [2004, 2005, 2006]   # C counts with B
    assert rb.shares[edges[1]].share == pytest.approx(0.5)
    assert row(df, 1, 2001)["total"] == pytest.approx(50) and row(df, 2, 2001)["total"] == pytest.approx(25)
    c = row(df, 3, 2001)
    assert c["total"] == pytest.approx(25) and c["holder_code"] == 2
    assert row(df, 2, 2005)["share_basis"] == "remainder" and row(df, 3, 2005)["total"] == pytest.approx(50)
    assert df[df["year"] == 2001]["total"].sum() == pytest.approx(100)


def test_a_child_of_several_parents_is_divided_by_area_cover_before_shares_are_measured():
    filed = {(1, y): 100 for y in range(2001, 2008)} | {(2, y): 100 for y in range(2001, 2008)} | {(3, y): 100 for y in range(2005, 2008)}
    edges = [Pooling(3, 1, SPLIT, 2005), Pooling(3, 2, SPLIT, 2005)]
    rb = Rebaser(panel(filed), edges, districts((1, 2, 3)), area([("A", 3, 0.3, 0.75), ("B", 3, 0.1, 0.25)]))
    df = rb.run()
    assert rb.weights[edges[0]] == pytest.approx(0.75) and rb.weights[edges[1]] == pytest.approx(0.25)
    assert rb.shares[edges[0]].share == pytest.approx(75 / 175) and rb.shares[edges[1]].share == pytest.approx(25 / 125)
    assert rb.shares[edges[0]].area_share == pytest.approx(0.3)
    c = row(df, 3, 2001)
    assert c["total"] == pytest.approx(100 * 75 / 175 + 100 * 25 / 125) and c["holder_code"] == 1 and c["share"] == pytest.approx(c["total"] / 200)
    assert row(df, 1, 2001)["total"] + row(df, 2, 2001)["total"] + c["total"] == pytest.approx(200)
    assert rb.diag.get("equal_weights", []) == []

    rb2 = Rebaser(panel(filed), edges, districts((1, 2, 3)))                 # no old map: equal parts, flagged
    rb2.run()
    assert rb2.weights[edges[0]] == 0.5 and rb2.diag["equal_weights"] == [{"member": 3, "holders": [1, 2]}]


def test_a_late_pooling_is_measured_on_the_years_before_it_and_always_groups():
    filed = {(1, y): 100 for y in (2001, 2002, 2003)} | {(1, y): 150 for y in (2004, 2005)} | {(2, y): 50 for y in (2001, 2002, 2003)}
    edge = Pooling(2, 1, JOIN, 2004)
    rb = Rebaser(panel(filed), [edge], districts((1, 2)))
    df = rb.run()
    assert rb.shares[edge].share == pytest.approx(1 / 3) and rb.shares[edge].years == [2003, 2002, 2001]
    assert row(df, 2, 2004)["total"] == pytest.approx(50) and row(df, 1, 2004)["share_basis"] == "remainder"
    assert row(df, 2, 2001)["share_basis"] == "observed"
    [e] = rb.group_edges()
    assert e["grouped"] and e["why"].startswith("late pooling") and (e["parent_code"], e["child_code"]) == (2, 1)


def test_map_only_districts_get_no_rows_and_link_their_holders_only_when_they_move():
    filed = {(1, y): 100 for y in range(2001, 2006)} | {(2, y): 100 for y in range(2001, 2006)}
    edges = [Pooling(5, 1, SPLIT, 2004), Pooling(5, 2, JOIN, 2004)]        # M: inside A to 2003, inside B from 2004
    rb = Rebaser(panel(filed), edges, districts((1, 2, 5)), area([("A", 5, 0.3, 1.0)]))
    df = rb.run()
    assert rb.map_only == {5} and rb.poolings == [] and set(df["district_code"]) == {1, 2}
    assert set(df["share_basis"]) == {"observed"}
    [note] = rb.map_only_notes()
    assert note["moves"] and note["inside_before"] == [(1, 2004)] and note["inside_from"] == [(2, 2004)]
    units = rb.build_units()
    assert [u["members"] for u in units] == [[1, 2, 5]]

    rb2 = Rebaser(panel(filed), edges, districts((1, 2, 5)))                # no area share: holders stay apart
    rb2.run()
    assert [u["members"] for u in rb2.build_units()] == [[1], [2], [5]]
    rb3 = Rebaser(panel(filed), [Pooling(5, 1, JOIN, 2001)], districts((1, 2, 5)), area([("A", 5, 0.9, 1.0)]))
    rb3.run()                                                                # always inside A: nothing moves, no group
    assert not rb3.map_only_notes()[0]["moves"] and [u["members"] for u in rb3.build_units()] == [[1], [2], [5]]


def test_a_year_a_district_did_not_file_stays_missing():
    filed = {(1, y): 100 for y in range(2001, 2006)} | {(2, y): 50 for y in (2001, 2002, 2003, 2005)}
    rb = Rebaser(panel(filed), [], districts((1, 2)))
    df = rb.run()
    assert df[(df["district_code"] == 2) & (df["year"] == 2004)].empty
    assert [c["missing_codes"] for c in rb.coverage] == [[], [], [], [2], []]
    assert set(df["share_basis"]) == {"observed"}


def test_an_edge_with_no_evidence_estimates_nothing_and_is_reported():
    filed = {(1, y): 100 for y in (2001, 2002, 2003)} | {(2, 2004): 50}         # never both filed, no old map
    edge = Pooling(2, 1, SPLIT, 2004)
    rb = Rebaser(panel(filed), [edge], districts((1, 2)))
    df = rb.run()
    assert rb.shares[edge].share is None
    assert row(df, 1, 2001)["share_basis"] == "observed" and df[df["district_code"] == 2]["year"].tolist() == [2004]
    assert {d["year"] for d in rb.diag["no_estimate"]} == {2001, 2002, 2003}
    assert rb.group_edges()[0]["why"] == "no share: nothing estimated"

    with_area = Rebaser(panel(filed), [edge], districts((1, 2)), area([("A", 2, 0.4, 1.0)]))
    df2 = with_area.run()
    assert with_area.shares[edge].basis == "area" and row(df2, 2, 2001)["total"] == pytest.approx(40)


def test_the_threshold_decides_grouping_from_the_share_actually_used():
    filed = {(1, y): 100 for y in range(2001, 2007)} | {(2, y): 30 for y in range(2004, 2007)}         # share 30/130 = 0.23
    edge = Pooling(2, 1, SPLIT, 2004)
    low = Rebaser(panel(filed), [edge], districts((1, 2)), threshold=0.2)
    low.run()
    assert low.group_edges()[0]["grouped"] and [u["members"] for u in low.build_units()] == [[1, 2]]
    high = Rebaser(panel(filed), [edge], districts((1, 2)), threshold=0.3)
    high.run()
    assert not high.group_edges()[0]["grouped"] and [u["members"] for u in high.build_units()] == [[1], [2]]
    assert row(high.run(), 2, 2001)["total"] == pytest.approx(100 * 30 / 130)            # the transfer still happens


def test_a_row_filed_before_the_lineage_year_stays_observed_and_is_listed():
    filed = {(1, y): 100 for y in range(2001, 2007)} | {(2, y): 50 for y in range(2003, 2007)}        # edge says 2004
    edge = Pooling(2, 1, SPLIT, 2004)
    rb = Rebaser(panel(filed), [edge], districts((1, 2)))
    df = rb.run()
    assert row(df, 2, 2003)["share_basis"] == "observed" and row(df, 2, 2002)["share_basis"] == "crime:3y"
    assert rb.filed_before_edge() == [{"edge": edge, "years": [2003]}]


def test_the_store_round_trip_writes_the_table_the_parquet_the_units_and_the_reports(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    store.replace_registry(conn, states=[(STATE, "S", "[]")],
                           districts=[(1, STATE, "A", "[]", 2016, 2022), (2, STATE, "B", "[]", 2016, 2022)], circles=[])
    counts = ", ".join(CORE_CATEGORIES)
    with conn:
        for code, year, total in ((1, 2001, 100), (1, 2004, 200), (2, 2004, 100), (None, 2001, 7)):
            conn.execute(f"INSERT INTO panel_observed (unit_id, state_code, district_code, year, era, {counts}, total, n_records) "
                         f"VALUES (?, ?, ?, ?, '2001_2014', {total}, {', '.join('0' for _ in CORE_CATEGORIES[1:])}, {total}, 1)",
                         (f"d{code}" if code else f"s{STATE}u", STATE, code, year))
        conn.execute("INSERT INTO district_lineage (parent_code, child_code, kind, year, confidence, reason, source, verified) "
                     "VALUES (1, 2, 'split', 2004, 1.0, 'r', 'human', 1)")
    out = rebase.build_rebased(conn, parquet_path=tmp_path / "p.parquet", units_path=tmp_path / "su.json",
                               reports_dir=tmp_path / "reports", area_path=tmp_path / "none.csv")
    assert out["rows"] == 5 and out["observed"] == 2 and out["adjusted"] == 1 and out["estimated"] == 1
    assert out["basis"] == {"crime:1y": 1} and out["groups"] == 1 and out["grouping_edges"] == 1 and out["largest_group"] == 2
    table = {(r["unit_id"], r["year"]): dict(r) for r in conn.execute("SELECT * FROM panel_rebased")}
    assert table[("d2", 2001)]["total"] == pytest.approx(100 / 3) and table[("d2", 2001)]["share_basis"] == "crime:1y"
    assert table[("d1", 2001)]["share_basis"] == "remainder" and table[(f"s{STATE}u", 2001)]["share_basis"] == "observed"
    assert len(pd.read_parquet(tmp_path / "p.parquet")) == 5
    units = json.loads((tmp_path / "su.json").read_text(encoding="utf-8"))
    assert units["meta"]["threshold"] == 0.2 and units["meta"]["edges"][0]["share"] == pytest.approx(1 / 3, abs=1e-4)
    assert [u["members"] for u in units["units"]] == [[1, 2]]
    assert sorted(p.name.split("_", 1)[1] for p in (tmp_path / "reports").iterdir()) == ["rebasing.md", "stable_units.md"]
    rebase.build_rebased(conn, parquet_path=tmp_path / "p.parquet", units_path=tmp_path / "su.json",
                         reports_dir=tmp_path / "reports", area_path=tmp_path / "none.csv")                 # rerun: same rows
    assert conn.execute("SELECT COUNT(*) FROM panel_rebased").fetchone()[0] == 5
