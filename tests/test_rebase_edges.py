"""Re-basing edge cases from the audit: a district that joined the member later, a grandchild carved in the
member's own split year, the peel order of late poolings and mixed kinds, holders without a row, rounding
remainders, old-map polygons, edges the map gives no land, and idempotent runs."""
import numpy as np
import pandas as pd
import pytest

from agents.rebase import JOIN, SPLIT, Pooling, Rebaser
from agents.schemas import CORE_CATEGORIES
from tests.test_rebase import NAMES, STATE, area, districts, panel, row

NAMES.update({6: "J", 7: "K", 8: "G", 9: "P", 10: "Q"})


def test_a_district_that_joined_the_member_later_is_not_part_of_the_split_unit():
    filed = {(1, y): 100 for y in range(2001, 2008)} | {(2, y): 50 for y in range(2005, 2008)} | {(6, y): 1000 for y in range(2001, 2008)}
    split, join = Pooling(2, 1, SPLIT, 2005), Pooling(6, 2, JOIN, 2005)           # J joins B in 2005 but keeps filing
    rb = Rebaser(panel(filed), [split, join], districts((1, 2, 6)))
    df = rb.run()
    assert rb.shares[split].share == pytest.approx(1 / 3)                            # 50 / (100 + 50), J's 1000 left out
    assert row(df, 2, 2001)["total"] == pytest.approx(100 / 3) and row(df, 6, 2001)["share_basis"] == "observed"
    assert set(df[df["district_code"] == 6]["share_basis"]) == {"observed"}          # the join never fires: J files every year


def test_a_grandchild_carved_in_the_members_own_split_year_stays_on_the_members_side():
    filed = {(1, y): 100 for y in range(2001, 2008)} | {(2, y): 50 for y in range(2005, 2008)} | {(3, y): 25 for y in range(2005, 2008)}
    edges = [Pooling(2, 1, SPLIT, 2005), Pooling(3, 2, SPLIT, 2005)]
    rb = Rebaser(panel(filed), edges, districts((1, 2, 3)))
    df = rb.run()
    assert rb.shares[edges[0]].share == pytest.approx(75 / 175) and rb.shares[edges[1]].share == pytest.approx(25 / 75)
    assert row(df, 1, 2001)["total"] == pytest.approx(100 * 100 / 175)
    assert row(df, 2, 2001)["total"] == pytest.approx(100 * 50 / 175) and row(df, 3, 2001)["total"] == pytest.approx(100 * 25 / 175)


def test_late_poolings_peel_latest_first_and_splits_peel_from_the_whole_row():
    # H keeps 100 of its own; J1 (50) joins in 2005, J2 (30) joins in 2010.
    filed = {(1, y): 100 for y in range(2001, 2005)} | {(1, y): 150 for y in range(2005, 2010)} | {(1, y): 180 for y in range(2010, 2013)}
    filed |= {(6, y): 50 for y in range(2001, 2005)} | {(7, y): 30 for y in range(2001, 2010)}
    j1, j2 = Pooling(6, 1, JOIN, 2005), Pooling(7, 1, JOIN, 2010)
    rb = Rebaser(panel(filed), [j1, j2], districts((1, 6, 7)))
    df = rb.run()
    assert rb.shares[j1].share == pytest.approx(1 / 3) and rb.shares[j2].share == pytest.approx(1 / 6)
    assert row(df, 7, 2011)["total"] == pytest.approx(30) and row(df, 6, 2011)["total"] == pytest.approx(50)
    assert row(df, 1, 2011)["total"] == pytest.approx(100)

    # H keeps 100; J (60) joined in 2003; M (40) split off in 2008. In 2005 both sit in H's row of 200.
    filed = {(1, y): 140 for y in (2001, 2002)} | {(6, y): 60 for y in (2001, 2002)} | {(1, y): 200 for y in range(2003, 2008)}
    filed |= {(1, y): 160 for y in range(2008, 2011)} | {(5, y): 40 for y in range(2008, 2011)}
    j, m = Pooling(6, 1, JOIN, 2003), Pooling(5, 1, SPLIT, 2008)
    rb = Rebaser(panel(filed), [j, m], districts((1, 5, 6)))
    df = rb.run()
    assert rb.shares[j].share == pytest.approx(0.3) and rb.shares[m].share == pytest.approx(0.2)
    assert row(df, 6, 2005)["total"] == pytest.approx(60) and row(df, 5, 2005)["total"] == pytest.approx(40)
    assert row(df, 1, 2005)["total"] == pytest.approx(100)


def test_a_member_whose_holder_has_no_row_that_year_is_left_out_and_reported():
    # A -> B has no evidence (never both filed), B -> C does: in 2001-03 C's holder B has no row, so C gets none either.
    filed = {(1, y): 100 for y in (2001, 2002, 2003)} | {(2, y): 100 for y in range(2004, 2009)} | {(3, y): 50 for y in range(2006, 2009)}
    rb = Rebaser(panel(filed), [Pooling(2, 1, SPLIT, 2004), Pooling(3, 2, SPLIT, 2006)], districts((1, 2, 3)))
    df = rb.run()
    assert df[(df["district_code"] == 3) & (df["year"] < 2004)].empty and row(df, 3, 2005)["total"] == pytest.approx(100 / 3)
    assert [c["missing_codes"] for c in rb.coverage[:3]] == [[2, 3]] * 3
    assert {(d["member"], d["year"]) for d in rb.diag["holder_unresolved"]} == {(3, 2001), (3, 2002), (3, 2003)}
    assert "cycles" not in rb.diag

    # M from A and B; B filed nothing in 2002: M gets no row that year rather than a silently partial one.
    filed = {(1, y): 100 for y in range(2001, 2008)} | {(2, y): 100 for y in range(2001, 2008) if y != 2002} | {(5, y): 100 for y in range(2005, 2008)}
    rb = Rebaser(panel(filed), [Pooling(5, 1, SPLIT, 2005), Pooling(5, 2, SPLIT, 2005)], districts((1, 2, 5)))
    df = rb.run()
    assert df[(df["district_code"] == 5) & (df["year"] == 2002)].empty and row(df, 5, 2001)["total"] == pytest.approx(200 / 3)
    assert row(df, 1, 2002)["share_basis"] == "observed" and rb.diag["holder_unresolved"] == [{"member": 5, "year": 2002, "holders": [2]}]


def test_rounding_remainders_are_clipped_to_zero_without_a_warning():
    filed = {(1, 2001): 11} | {(1, y): 0 for y in range(2005, 2008)} | {(2, y): 1 for y in range(2005, 2008)} | {(3, y): 5 for y in range(2005, 2008)}
    rb = Rebaser(panel(filed), [Pooling(2, 1, SPLIT, 2005), Pooling(3, 1, SPLIT, 2005)], districts((1, 2, 3)))
    df = rb.run()
    assert (df[list(CORE_CATEGORIES)] >= 0).all().all() and row(df, 1, 2001)["total"] == 0.0
    assert "over_assigned" not in rb.diag and row(df, 2, 2001)["total"] + row(df, 3, 2001)["total"] == pytest.approx(11)


def test_an_edge_the_old_map_gives_no_land_moves_nothing_and_says_so():
    filed = {(1, y): 100 for y in range(2001, 2008)} | {(2, y): 100 for y in range(2001, 2008)} | {(3, y): 100 for y in range(2005, 2008)}
    edges = [Pooling(3, 1, SPLIT, 2005), Pooling(3, 2, SPLIT, 2005)]
    old_map = area([("A", 1, 0.7, 1.0), ("A", 3, 0.3, 1.0), ("B", 2, 1.0, 1.0)])     # B's polygon is B alone: no land in C
    rb = Rebaser(panel(filed), edges, districts((1, 2, 3)), old_map)
    df = rb.run()
    assert rb.weights[edges[0]] == 1.0 and rb.weights[edges[1]] == 0.0
    assert rb.shares[edges[1]].share == 0.0 and rb.shares[edges[1]].basis == "area"
    assert set(df[df["district_code"] == 2]["share_basis"]) == {"observed"}          # B is never touched
    c = row(df, 3, 2001)
    assert c["holder_code"] == 1 and c["total"] == pytest.approx(100 * 100 / 200) and c["share_basis"] == "crime:3y"
    assert [e["why"] for e in rb.group_edges() if e["parent_code"] == 2] == ["the old map gives this holder no land in the member: nothing moves"]
    assert rb.diag["zero_weight"] == [{"edge": edges[1]}]


def test_old_map_polygons_need_a_real_overlap_or_an_unclaimed_adoption():
    old_map = pd.concat([area([("A", 1, 0.7, 0.9), ("A", 8, 0.256, 0.8), ("P", 9, 1.0, 1.0), ("P", 10, 0.5, 0.8)]),
                         area([("A", 8, 0.942, 0.19), ("A", 1, 0.011, 0.001)]).assign(old_state="elsewhere", old_state_norm="elsewhere")])
    rb = Rebaser(panel({(1, 2001): 1}), [], districts((1, 8, 9, 10)), old_map)
    assert rb.area_share(1, 8) == pytest.approx(0.256)                              # the stray 'A' barely touches A: not A's
    assert rb._holder_polygons(10) == set() and rb.area_cover(10, 8) is None         # Q is 80% old P, but P is P's name
    assert rb._holder_polygons(9) == {("s", "p")}


def test_running_twice_gives_the_same_rows_and_diagnostics():
    filed = {(1, y): 100 for y in range(2001, 2007)} | {(2, y): 30 for y in range(2004, 2007)}
    rb = Rebaser(panel(filed), [Pooling(2, 1, SPLIT, 2004)], districts((1, 2)))
    first = rb.run()
    second = rb.run()
    assert first.equals(second) and len(rb.coverage) == 6
