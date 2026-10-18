"""The split-weights decision record: a verified row replaces the old map's division of a multi-parent child, a bad
row stops the build with a name, a row naming no edge is reported, and the report lists what changed."""
import pytest

from agents import rebase
from agents.rebase import SPLIT, Pooling, Rebaser
from test_rebase import area, districts, panel, row

WORLD = {(1, y): 100 for y in range(2001, 2008)} | {(2, y): 100 for y in range(2001, 2008)} | {(3, y): 100 for y in range(2005, 2008)}
EDGES = [Pooling(3, 1, SPLIT, 2005), Pooling(3, 2, SPLIT, 2005)]
MAP = area([("A", 1, 0.7, 1.0), ("A", 3, 0.3, 0.75), ("B", 2, 0.9, 1.0), ("B", 3, 0.1, 0.25)])      # the map says 0.75 / 0.25


def test_verified_weights_replace_the_old_maps_division(tmp_path):
    rb = Rebaser(panel(WORLD), EDGES, districts((1, 2, 3)), MAP, weights={(3, 1, SPLIT): 0.5, (3, 2, SPLIT): 0.5})
    df = rb.run()
    assert rb.weights[EDGES[0]] == 0.5 and rb.weights[EDGES[1]] == 0.5
    assert rb.diag["weight_overrides"] == [{"member": 3, "holders": {1: 0.5, 2: 0.5}}] and rb.diag.get("equal_weights", []) == []
    assert rb.shares[EDGES[0]].share == pytest.approx(50 / 150) and rb.shares[EDGES[1]].share == pytest.approx(50 / 150)
    assert row(df, 1, 2001)["total"] + row(df, 2, 2001)["total"] + row(df, 3, 2001)["total"] == pytest.approx(200)
    units = rb.build_units(rb.group_edges())
    text = rebase.write_report(rb, df, units, tmp_path / "r.md", rebase.units_changes(None, units))
    assert "divided by the decision record, data/split_weights.csv (1)" in text and "- C <- A 0.50, B 0.50" in text
    assert "No earlier stable-units file to compare with." in text


def test_a_weight_for_some_but_not_all_parents_stops_the_build():
    with pytest.raises(ValueError, match="1 of its 2 split holders"):
        Rebaser(panel(WORLD), EDGES, districts((1, 2, 3)), MAP, weights={(3, 1, SPLIT): 1.0})


def test_a_weight_row_naming_no_verified_edge_is_reported_not_used(tmp_path):
    rb = Rebaser(panel(WORLD), EDGES, districts((1, 2, 3)), MAP, weights={(2, 1, SPLIT): 1.0})
    assert rb.diag["orphan_weights"] == [(2, 1, SPLIT)] and rb.weights[EDGES[0]] == pytest.approx(0.75)
    df = rb.run()
    text = rebase.write_report(rb, df, rb.build_units(rb.group_edges()), tmp_path / "r.md")
    assert "## Weight rows naming no verified edge (1; ignored)" in text and "- B <- A (split)" in text


def test_read_weights_checks_the_decision_record(tmp_path):
    p = tmp_path / "w.csv"

    def write(rows):
        p.write_text("\n".join([",".join(rebase.WEIGHT_COLUMNS), *rows]) + "\n", encoding="utf-8")

    write(["3,C,1,A,split,0.6,map,human,1,ok", "3,C,2,B,split,0.4,map,human,1,ok", "5,M,4,D,split,0.9,typo,human,0,unverified: ignored"])
    assert rebase.read_weights(p) == {(3, 1, SPLIT): 0.6, (3, 2, SPLIT): 0.4}
    write(["3,C,1,A,split,0.6,map,human,1,", "3,C,2,B,split,0.3,map,human,1,"])
    with pytest.raises(ValueError, match="sum to 0.900"):
        rebase.read_weights(p)
    write(["3,C,1,A,split,0.5,map,human,1,", "3,C,1,A,split,0.5,map,human,1,"])
    with pytest.raises(ValueError, match="appears twice"):
        rebase.read_weights(p)
    assert rebase.read_weights(tmp_path / "missing.csv") == {}


def test_units_changes_lists_groups_that_appeared_or_disappeared():
    old = [{"members": [1, 2]}, {"members": [3]}]
    new = [{"members": [1]}, {"members": [2, 3]}]
    assert rebase.units_changes(old, new) == {"appeared": [[1], [2, 3]], "disappeared": [[1, 2], [3]], "compared": True}
    assert rebase.units_changes(None, new)["compared"] is False and rebase.units_changes(new, new)["appeared"] == []


def test_the_real_decision_record_reads_cleanly():
    weights = rebase.read_weights()
    assert {k[0] for k in weights} == {669, 763} and all(k[2] == SPLIT for k in weights)
    assert sum(w for (c, _p, _k), w in weights.items() if c == 669) == pytest.approx(1.0)
