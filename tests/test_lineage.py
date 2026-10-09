"""lineage.csv is the human's decision record for boundary changes. These pin how candidates are found,
how the model's proposals are guarded and written, and how verified facts become stable units."""
import json

import pandas as pd
import pytest

from agents import store
from agents.lineage import (CSV_COLUMNS, LineageBatch, LineageEdge, apply, build_units, detect_candidates, load_nodes,
                            propose, read_rows, size_distribution, totals_by_district, unusable_edges, write_report,
                            write_rows, write_stable_units)
from agents.llm_client import LLMResult


class FakeClient:
    """Plays back scripted proposals; records what it was asked."""

    def __init__(self, script):
        self.script, self.calls = list(script), []

    def complete(self, prompt, schema, *, tag, system=None):
        self.calls.append((tag, prompt, system))
        return LLMResult(parsed=self.script.pop(0), provider="fake", model="fake-model", cached=False,
                         prompt_tokens=1, completion_tokens=1, latency_ms=1, tries=1, cache_key="k")


def edge(parent, child, kind, year=2017, confidence=0.9, reason="known reorganisation"):
    return LineageEdge(parent_code=parent, child_code=child, kind=kind, year=year, confidence=confidence, reason=reason)


def batch(*edges):
    return LineageBatch(items=list(edges))


def district(code, name, state, first, last, circles=(), aliases=None):
    return {"district_code": code, "state_code": state, "state_codes_seen": [state], "name": name,
            "aliases": aliases if aliases is not None else [name.lower()],
            "first_seen_year": first, "last_seen_year": last,
            "circles": [{"norm": n, "name": nm, "first_seen_year": f, "last_seen_year": l} for n, nm, f, l in circles]}


def csv_row(parent, child, kind, *, verified="0", proposed_by="llm:old", reason="", reviewer_note=""):
    return {"parent_code": str(parent), "parent_name": "", "child_code": str(child), "child_name": "", "kind": kind,
            "year": "", "evidence": "", "confidence": "", "reason": reason, "proposed_by": proposed_by,
            "verified": verified, "reviewer_note": reviewer_note}


STATES = [{"state_code": 36, "name": "Telangana", "aliases": ["telangana"]},
          {"state_code": 28, "name": "Andhra Pradesh", "aliases": ["andhra pradesh"]},
          {"state_code": 8, "name": "Rajasthan", "aliases": ["rajasthan"]}]

REGISTRY = {
    "meta": {"years": [2016, 2022]},
    "states": STATES,
    "districts": [
        district(507, "Hyderabad", 36, 2016, 2022, [("cyberabad commissionerate", "Cyberabad Commissionerate", 2017, 2022),
                                                    ("hyderabad commissionerate", "Hyderabad Commissionerate", 2017, 2022)]),
        district(518, "Ranga Reddy", 36, 2016, 2016, [("cyberabad", "Cyberabad", 2016, 2016),
                                                      ("ranga reddy", "Ranga Reddy", 2016, 2016)]),
        district(516, "Nizamabad", 36, 2016, 2022, [("nizamabad", "Nizamabad", 2016, 2022)]),
        district(685, "Kamareddy", 36, 2017, 2022, [("kamareddy", "Kamareddy", 2017, 2022)]),
        district(502, "Ananthapuramu", 28, 2016, 2022, [("anantapur", "Anantapur", 2016, 2022)], aliases=["anantapur", "ananthapuramu"]),
        district(102, "Jaipur", 8, 2016, 2022, [("jaipur rural", "Jaipur Rural", 2017, 2022)]),
        district(783, "Jaipur (Gramin)", 8, 2016, 2016, [("jaipur rural", "Jaipur Rural", 2016, 2016)]),
    ],
}


def nodes_of(registry, extra=()):
    state_names = {s["state_code"]: s["name"] for s in registry["states"]}
    return [{"district_code": d["district_code"], "name": d["name"], "state_code": d["state_code"],
             "state_name": state_names[d["state_code"]], "in_files": 1} for d in registry["districts"]] + list(extra)


# The node list: every registry code plus one district that exists only in the current GeoJSON.
NODES = nodes_of(REGISTRY, [{"district_code": 9999, "name": "Medchal Malkajgiri", "state_code": 36,
                             "state_name": "Telangana", "in_files": 0}])

OVERLAP = pd.DataFrame([
    {"old_state": "Andhra Pradesh", "old_name": "Nizamabad", "old_norm": "nizamabad", "old_state_norm": "andhra pradesh",
     "new_code": 685, "new_name": "Kamareddy", "new_state_code": 36, "share": 0.4, "new_share": 0.9},
    {"old_state": "Andhra Pradesh", "old_name": "Medak", "old_norm": "medak", "old_state_norm": "andhra pradesh",
     "new_code": 685, "new_name": "Kamareddy", "new_state_code": 36, "share": 0.05, "new_share": 0.1},
    {"old_state": "Rajasthan", "old_name": "Jaipur", "old_norm": "jaipur", "old_state_norm": "rajasthan",
     "new_code": 102, "new_name": "Jaipur", "new_state_code": 8, "share": 1.0, "new_share": 1.0},
])

UNIT_MAP = pd.DataFrame([
    # era, state, state_code, unit, first_year, last_year, n_years, district_code, stage, verified
    ("2001_2014", "andhra pradesh", 28, "nizamabad", 2001, 2014, 14, 516, "exact", 1),
    ("2001_2014", "andhra pradesh", 28, "cyberabad", 2005, 2014, 10, 518, "circle", 0),
    ("2001_2014", "andhra pradesh", 28, "ranga reddy", 2001, 2014, 14, 518, "exact", 1),
    ("2001_2014", "rajasthan", 8, "jaipur rural", 2005, 2014, 10, 102, "circle", 0),          # 102 first seen 2005
    ("2001_2014", "telangana", 36, "kamareddy", 2010, 2014, 5, 685, "exact", 1),               # born after 2016 too
    ("2001_2014", "telangana", 36, "hyderabad", 2001, 2014, 14, 507, "exact", 1),             # continuous since 2001
    ("2001_2014", "andhra pradesh", 28, "anantapur", 2001, 2014, 14, 502, "exact", 1),
    ("2001_2014", "rajasthan", 8, "jaipur gramin", 2001, 2014, 14, 783, "exact", 1),
    ("2001_2014", "rajasthan", 8, "stf", 2003, 2014, 12, None, "unresolved", 0),               # unresolved: ignored
], columns=["era", "state", "state_code", "unit", "first_year", "last_year", "n_years", "district_code", "stage", "verified"])

ALL_HINTS = {("born_after_2014", 685), ("vanished", 518), ("vanished", 783), ("born_in_era_a", 102), ("reported_together", 507)}


def by_hint(cands, hint):
    return {c["child_code"]: c for c in cands if c["kind_hint"] == hint}


def hints(existing):
    return {(c["kind_hint"], c["child_code"]) for c in detect_candidates(REGISTRY, NODES, OVERLAP, UNIT_MAP, existing=existing)}


# ---- candidate detection ------------------------------------------------------------------------------

def test_born_after_2014_needs_parents_and_carries_the_top_overlaps():
    cands = by_hint(detect_candidates(REGISTRY, NODES, OVERLAP, UNIT_MAP), "born_after_2014")
    assert list(cands) == [685]
    c = cands[685]
    assert (c["child_name"], c["state_code"], c["year_hint"]) == ("Kamareddy", 36, 2017)
    assert c["evidence"] == "Nizamabad (Andhra Pradesh): 0.90; Medak (Andhra Pradesh): 0.10"


def test_vanished_codes_carry_their_circles_and_where_those_circles_reappear():
    cands = by_hint(detect_candidates(REGISTRY, NODES), "vanished")
    assert set(cands) == {518, 783}
    rr = cands[518]
    assert rr["year_hint"] == 2017
    assert "last seen 2016, files run to 2022" in rr["evidence"]
    assert "had circle 'Cyberabad' (2016-2016)" in rr["evidence"]
    assert "circle 'cyberabad' also under 507 Hyderabad as 'Cyberabad Commissionerate' (2017-2022)" in rr["evidence"]
    assert "circle 'jaipur rural' also under 102 Jaipur as 'Jaipur Rural' (2017-2022)" in cands[783]["evidence"]


def test_a_unit_first_seen_late_in_era_a_needs_a_parent_unless_it_is_already_born_after_2014():
    cands = by_hint(detect_candidates(REGISTRY, NODES, OVERLAP, UNIT_MAP), "born_in_era_a")
    assert list(cands) == [102]                     # 516/518 seen from 2001; 685 is in (a); the unresolved row is ignored
    c = cands[102]
    assert c["year_hint"] == 2005
    assert c["evidence"] == "first seen in the 2001-2014 file in 2005; Jaipur (Rajasthan): 1.00"
    assert by_hint(detect_candidates(REGISTRY, NODES), "born_in_era_a") == {}       # no unit map, no (c)


def test_a_commissionerate_not_named_after_its_district_hints_pooled_reporting():
    cands = by_hint(detect_candidates(REGISTRY, NODES), "reported_together")
    assert list(cands) == [507]                     # 'Hyderabad Commissionerate' is named after Hyderabad: no hint
    c = cands[507]
    assert c["year_hint"] == 2017
    assert c["evidence"] == "circle 'Cyberabad Commissionerate' (2017-2022) is filed under 507 Hyderabad but is not named after it"


def test_codes_with_a_verified_row_are_skipped_but_unverified_rows_do_not_skip():
    assert hints({}) == ALL_HINTS
    existing = {("516", "685", "split"): csv_row(516, 685, "split", verified="1"),                       # 685 answered as child
                ("518", "507", "reported_together"): csv_row(518, 507, "reported_together", verified="1"),  # 518 as parent, 507 as child
                ("102", "783", "recode"): csv_row(102, 783, "recode", verified="0")}                     # unverified: 783 stays
    assert hints(existing) == {("vanished", 783), ("born_in_era_a", 102)}


def test_skipping_is_role_aware_a_split_where_the_code_is_only_the_parent_answers_nothing_about_it():
    # 685 is the parent of a verified split: where 685 itself came from is still open; 507, the child, is answered
    assert hints({("685", "507", "split"): csv_row(685, 507, "split", verified="1")}) == ALL_HINTS - {("reported_together", 507)}
    # 518 is the parent of a verified split: that does not say where 518 went, so 'vanished' stays; 685 is answered
    assert hints({("518", "685", "split"): csv_row(518, 685, "split", verified="1")}) == ALL_HINTS - {("born_after_2014", 685)}
    # 518 is the parent of a verified merge: where it went is known
    assert hints({("518", "516", "merge"): csv_row(518, 516, "merge", verified="1")}) == ALL_HINTS - {("vanished", 518)}


def test_nodes_come_from_the_lgd_csv_when_it_exists_else_from_the_registry(tmp_path):
    assert [n["district_code"] for n in load_nodes(REGISTRY, tmp_path / "missing.csv")] == [507, 518, 516, 685, 502, 102, 783]
    assert load_nodes(REGISTRY, tmp_path / "missing.csv")[0]["state_name"] == "Telangana"
    lgd = tmp_path / "lgd_districts.csv"
    lgd.write_text("district_code,name,state_code,state_name,in_files\n507,Hyderabad,36,Telangana,1\n"
                   "9999,Medchal Malkajgiri,36,Telangana,0\n", encoding="utf-8")
    nodes = load_nodes(REGISTRY, lgd)
    assert nodes[1] == {"district_code": 9999, "name": "Medchal Malkajgiri", "state_code": 36, "state_name": "Telangana", "in_files": 0}


# ---- model proposal --------------------------------------------------------------------------------------------

@pytest.fixture
def conn(tmp_path):
    c = store.connect(tmp_path / "t.db")
    yield c
    c.close()


def telangana_candidates():
    return [c for c in detect_candidates(REGISTRY, NODES, OVERLAP, UNIT_MAP) if c["state_code"] == 36]


def test_propose_writes_unverified_rows_with_evidence_one_call_per_state(tmp_path):
    path = tmp_path / "lineage.csv"
    client = FakeClient([batch(edge(516, 685, "split"), edge(518, 507, "reported_together", year=2017, confidence=0.8))])
    summary = propose(client, telangana_candidates(), REGISTRY, NODES, csv_path=path)
    assert [t for t, _, _ in client.calls] == ["lineage:36"]
    tag, prompt, system = client.calls[0]
    assert system and system.startswith("You are assisting")
    assert "State: Telangana (code 36)" in prompt
    assert "  507 | Hyderabad | seen 2016-2022 | in data files: yes" in prompt
    assert "  9999 | Medchal Malkajgiri | seen never | in data files: no" in prompt
    assert "  102 | Jaipur" not in prompt                                        # other states stay out
    assert "  502 | Ananthapuramu" in prompt                                    # the successor state's districts are listed
    assert "hint: born_after_2014 | year: 2017 | evidence: Nizamabad (Andhra Pradesh): 0.90" in prompt

    rows = read_rows(path)
    assert list(rows) == [("518", "507", "reported_together"), ("516", "685", "split")]     # every candidate answered
    split = rows[("516", "685", "split")]
    assert (split["parent_name"], split["child_name"], split["year"], split["confidence"]) == ("Nizamabad", "Kamareddy", "2017", "0.90")
    assert (split["verified"], split["proposed_by"], split["reviewer_note"]) == ("0", "llm:fake-model", "")
    assert split["evidence"] == "Nizamabad (Andhra Pradesh): 0.90; Medak (Andhra Pradesh): 0.10"
    assert "Cyberabad" in rows[("518", "507", "reported_together")]["evidence"]
    assert summary["by_state"][36] | {"edges": 2, "placeholders": 0} == summary["by_state"][36]
    assert summary["escalations"] == 0
    with open(path, encoding="utf-8") as fh:
        assert fh.readline().rstrip("\n").split(",") == CSV_COLUMNS


def test_an_edge_with_a_code_outside_the_state_is_dropped_and_escalated(conn, tmp_path):
    path = tmp_path / "lineage.csv"
    cands = [c for c in telangana_candidates() if c["child_code"] == 685]
    client = FakeClient([batch(edge(102, 685, "split"), edge(502, 685, "split", confidence=0.4))])   # Jaipur: no; Anantapur (AP): allowed
    summary = propose(client, cands, REGISTRY, NODES, csv_path=path, conn=conn)
    assert list(read_rows(path)) == [("502", "685", "split")]
    assert summary["escalations"] == 1 and summary["by_state"][36]["dropped"] == 1
    esc = conn.execute("SELECT ref_json, violations_json FROM escalations WHERE kind = 'unit'").fetchall()
    assert len(esc) == 1
    assert json.loads(esc[0]["ref_json"]) == {"child_code": 685, "parent_code": 102}
    assert json.loads(esc[0]["violations_json"])[0]["rule"] == "lineage_code_not_in_state"
    # without a connection the drop is only counted
    assert propose(FakeClient([batch(edge(102, 685, "split"))]), cands, REGISTRY, NODES, csv_path=tmp_path / "b.csv")["escalations"] == 1
    # a code once mis-filed under the state (state_codes_seen) is not in the prompt's lists either: still dropped
    reg = json.loads(json.dumps(REGISTRY))
    next(d for d in reg["districts"] if d["district_code"] == 102)["state_codes_seen"] = [8, 36]
    assert propose(FakeClient([batch(edge(102, 685, "split"))]), cands, reg, NODES, csv_path=tmp_path / "c.csv")["escalations"] == 1


def test_a_candidate_the_model_says_nothing_about_gets_a_placeholder_row(tmp_path):
    path = tmp_path / "lineage.csv"
    summary = propose(FakeClient([batch(edge(516, 685, "split"))]), telangana_candidates(), REGISTRY, NODES, csv_path=path)
    rows = read_rows(path)
    assert set(rows) == {("516", "685", "split"), ("", "507", ""), ("", "518", "")}
    assert summary["by_state"][36]["placeholders"] == 2
    ph = rows[("", "518", "")]
    assert ph["reason"] == "NOT PROPOSED: hints vanished; the model returned no relationship"
    assert (ph["kind"], ph["verified"], ph["proposed_by"], ph["child_name"], ph["year"]) == ("", "0", "rule", "Ranga Reddy", "2017")
    assert "Cyberabad" in ph["evidence"]


def test_a_placeholder_names_every_open_hint_of_the_code(tmp_path):
    registry = {"meta": {"years": [2016, 2022]}, "states": STATES,
                "districts": [district(700, "Bhupalpally", 36, 2017, 2022,
                                       [("warangal commissionerate", "Warangal Commissionerate", 2017, 2022)])]}
    nodes = nodes_of(registry)
    cands = detect_candidates(registry, nodes)
    assert [c["kind_hint"] for c in cands] == ["born_after_2014", "reported_together"]
    propose(FakeClient([batch()]), cands, registry, nodes, csv_path=tmp_path / "l.csv")
    rows = read_rows(tmp_path / "l.csv")
    assert list(rows) == [("", "700", "")]
    ph = rows[("", "700", "")]
    assert ph["reason"] == "NOT PROPOSED: hints born_after_2014, reported_together; the model returned no relationship"
    assert ph["evidence"] == "circle 'Warangal Commissionerate' (2017-2022) is filed under 700 Bhupalpally but is not named after it"
    assert ph["year"] == "2017"


def test_a_placeholder_covers_only_the_hints_no_edge_answers_and_a_split_answers_the_child_only(tmp_path):
    cands = [c for c in telangana_candidates() if c["child_code"] == 518]        # 518: vanished
    propose(FakeClient([batch(edge(518, 685, "split"))]), cands, REGISTRY, NODES, csv_path=tmp_path / "l.csv")
    rows = read_rows(tmp_path / "l.csv")
    assert set(rows) == {("518", "685", "split"), ("", "518", "")}               # the split says nothing about where 518 went
    propose(FakeClient([batch(edge(518, 516, "merge"))]), cands, REGISTRY, NODES, csv_path=tmp_path / "m.csv")
    assert set(read_rows(tmp_path / "m.csv")) == {("518", "516", "merge")}


def test_an_edge_from_one_state_batch_answers_a_candidate_of_its_successor_state(tmp_path):
    registry = {"meta": {"years": [2016, 2022]}, "states": STATES,
                "districts": [district(900, "Old Unit", 28, 2016, 2016), district(901, "New Unit", 36, 2017, 2022)]}
    nodes = nodes_of(registry)
    cands = detect_candidates(registry, nodes)
    assert {(c["state_code"], c["kind_hint"]) for c in cands} == {(28, "vanished"), (36, "born_after_2014")}
    client = FakeClient([batch(edge(900, 901, "recode")), batch()])      # Andhra Pradesh (28) answers first; Telangana says nothing
    summary = propose(client, cands, registry, nodes, csv_path=tmp_path / "l.csv")
    assert [t for t, _, _ in client.calls] == ["lineage:28", "lineage:36"]
    assert list(read_rows(tmp_path / "l.csv")) == [("900", "901", "recode")]    # survives the later batch; no placeholder either side
    assert (summary["by_state"][28]["edges"], summary["by_state"][36]["placeholders"]) == (1, 0)


def test_verified_rows_survive_a_re_propose_and_earlier_guesses_are_not_deleted(tmp_path):
    path = tmp_path / "lineage.csv"
    propose(FakeClient([batch(edge(516, 685, "split"), edge(518, 507, "reported_together"))]), telangana_candidates(),
            REGISTRY, NODES, csv_path=path)
    rows = read_rows(path)
    rows[("516", "685", "split")].update(reason="Kamareddy carved out of Nizamabad, Oct 2016", verified="1",
                                         reviewer_note="checked against the GO")
    write_rows(path, rows)

    again = [batch(edge(516, 685, "split", reason="new guess"), edge(516, 507, "merge"))]
    cands = detect_candidates(REGISTRY, NODES, OVERLAP, UNIT_MAP, existing=read_rows(path))
    assert 685 not in {c["child_code"] for c in cands}                               # not even asked again
    propose(FakeClient(again), [c for c in cands if c["state_code"] == 36], REGISTRY, NODES, csv_path=path)
    rows = read_rows(path)
    kept = rows[("516", "685", "split")]
    assert (kept["reason"], kept["verified"], kept["reviewer_note"]) == ("Kamareddy carved out of Nizamabad, Oct 2016", "1", "checked against the GO")
    assert rows[("518", "507", "reported_together")]["verified"] == "0"         # the earlier guess stays for the human to judge
    assert rows[("516", "507", "merge")]["verified"] == "0"
    assert ("", "518", "") in rows                                                # 518 got no answer this run


def test_human_rows_and_rows_with_a_reviewer_note_are_never_overwritten(tmp_path):
    path = tmp_path / "lineage.csv"
    write_rows(path, {("516", "507", "merge"): csv_row(516, 507, "merge", proposed_by="human", reason="per GO 123"),
                      ("518", "507", "reported_together"): csv_row(518, 507, "reported_together", reason="old guess",
                                                                   reviewer_note="doubtful"),
                      ("516", "685", "split"): csv_row(516, 685, "split", reason="old guess")})
    client = FakeClient([batch(edge(516, 507, "merge", reason="model text"), edge(518, 507, "reported_together", reason="model text"),
                               edge(516, 685, "split", reason="model text"))])
    summary = propose(client, telangana_candidates(), REGISTRY, NODES, csv_path=path)
    rows = read_rows(path)
    assert (rows[("516", "507", "merge")]["reason"], rows[("516", "507", "merge")]["proposed_by"]) == ("per GO 123", "human")
    assert (rows[("518", "507", "reported_together")]["reason"], rows[("518", "507", "reported_together")]["reviewer_note"]) == ("old guess", "doubtful")
    assert (rows[("516", "685", "split")]["reason"], rows[("516", "685", "split")]["proposed_by"]) == ("model text", "llm:fake-model")
    assert summary["by_state"][36]["edges"] == 1                                   # only the plain machine row was rewritten


def test_a_placeholder_goes_away_once_a_later_run_answers_the_code_unless_a_human_annotated_it(tmp_path):
    path = tmp_path / "lineage.csv"
    propose(FakeClient([batch()]), telangana_candidates(), REGISTRY, NODES, csv_path=path)
    assert set(read_rows(path)) == {("", "685", ""), ("", "518", ""), ("", "507", "")}
    rows = read_rows(path)
    rows[("", "507", "")]["reviewer_note"] = "checked: Cyberabad pools nothing"
    write_rows(path, rows)
    propose(FakeClient([batch(edge(518, 516, "merge"), edge(518, 507, "reported_together"))]), telangana_candidates(),
            REGISTRY, NODES, csv_path=path)
    assert set(read_rows(path)) == {("518", "516", "merge"), ("518", "507", "reported_together"), ("", "685", ""), ("", "507", "")}


# ---- stable units -----------------------------------------------------------------------------------------------

def vrow(parent, child, kind, verified="1", year="2017"):
    return {"parent_code": str(parent), "child_code": str(child), "kind": kind, "year": year, "verified": verified}


FOUR = [{"district_code": c, "name": n, "in_files": 1} for c, n in ((1, "A"), (2, "B"), (3, "C"), (4, "D"))]


def test_union_find_joins_verified_edges_and_ignores_unverified_or_unknown_kinds():
    edges = [vrow(1, 2, "split"), vrow(2, 3, "recode"), vrow(3, 4, "bogus"), vrow(1, 4, "merge", verified="0"),
             vrow(4, 77, "split"), vrow(4, 78, "bogus")]                              # 77/78 are not nodes
    units = build_units(edges, FOUR, {1: 10, 2: 50, 3: 50, 4: 5})
    assert [u["members"] for u in units] == [[1, 2, 3], [4]]
    abc, d = units
    assert (abc["unit_id"], abc["name"], abc["largest_member_code"]) == ("su_1", "B", 2)     # 2 and 3 tie at 50: smallest code
    assert abc["edges"] == [{"parent_code": 1, "child_code": 2, "kind": "split", "year": 2017},
                            {"parent_code": 2, "child_code": 3, "kind": "recode", "year": 2017}]
    assert (d["unit_id"], d["name"], d["edges"]) == ("su_4", "D", [])
    assert sorted(c for u in units for c in u["members"]) == [1, 2, 3, 4]
    # a verified edge of a known kind that names a code outside the node list is reported, not silently dropped
    assert unusable_edges(edges, FOUR) == [{"parent_code": 4, "child_code": 77, "kind": "split", "year": 2017}]


def test_all_zero_totals_name_the_group_after_the_smallest_code_and_absent_codes_count_as_zero():
    units = build_units([vrow(3, 2, "rename"), vrow(2, 1, "reported_together", year="")], FOUR, {})
    assert units[0]["name"] == "A" and units[0]["edges"][1]["year"] is None
    assert build_units([vrow(3, 1, "split")], FOUR, {3: 1})[0]["name"] == "C"


def test_stable_units_json_and_report(tmp_path):
    nodes = FOUR + [{"district_code": 5, "name": "E", "in_files": 0}]
    edges = [vrow(1, 2, "split"), vrow(2, 3, "recode"), vrow(4, 5, "split"), vrow(5, 66, "recode")]
    units = build_units(edges, nodes, {5: 9})
    meta = write_stable_units(units, tmp_path / "su.json", unusable_edges(edges, nodes))
    assert meta == {"generated": meta["generated"], "n_nodes": 5, "n_groups": 2, "n_edges_used": 3, "largest_group_size": 3,
                    "n_edges_skipped": 1, "skipped_edges": [{"parent_code": 5, "child_code": 66, "kind": "recode", "year": 2017}]}
    written = json.loads((tmp_path / "su.json").read_text(encoding="utf-8"))
    assert set(written) == {"meta", "units"} and set(written["units"][0]) == {"unit_id", "name", "largest_member_code", "members", "edges"}
    assert size_distribution(units) == {"1": 0, "2": 1, "3-5": 1, "6-10": 0, "11+": 0}
    assert write_stable_units(units, tmp_path / "su2.json")["n_edges_skipped"] == 0

    text = write_report(units, tmp_path / "reports" / "x.md", nodes)
    assert (tmp_path / "reports" / "x.md").read_text(encoding="utf-8") == text
    assert "| 3-5 | 1 |" in text and "| 1 | 0 |" in text
    assert "- su_1 A (3 members): 1 A, 2 B, 3 C" in text
    assert "1 member(s) never appear in the crime files (in_files = 0): 5 E" in text


def test_2022_totals_add_every_category_column_per_district():
    df = pd.DataFrame({"id": [1, 2, 3], "year": [2022, 2022, 2021], "state_code": [36, 36, 36],
                       "district_code": [507, 507, 507], "district_name": ["Hyderabad"] * 3,
                       "rape_women_above_18": [1.0, 2.0, 100.0], "dowry_deaths": [3.0, 0.0, 100.0]})
    assert totals_by_district(df) == {507: 6}


def test_first_seen_years_come_from_the_2001_14_rows_only():
    from agents.lineage import _era_a_first_years
    um = pd.DataFrame([("2001_2014", "x", 28, "u", 2005, 2014, 10, 502, "exact", 1),
                       ("2015", "x", 28, "v", 2015, 2015, 1, 507, "exact", 1)], columns=UNIT_MAP.columns)
    assert _era_a_first_years(um) == {502: 2005}                                  # the 2015 workbook never says "born"


def test_a_district_first_seen_in_2016_with_no_2001_14_unit_is_born_after_2014():
    registry = {"meta": {"years": [2016, 2022]}, "states": STATES,
                "districts": [district(709, "Hojai", 36, 2016, 2022), district(516, "Nizamabad", 36, 2016, 2022)]}
    nodes = nodes_of(registry)
    with_map = {(c["kind_hint"], c["child_code"]) for c in detect_candidates(registry, nodes, None, UNIT_MAP)}
    assert with_map == {("born_after_2014", 709)}             # Nizamabad has a 2001-14 unit, Hojai has none
    assert detect_candidates(registry, nodes) == []            # without a unit map the gap cannot be told from continuity


def test_apply_loads_verified_edges_and_names_an_unknown_code(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    store.replace_registry(conn, states=[(36, "Telangana", "[]")],
                           districts=[(518, 36, "Ranga Reddy", "[]", 2016, 2016), (507, 36, "Hyderabad", "[]", 2016, 2022)],
                           circles=[])
    path = tmp_path / "l.csv"
    rows = {("518", "507", "reported_together"): csv_row(518, 507, "reported_together", verified="1", proposed_by="human",
                                                         reason="Cyberabad is filed under Hyderabad"),
            ("518", "685", "split"): csv_row(518, 685, "split")}
    write_rows(path, rows)
    assert apply(conn, csv_path=path) == {"loaded": 1, "still_unverified": 1}
    got = conn.execute("SELECT parent_code, child_code, kind, year, source, verified FROM district_lineage").fetchone()
    assert tuple(got) == (518, 507, "reported_together", None, "human", 1)
    rows[("518", "685", "split")]["verified"] = "1"                     # 685 is not in this store's districts
    write_rows(path, rows)
    with pytest.raises(ValueError, match=r"518 -> 685 \(split\): district_code 685 is not in the districts table"):
        apply(conn, csv_path=path)
    conn.close()
