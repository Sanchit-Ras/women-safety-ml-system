"""unit_map.csv is the human's decision record for 2001-2014 unit names. These pin every stage,
the two model guards, the auto-verify policy and the file contract."""
import datetime
import json

import pandas as pd
import pytest

from agents import store
from agents.llm_client import LLMResult
from agents.prompts import PROJECT_CONTEXT
from agents.resolver import (CSV_COLUMNS, NOT_A_DISTRICT_REASON, ResolveBatch, ResolveItem, UnmappedState, apply,
                             build_pools, geo_match, load_units, read_rows, report, resolve_free, resolve_state_name,
                             run, spellings, write_rows)


def district(code, state, name, aliases, circles=(), seen_in=None):
    return {"district_code": code, "state_code": state, "state_codes_seen": seen_in or [state], "name": name,
            "aliases": list(aliases), "first_seen_year": 2016, "last_seen_year": 2022,
            "circles": [{"norm": c, "name": c.title(), "first_seen_year": 2016, "last_seen_year": 2022} for c in circles]}


REGISTRY = {
    "states": [{"state_code": 28, "name": "Andhra Pradesh", "aliases": ["andhra pradesh"]},
               {"state_code": 36, "name": "Telangana", "aliases": ["telangana"]},
               {"state_code": 10, "name": "Bihar", "aliases": ["bihar"]},
               {"state_code": 24, "name": "Gujarat", "aliases": ["gujarat"]},
               {"state_code": 35, "name": "Andaman And Nicobar Islands", "aliases": ["andaman and nicobar islands"]},
               {"state_code": 1, "name": "Jammu And Kashmir", "aliases": ["jammu and kashmir"]},
               {"state_code": 37, "name": "Ladakh", "aliases": ["ladakh"]},
               {"state_code": 9, "name": "Uttar Pradesh", "aliases": ["uttar pradesh"]},
               {"state_code": 5, "name": "Uttarakhand", "aliases": ["uttarakhand"]},
               {"state_code": 33, "name": "Tamil Nadu", "aliases": ["tamil nadu"]},
               {"state_code": 34, "name": "Puducherry", "aliases": ["puducherry"]}],
    "districts": [
        district(502, 28, "Anantapur", ["anantapur", "ananthapuramu"], ["anantapur", "guntakal railway"]),
        district(504, 28, "Y.S.R.", ["ysr"], ["cuddapah", "ysr"]),
        district(503, 28, "Chittoor", ["chittoor"], ["chittoor", "tirupati urban"]),
        district(505, 28, "East Godavari", ["east godavari"], ["east godavari", "kakinada"]),
        district(522, 36, "Warangal", ["warangal"], ["warangal city", "warangal rural"]),
        district(521, 36, "Hyderabad", ["hyderabad"], ["hyderabad city"]),
        district(203, 10, "Patna", ["patna"], ["patna", "patna railway"]),
        district(204, 10, "Purbi Champaran", ["east champaran", "purbi champaran"]),
        district(205, 10, "Pashchim Champaran", ["west champaran", "pashchim champaran"]),
        district(438, 24, "Ahmadabad", ["ahmedabad", "ahmadabad"], ["ahmedabad city", "ahmedabad rural"]),
        district(469, 24, "Sabarkantha", ["sabarkantha"]),
        district(471, 24, "Aravalli", ["aravalli"]),
        district(632, 35, "North And Middle Andaman", ["north and middle andaman"]),
        district(603, 35, "Nicobars", ["nicobar", "nicobars"]),
        district(9, 37, "Leh Ladakh", ["leh"], ["leh"], seen_in=[1, 37]),          # reported under J&K until 2019
        # the real collisions: one circle norm listed under two districts of one state's pool
        district(162, 9, "Lucknow", ["lucknow"], ["lucknow", "grp"]),
        district(49, 5, "Dehradun", ["dehradun"], ["dehradun", "grp"], seen_in=[5, 9]),   # once reported under UP
        district(568, 33, "Chennai", ["chennai"], ["chennai", "cyber cell", "crime branch chennai"]),
        district(569, 33, "Coimbatore", ["coimbatore"], ["coimbatore", "cyber cell"]),
        district(598, 34, "Karaikal", ["karaikal"], ["karaikal"]),
        district(600, 34, "Puducherry", ["puducherry", "pondicherry"], ["puducherry", "karaikal"]),
    ],
}
UNIT_COLUMNS = ["state", "unit", "first_year", "last_year", "n_years"]


def units(*rows):
    return pd.DataFrame([(s, u, 2001, 2014, 14) for s, u in rows], columns=UNIT_COLUMNS)


def batch(**answers):
    """kadapa=(504, 'rename') -> a ResolveBatch answering those units."""
    return ResolveBatch(items=[ResolveItem(unit=u, district_code=code, kind=kind, confidence=0.9, reason="because")
                               for u, (code, kind) in answers.items()])


class FakeClient:
    """Plays back scripted answers; records what it was asked."""

    def __init__(self, script):
        self.script, self.calls = list(script), []

    def complete(self, prompt, schema, *, tag, system=None):
        self.calls.append((tag, prompt, system))
        return LLMResult(parsed=self.script.pop(0), provider="fake", model="fake-model", cached=False,
                         prompt_tokens=1, completion_tokens=1, latency_ms=1, tries=1, cache_key="k")


@pytest.fixture
def reg_path(tmp_path):
    p = tmp_path / "registry.json"
    p.write_text(json.dumps(REGISTRY), encoding="utf-8")
    return p


@pytest.fixture
def pools():
    return build_pools(REGISTRY)


def run_units(reg_path, tmp_path, rows, client=None, overlap_path=None):
    path = tmp_path / "unit_map.csv"
    summary = run(client, csv_path=path, registry_path=reg_path, overlap_path=overlap_path or tmp_path / "none.csv",
                  units=units(*rows))
    return summary, read_rows(path), path


# ---- inputs -----------------------------------------------------------------------------------

def test_load_units_drops_state_totals_and_counts_distinct_years(tmp_path):
    raw = tmp_path / "era_a.csv"
    pd.DataFrame({"STATE/UT": ["A & N Islands", "A&N Islands", "A & N Islands", "A & N Islands"],
                  "DISTRICT": ["Andaman", "ANDAMAN", "Total", "Nicobar"], "Year": [2001, 2002, 2002, 2003],
                  "Rape": [1, 2, 3, 4]}).to_csv(raw, index=False)
    out = load_units(path=raw)
    assert out.columns.tolist() == UNIT_COLUMNS
    assert out.values.tolist() == [["a n islands", "andaman", 2001, 2002, 2], ["a n islands", "nicobar", 2003, 2003, 1]]


def test_state_names_resolve_by_manual_alias_then_registry_then_fuzzy():
    assert resolve_state_name("a n islands", REGISTRY) == 35
    assert resolve_state_name("Jammu & Kashmir", REGISTRY) == 1               # raw spelling: normalised here
    assert resolve_state_name("bihar", REGISTRY) == 10
    assert resolve_state_name("andhra pradsh", REGISTRY) == 28                # typo: fuzzy last resort
    with pytest.raises(UnmappedState):
        resolve_state_name("atlantis", REGISTRY)


def test_run_names_every_state_it_cannot_map(reg_path, tmp_path):
    with pytest.raises(UnmappedState, match=r"\['atlantis', 'narnia'\]"):
        run_units(reg_path, tmp_path, [("atlantis", "x"), ("narnia", "y"), ("bihar", "patna")])


def test_pools_prefer_a_district_name_over_a_circle_and_absorb_successor_states(pools):
    assert pools[28]["anantapur"] == (502, "alias")                          # alias and circle: alias wins
    assert pools[28]["guntakal railway"] == (502, "circle")
    assert pools[28]["warangal"] == (522, "alias") and pools[28].district_state[522] == 36    # Telangana in AP's pool
    assert "warangal" not in pools[10] and "anantapur" not in pools[36]     # never the other way round
    assert pools[1]["leh"] == (9, "alias") and pools[1].district_state[9] == 37


def test_a_spelling_two_districts_claim_leaves_the_pool_and_is_marked_ambiguous(pools):
    assert "grp" not in pools[9] and pools[9].ambiguous == {"grp": {49, 162}}    # Lucknow's and Dehradun's circle
    assert pools[33].ambiguous == {"cyber cell": {568, 569}}
    assert pools[34]["karaikal"] == (598, "alias") and pools[34].ambiguous == {}   # a district's own name still beats another's circle
    assert pools[9].codes_for("grp") == {49, 162} and pools[9].codes_for("lucknow") == {162}
    assert "grp" in pools[9].names and pools[9].codes == {49, 162}


# ---- deterministic stages ---------------------------------------------------------------------------

def test_exact_is_a_district_spelling(pools):
    res = resolve_free("andhra pradesh", "ananthapuramu", pools[28])
    assert (res.district_code, res.kind, res.stage, res.confidence, res.proposed_by) == (502, "district", "exact", 1.0, "rule")


def test_circle_gives_railway_kind_when_the_name_says_so(pools):
    rail = resolve_free("andhra pradesh", "guntakal railway", pools[28])
    assert (rail.district_code, rail.kind, rail.stage, rail.confidence) == (502, "railway", "circle", 1.0)
    plain = resolve_free("andhra pradesh", "cuddapah", pools[28])
    assert (plain.district_code, plain.kind, plain.stage) == (504, "circle", "circle")


def test_rule_strips_railway_and_commissionerate_words(pools):
    rly = resolve_free("bihar", "patna rly", pools[10])
    assert (rly.district_code, rly.kind, rly.stage, rly.confidence) == (203, "railway", "rule", 1.0)
    grp = resolve_free("bihar", "g r p patna", pools[10])
    assert (grp.district_code, grp.kind, grp.stage) == (203, "railway", "rule")
    commr = resolve_free("gujarat", "ahmedabad commr", pools[24])
    assert (commr.district_code, commr.kind, commr.stage) == (438, "commissionerate", "rule")
    urban = resolve_free("telangana", "warangal urban", pools[36])
    assert (urban.district_code, urban.kind, urban.stage) == (522, "district", "rule")
    assert resolve_free("bihar", "rly", pools[10]).stage == "unresolved"     # nothing left after stripping


def test_rule_strips_one_edge_word_at_a_time_so_a_direction_word_in_the_name_survives(pools):
    res = resolve_free("andhra pradesh", "east godavari rly", pools[28])   # 'east godavari' hits before 'east' comes off
    assert (res.district_code, res.kind, res.stage, res.confidence) == (505, "railway", "rule", 1.0)
    assert spellings("east godavari rly") == [("east godavari rly", []), ("east godavari", ["rly"]), ("godavari", ["rly", "east"])]


def test_successor_stage_for_a_district_now_in_another_state(pools):
    warangal = resolve_free("andhra pradesh", "warangal", pools[28])        # Telangana district under AP, 2001-2013
    assert (warangal.district_code, warangal.kind, warangal.stage, warangal.confidence) == (522, "district", "successor", 1.0)
    hyd = resolve_free("andhra pradesh", "hyderabad city", pools[28])
    assert (hyd.district_code, hyd.kind, hyd.stage) == (521, "circle", "successor")
    leh = resolve_free("jammu kashmir", "leh", pools[1])                   # seen under J&K, belongs to Ladakh now
    assert (leh.district_code, leh.stage) == (9, "successor")
    assert resolve_free("telangana", "warangal", pools[36]).stage == "exact"  # its own state: plain exact


def test_fuzzy_accepts_a_clear_best_district(pools):
    res = resolve_free("andhra pradesh", "cudapah", pools[28])
    assert (res.district_code, res.kind, res.stage, res.proposed_by) == (504, "district", "fuzzy", "fuzzy")
    assert res.confidence == pytest.approx(0.933)
    assert res.candidates[0] == (504, "Y.S.R.", 93.3) and len(res.candidates) == 3


def test_fuzzy_ambiguous_is_left_for_the_model_with_three_hints(pools):
    res = resolve_free("bihar", "champaran", pools[10])                    # East and West Champaran both score 90
    assert (res.district_code, res.kind, res.stage) == (None, "unknown", "unresolved")
    assert res.reason.startswith("ambiguous fuzzy match")
    assert res.candidates == [(204, "Purbi Champaran", 90.0), (205, "Pashchim Champaran", 90.0), (203, "Patna", 54.0)]


def test_not_a_district_regex_wins_over_a_fuzzy_lookalike(pools):
    caw = resolve_free("bihar", "caw", pools[10])
    assert (caw.district_code, caw.kind, caw.stage, caw.confidence, caw.reason) == (None, "not_a_district", "unresolved", 1.0, NOT_A_DISTRICT_REASON)
    crime = resolve_free("bihar", "crime patna", pools[10])                # WRatio 90 against 'patna', still not a district
    assert crime.kind == "not_a_district"
    assert resolve_free("bihar", "cbcid", pools[10]).kind == "not_a_district"
    assert resolve_free("andhra pradesh", "cuddapah", pools[28]).kind != "not_a_district"   # 'cid' must be a whole word
    cyber = resolve_free("tamil nadu", "cyber cell", pools[33])            # a circle of two districts: no identity hit, so the regex decides
    assert (cyber.kind, cyber.stage, cyber.district_code) == ("not_a_district", "unresolved", None)


def test_an_identity_match_that_looks_like_a_police_unit_is_held_for_review(reg_path, tmp_path, pools):
    res = resolve_free("tamil nadu", "crime branch chennai", pools[33])    # a circle of exactly one district, but a crime branch
    assert (res.district_code, res.kind, res.stage, res.confidence, res.needs_review) == (568, "circle", "circle", 1.0, True)
    assert res.reason.endswith("; held for review: the name looks like a police unit with no district geography")
    _, rows, _ = run_units(reg_path, tmp_path, [("tamil nadu", "crime branch chennai"), ("tamil nadu", "chennai")])
    held, plain = rows[("2001_2014", "tamil nadu", "crime branch chennai")], rows[("2001_2014", "tamil nadu", "chennai")]
    assert (held["stage"], held["district_code"], held["proposed_by"], held["verified"]) == ("circle", "568", "rule", "0")
    assert (plain["stage"], plain["verified"]) == ("exact", "1")


def test_a_spelling_two_districts_claim_is_never_an_identity_match(reg_path, tmp_path, pools):
    res = resolve_free("uttar pradesh", "grp", pools[9])
    assert (res.district_code, res.kind, res.stage, res.confidence) == (None, "unknown", "unresolved", 0.0)
    assert res.reason == "'grp' is listed under 2 districts: 49 Dehradun, 162 Lucknow"
    assert res.candidates[:2] == [(49, "Dehradun", 100.0), (162, "Lucknow", 100.0)]   # same score for both: fuzzy cannot pick
    assert resolve_free("uttar pradesh", "grp lucknow", pools[9]).stage == "rule"     # a spelling that identifies one district still does

    _, rows, _ = run_units(reg_path, tmp_path, [("uttar pradesh", "grp")])
    r = rows[("2001_2014", "uttar pradesh", "grp")]
    assert (r["stage"], r["district_code"], r["verified"]) == ("unresolved", "", "0")
    assert r["reason"] == "model stage not run; 'grp' is listed under 2 districts: 49 Dehradun, 162 Lucknow"

    client = FakeClient([batch(grp=(162, "railway"))])
    _, rows, _ = run_units(reg_path, tmp_path, [("uttar pradesh", "grp")], client)
    prompt = client.calls[0][1]
    assert "- grp (seen 2001-2014) hints: 49:Dehradun:100 | 162:Lucknow:100" in prompt
    assert "162 | Lucknow | spellings: lucknow | circles: lucknow, grp" in prompt and "49 | Dehradun" in prompt
    r = rows[("2001_2014", "uttar pradesh", "grp")]
    assert (r["stage"], r["district_code"], r["district_name"], r["kind"], r["verified"]) == ("llm", "162", "Lucknow", "railway", "0")


def overlap_frame(*rows):
    cols = ["old_state", "old_name", "old_norm", "old_state_norm", "new_code", "new_name", "new_state_code", "share", "new_share"]
    return pd.DataFrame([(s.title(), n.title(), n, s, code, name, 24, share, 0.5) for s, n, code, name, share in rows], columns=cols)


def test_geometry_stage_uses_the_old_map_overlap(pools):
    ov = overlap_frame(("gujarat", "himatnagar", 469, "Sabarkantha", 0.97))
    res = resolve_free("gujarat", "himatnagar", pools[24], ov)
    assert (res.district_code, res.kind, res.stage, res.confidence, res.proposed_by) == (469, "district", "geometry", 0.97, "geometry")
    assert (res.geo_code, res.geo_share) == (469, 0.97)
    low = resolve_free("gujarat", "himatnagar", pools[24], overlap_frame(("gujarat", "himatnagar", 469, "Sabarkantha", 0.4)))
    assert (low.stage, low.kind, low.geo_code, low.geo_share) == ("unresolved", "unknown", 469, 0.4)   # below 0.5: shown, not used
    exact = resolve_free("gujarat", "aravalli", pools[24], overlap_frame(("gujarat", "aravalli", 471, "Aravalli", 0.99)))
    assert (exact.stage, exact.geo_code) == ("exact", 471)                 # geo columns filled whatever stage decided
    assert resolve_free("gujarat", "himatnagar", pools[24], None).geo_code is None


def test_geo_match_prefers_the_district_named_like_the_unit_else_the_largest_share():
    ov = overlap_frame(("gujarat", "sabarkantha", 469, "Sabarkantha", 0.45), ("gujarat", "sabarkantha", 471, "Aravalli", 0.55),
                       ("gujarat", "somewhere", 469, "Sabarkantha", 0.45), ("gujarat", "somewhere", 471, "Aravalli", 0.55))
    assert geo_match("gujarat", "sabarkantha", ov) == (469, 0.45)
    assert geo_match("gujarat", "somewhere", ov) == (471, 0.55)
    assert geo_match("bihar", "somewhere", ov) is None


def test_run_reads_the_overlap_csv_and_matches_the_era_a_state_spelling(reg_path, tmp_path):
    ov_path = tmp_path / "area_overlap.csv"
    overlap_frame(("andaman and nicobar islands", "diglipur", 632, "North And Middle Andaman", 0.8),
                  ("andaman and nicobar islands", "nicobar", 603, "Nicobars", 0.99)).to_csv(ov_path, index=False)
    _, rows, _ = run_units(reg_path, tmp_path, [("a n islands", "diglipur"), ("a n islands", "nicobar")], overlap_path=ov_path)
    geo = rows[("2001_2014", "a n islands", "diglipur")]
    assert (geo["stage"], geo["district_code"], geo["district_name"], geo["geo_code"], geo["geo_share"], geo["verified"]) == \
        ("geometry", "632", "North And Middle Andaman", "632", "0.8", "0")
    assert (rows[("2001_2014", "a n islands", "nicobar")]["stage"], rows[("2001_2014", "a n islands", "nicobar")]["geo_code"]) == ("exact", "603")


# ---- the model stage --------------------------------------------------------------------------------

def test_model_stage_gets_one_call_per_state_with_candidates_and_hints(reg_path, tmp_path):
    client = FakeClient([batch(kadapa=(504, "rename")), batch(champaran=(204, "district"))])
    summary, rows, _ = run_units(reg_path, tmp_path, [("andhra pradesh", "kadapa"), ("andhra pradesh", "anantapur"),
                                                      ("bihar", "champaran")], client)
    assert [tag for tag, _, _ in client.calls] == ["resolve:28", "resolve:10"]
    tag, prompt, system = client.calls[0]
    assert system == PROJECT_CONTEXT
    assert "Andhra Pradesh (state code 28)" in prompt
    assert "504 | Y.S.R. | spellings: ysr | circles: cuddapah, ysr | seen 2016-2022" in prompt
    assert "522 | Warangal" in prompt                                         # successor-state districts are candidates too
    assert "- kadapa (seen 2001-2014) hints:" in prompt and "anantapur" not in prompt.split("Names to resolve")[1]
    assert "hints: 204:Purbi Champaran:90 | 205:Pashchim Champaran:90 | 203:Patna:54" in client.calls[1][1]
    r = rows[("2001_2014", "andhra pradesh", "kadapa")]
    assert (r["stage"], r["district_code"], r["district_name"], r["kind"], r["confidence"], r["reason"], r["proposed_by"],
            r["verified"]) == ("llm", "504", "Y.S.R.", "rename", "0.9", "because", "llm:fake-model", "0")
    assert summary["by_stage"] == {"llm": 2, "exact": 1} and summary["pending_states"] == 2


def test_a_code_outside_the_candidate_list_is_rejected(reg_path, tmp_path):
    client = FakeClient([batch(kadapa=(999, "district"), nellur=(203, "district"))])   # 203 is Patna: a real code, wrong state
    _, rows, _ = run_units(reg_path, tmp_path, [("andhra pradesh", "kadapa"), ("andhra pradesh", "nellur")], client)
    r = rows[("2001_2014", "andhra pradesh", "kadapa")]
    assert (r["stage"], r["kind"], r["district_code"], r["district_name"], r["proposed_by"], r["verified"]) == \
        ("unresolved", "unknown", "", "", "llm:fake-model", "0")
    assert r["reason"] == "model proposed code 999 which is not a candidate for this state"
    assert rows[("2001_2014", "andhra pradesh", "nellur")]["reason"] == "model proposed code 203 which is not a candidate for this state"


def test_a_forgotten_unit_gets_one_repair_call_then_is_unresolved(reg_path, tmp_path):
    client = FakeClient([batch(kadapa=(504, "rename")), batch(kadapa=(504, "rename"))])    # forgets nellur twice
    _, rows, _ = run_units(reg_path, tmp_path, [("andhra pradesh", "kadapa"), ("andhra pradesh", "nellur")], client)
    assert [tag for tag, _, _ in client.calls] == ["resolve:28", "resolve:28:repair"]
    assert "Missing names that MUST be included: ['nellur']" in client.calls[1][1]
    assert rows[("2001_2014", "andhra pradesh", "nellur")]["reason"] == "not answered by the model"
    assert rows[("2001_2014", "andhra pradesh", "nellur")]["stage"] == "unresolved"
    assert rows[("2001_2014", "andhra pradesh", "kadapa")]["stage"] == "llm"


def test_without_a_client_pending_rows_say_the_model_stage_was_skipped(reg_path, tmp_path):
    summary, rows, _ = run_units(reg_path, tmp_path, [("andhra pradesh", "kadapa"), ("bihar", "champaran")])
    assert summary["model_stage"] == "skipped" and summary["by_stage"] == {"unresolved": 2}
    assert rows[("2001_2014", "andhra pradesh", "kadapa")]["reason"] == "model stage not run; no deterministic match"
    assert rows[("2001_2014", "bihar", "champaran")]["reason"].startswith("model stage not run; ambiguous fuzzy match")
    assert rows[("2001_2014", "bihar", "champaran")]["proposed_by"] == "none"


# ---- the decision record ------------------------------------------------------------------------------

def test_verified_rows_survive_a_rerun_and_are_not_asked_again(reg_path, tmp_path):
    _, rows, path = run_units(reg_path, tmp_path, [("andhra pradesh", "kadapa")], FakeClient([batch(kadapa=(504, "rename"))]))
    rows[("2001_2014", "andhra pradesh", "kadapa")].update(district_code="503", district_name="Chittoor", stage="human",
                                              proposed_by="human", verified="1", reviewer_note="checked the gazette")
    write_rows(path, rows)
    client = FakeClient([batch(kadapa=(502, "district"))])
    summary, rows, _ = run_units(reg_path, tmp_path, [("andhra pradesh", "kadapa")], client)
    r = rows[("2001_2014", "andhra pradesh", "kadapa")]
    assert (r["district_code"], r["stage"], r["verified"], r["reviewer_note"]) == ("503", "human", "1", "checked the gazette")
    assert client.calls == [] and summary["kept_verified"] == 1


def test_a_verified_row_stays_when_the_input_no_longer_produces_its_key(reg_path, tmp_path):
    _, rows, path = run_units(reg_path, tmp_path, [("andhra pradesh", "kadapa")])
    rows[("2001_2014", "andhra pradesh", "kadapa")].update(district_code="504", district_name="Y.S.R.", stage="human",
                                              proposed_by="human", verified="1", reviewer_note="checked")
    write_rows(path, rows)
    summary, rows, _ = run_units(reg_path, tmp_path, [("andhra pradesh", "anantapur")])    # kadapa gone from the input
    assert list(rows) == [("2001_2014", "andhra pradesh", "anantapur"), ("2001_2014", "andhra pradesh", "kadapa")]
    k = rows[("2001_2014", "andhra pradesh", "kadapa")]
    assert (k["district_code"], k["stage"], k["verified"], k["reviewer_note"]) == ("504", "human", "1", "checked")
    assert (summary["total"], summary["kept_verified"], summary["verified_not_in_input"]) == (2, 0, 1)
    assert summary["by_stage"] == {"exact": 1, "human": 1}


def test_reviewer_note_is_kept_on_unverified_rows_but_the_rest_is_refreshed(reg_path, tmp_path):
    _, rows, path = run_units(reg_path, tmp_path, [("andhra pradesh", "kadapa")])
    rows[("2001_2014", "andhra pradesh", "kadapa")].update(reviewer_note="ask the AP archive", district_code="123")
    write_rows(path, rows)
    _, rows, _ = run_units(reg_path, tmp_path, [("andhra pradesh", "kadapa")], FakeClient([batch(kadapa=(504, "rename"))]))
    r = rows[("2001_2014", "andhra pradesh", "kadapa")]
    assert (r["reviewer_note"], r["district_code"], r["stage"]) == ("ask the AP archive", "504", "llm")


def test_auto_verify_policy(reg_path, tmp_path):
    client = FakeClient([batch(kadapa=(504, "rename"))])
    _, rows, _ = run_units(reg_path, tmp_path, [("andhra pradesh", "anantapur"), ("andhra pradesh", "guntakal railway"),
                                                ("andhra pradesh", "warangal"), ("andhra pradesh", "cudapah"),
                                                ("andhra pradesh", "kadapa"), ("andhra pradesh", "caw"),
                                                ("bihar", "patna rly")], client)
    verified = {r["stage"]: r["verified"] for r in rows.values()}
    assert verified == {"exact": "1", "circle": "1", "successor": "1", "rule": "1", "fuzzy": "0", "llm": "0", "unresolved": "0"}
    assert {r["proposed_by"] for r in rows.values() if r["verified"] == "1"} == {"rule"}


def test_csv_columns_are_exactly_the_contract_and_rows_are_ordered_by_state_then_unit(reg_path, tmp_path):
    _, rows, path = run_units(reg_path, tmp_path, [("bihar", "patna"), ("andhra pradesh", "ysr"), ("andhra pradesh", "anantapur")])
    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert header == ",".join(CSV_COLUMNS)
    assert CSV_COLUMNS == ["era", "state", "state_code", "unit", "first_year", "last_year", "n_years", "district_code",
                           "district_name", "kind", "stage", "confidence", "reason", "candidates", "geo_code",
                           "geo_share", "proposed_by", "verified", "reviewer_note"]
    assert list(rows) == [("2001_2014", "andhra pradesh", "anantapur"), ("2001_2014", "andhra pradesh", "ysr"), ("2001_2014", "bihar", "patna")]
    r = rows[("2001_2014", "andhra pradesh", "anantapur")]
    assert (r["era"], r["state_code"], r["first_year"], r["last_year"], r["n_years"]) == ("2001_2014", "28", "2001", "2014", "14")
    assert r["candidates"].startswith("502:Anantapur:100")


def test_report_writes_the_markdown_file(reg_path, tmp_path):
    _, _, path = run_units(reg_path, tmp_path, [("andhra pradesh", "anantapur"), ("andhra pradesh", "kadapa"), ("bihar", "caw")])
    text = report(path, tmp_path / "reports", today=datetime.date(2026, 9, 27))
    assert (tmp_path / "reports" / "2026-09-27_resolver.md").read_text(encoding="utf-8") == text
    assert "| exact | 1 | 33.3% |" in text and "| unresolved | 2 | 66.7% |" in text
    assert "1 of 2 rows that can have a district have one (50.0%); 1 rows are police units" in text
    assert "### Unresolved (1)" in text and "| andhra pradesh | kadapa |" in text and "caw" not in text.split("### Unresolved")[1]


def test_apply_loads_only_verified_rows_into_the_store(reg_path, tmp_path):
    conn = store.connect(tmp_path / "t.db")
    store.replace_registry(conn, states=[(28, "Andhra Pradesh", "[]"), (10, "Bihar", "[]")],
                           districts=[(502, 28, "Anantapur", "[]", 2016, 2022), (504, 28, "Y.S.R.", "[]", 2016, 2022)], circles=[])
    _, rows, path = run_units(reg_path, tmp_path, [("andhra pradesh", "anantapur"), ("andhra pradesh", "kadapa"), ("bihar", "caw")],
                              FakeClient([batch(kadapa=(504, "rename"))]))
    rows[("2001_2014", "bihar", "caw")]["verified"] = "1"                                  # a human confirms the police-unit call
    write_rows(path, rows)
    assert apply(conn, csv_path=path) == {"loaded": 2, "still_unverified": 1}
    got = {r["unit"]: dict(r) for r in conn.execute("SELECT * FROM unit_map ORDER BY unit")}
    assert set(got) == {"anantapur", "caw"}
    a = got["anantapur"]
    assert (a["era"], a["state"], a["district_code"], a["kind"], a["stage"], a["confidence"], a["verified"]) == \
        ("2001_2014", "andhra pradesh", 502, "district", "exact", 1.0, 1)
    assert (got["caw"]["district_code"], got["caw"]["kind"], got["caw"]["stage"]) == (None, "not_a_district", "unresolved")
    assert apply(conn, csv_path=path)["loaded"] == 2                          # INSERT OR REPLACE: a re-apply is idempotent
    assert conn.execute("SELECT COUNT(*) FROM unit_map").fetchone()[0] == 2

    rows[("2001_2014", "andhra pradesh", "kadapa")].update(verified="1", stage="guess")
    write_rows(path, rows)
    with pytest.raises(ValueError, match="stage 'guess'"):
        apply(conn, csv_path=path)
    conn.close()


def test_a_decision_verified_in_one_era_is_carried_to_the_same_name_in_another(reg_path, tmp_path):
    path = tmp_path / "unit_map.csv"
    run(None, csv_path=path, registry_path=reg_path, overlap_path=tmp_path / "none.csv", units=units(("andhra pradesh", "anantapur")))
    rows = read_rows(path)
    rows[("2001_2014", "andhra pradesh", "anantapur")].update(district_code="504", district_name="Y.S.R.", kind="rename",
                                                           stage="human", verified="1", reviewer_note="decided by hand")
    write_rows(path, rows)
    summary = run(None, era="2015", csv_path=path, registry_path=reg_path, overlap_path=tmp_path / "none.csv",
                  units=units(("andhra pradesh", "anantapur"), ("andhra pradesh", "kadapa")))
    rows = read_rows(path)
    got = rows[("2015", "andhra pradesh", "anantapur")]
    assert (got["district_code"], got["kind"], got["stage"], got["verified"], got["proposed_by"]) == \
        ("504", "rename", "human", "1", "carried:2001_2014")                     # the human's call, not the exact match
    assert got["reviewer_note"] == "carried from 2001_2014: decided by hand"
    assert rows[("2015", "andhra pradesh", "kadapa")]["stage"] == "unresolved"      # a new name still goes through the stages
    assert (summary["carried"], summary["total"]) == (1, 2)
    assert rows[("2001_2014", "andhra pradesh", "anantapur")]["district_code"] == "504"    # the source row is untouched


def test_apply_names_the_row_whose_code_the_store_does_not_know(reg_path, tmp_path):
    conn = store.connect(tmp_path / "t.db")
    geography = dict(states=[(28, "Andhra Pradesh", "[]")], districts=[(502, 28, "Anantapur", "[]", 2016, 2022)], circles=[])
    store.replace_registry(conn, **geography)
    _, rows, path = run_units(reg_path, tmp_path, [("andhra pradesh", "anantapur")])
    rows[("2001_2014", "andhra pradesh", "anantapur")].update(district_code="526", district_name="Bengaluru Rural")
    write_rows(path, rows)
    with pytest.raises(ValueError, match="anantapur: district_code 526 is not in the districts table"):
        apply(conn, csv_path=path)
    store.replace_registry(conn, **geography, map_only=[(526, 28, "Bengaluru Rural")])     # the map-only code arrives
    assert apply(conn, csv_path=path)["loaded"] == 1
    assert tuple(conn.execute("SELECT in_files, first_seen_year FROM districts WHERE district_code = 526").fetchone()) == (0, None)
    conn.close()
