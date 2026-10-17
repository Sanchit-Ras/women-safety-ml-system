"""The prompt text lives in prompts/<name>.v<N>.txt and the builders pour data into it. These pin that the files
render exactly the text the pipeline has always sent: golden copies captured from the builders on 2026-10-18,
before the text moved out of the code, and the stored hash of the crosswalk prompt the model actually answered."""
import hashlib
from pathlib import Path

import pytest

from agents import lineage, prompts, resolver
from data_ingestion import crosswalk

GOLDEN = Path(__file__).parent / "golden" / "prompts"


def district(code, state, name, aliases, circles=()):
    return {"district_code": code, "state_code": state, "state_codes_seen": [state], "name": name, "aliases": list(aliases),
            "first_seen_year": 2016, "last_seen_year": 2022,
            "circles": [{"norm": c, "name": c.title(), "first_seen_year": 2016, "last_seen_year": 2022} for c in circles]}


REGISTRY = {"states": [{"state_code": 28, "name": "Andhra Pradesh", "aliases": ["andhra pradesh"]},
                       {"state_code": 36, "name": "Telangana", "aliases": ["telangana"]}],
            "districts": [district(502, 28, "Anantapur", ["anantapur", "ananthapuramu"], ["anantapur", "guntakal railway"]),
                          district(504, 28, "Y.S.R.", ["ysr"], ["cuddapah", "ysr"]),
                          district(522, 36, "Warangal", ["warangal"], ["warangal city"])]}


def resolver_inputs():
    pool = resolver.build_pools(REGISTRY)[28]
    pending = [resolver.PendingUnit(u, 2001, 2013, resolver.fuzzy_hints(u, pool)) for u in ("cudapah", "guntakal rly")]
    return "Andhra Pradesh", 28, resolver.candidate_lines(pool, REGISTRY), pending


def lineage_inputs():
    districts = [{"district_code": 502, "name": "Anantapur", "in_files": 1}, {"district_code": 504, "name": "Y.S.R.", "in_files": 1},
                 {"district_code": 685, "name": "Kamareddy", "in_files": 0}]
    by_code = {502: {"first_seen_year": 2016, "last_seen_year": 2022}, 504: {"first_seen_year": 2016, "last_seen_year": 2019}}
    candidates = [{"child_code": 685, "child_name": "Kamareddy", "kind_hint": "born_after_2014", "year_hint": 2017,
                   "evidence": "Nizamabad (Andhra Pradesh): 0.60"},
                  {"child_code": 504, "child_name": "Y.S.R.", "kind_hint": "vanished", "year_hint": None, "evidence": ""}]
    neighbour = ("Telangana", [{"district_code": 522, "name": "Warangal", "in_files": 1}, {"district_code": 686, "name": "Hanumakonda", "in_files": 0}])
    return "Andhra Pradesh", 28, districts, candidates, by_code, neighbour


CROSSWALK_COLUMNS = ["Rape", "Dowry Deaths", "Total Crimes against Women"]

CASES = {
    "project_context": lambda: prompts.PROJECT_CONTEXT,
    "resolver": lambda: resolver.build_prompt(*resolver_inputs()),
    "resolver_repair": lambda: resolver.build_prompt(*resolver_inputs(), missing={"cudapah"}, unknown={"kadapa"}),
    "lineage": lambda: lineage.build_prompt(*lineage_inputs()[:5]),
    "lineage_neighbour": lambda: lineage.build_prompt(*lineage_inputs()),
    "crosswalk": lambda: crosswalk.build_prompt("2016", CROSSWALK_COLUMNS),
    "crosswalk_repair": lambda: crosswalk.build_prompt("2016", CROSSWALK_COLUMNS, missing={"Rape"}, unknown={"Rapes"}),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_rendered_prompt_is_the_golden_text_captured_before_the_move(name):
    assert CASES[name]() == (GOLDEN / f"{name}.txt").read_text(encoding="utf-8")


def test_every_version_in_use_has_a_file_ending_in_one_newline_that_is_not_part_of_the_prompt():
    for name in prompts.VERSIONS:
        raw = prompts.path(name).read_bytes()
        assert raw.endswith(b"\n") and not raw.endswith(b"\n\n") and b"\r" not in raw, name
        assert prompts.load(name) == raw[:-1].decode("utf-8")


def test_the_fingerprint_names_every_prompt_and_is_stable():
    a = prompts.fingerprint()
    assert set(a) == set(prompts.VERSIONS) and all(len(v) == 12 for v in a.values()) and a == prompts.fingerprint()


def test_repair_lines_are_empty_unless_something_was_missed_or_invented():
    assert prompts.repair_lines("names", 3) == []
    assert prompts.repair_lines("names", 3, missing={"b", "a"}) == ["", "Your previous answer was incomplete.",
                                                                     "Missing names that MUST be included: ['a', 'b']",
                                                                     "Return all 3 names exactly once."]


def test_the_2015_crosswalk_prompt_is_the_one_the_model_answered():
    # llm_calls.prompt_hash of tag crosswalk:2015 (gemini-3.5-flash-lite, 2026-10-10): the pipeline's cache finds that
    # answer again only while this text is unchanged to the byte.
    text = crosswalk.build_prompt("2015", crosswalk.era_columns("2015"))
    assert hashlib.sha256(text.encode()).hexdigest() == "11211fc45759f37a5da2510e11e592f2f3ba2d3b3a79297bf70ab869806e99ad"
