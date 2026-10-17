"""The live check's name selection and scoring, with a scripted model: no network."""
import pandas as pd

from agents import resolver
from agents.llm_client import LLMResult
from agents.resolver import ResolveBatch, ResolveItem
from evals import live_check


def district(code, state, name, aliases, circles=()):
    return {"district_code": code, "state_code": state, "state_codes_seen": [state], "name": name, "aliases": list(aliases),
            "first_seen_year": 2016, "last_seen_year": 2022,
            "circles": [{"norm": c, "name": c.title(), "first_seen_year": 2016, "last_seen_year": 2022} for c in circles]}


REGISTRY = {"states": [{"state_code": 28, "name": "Andhra Pradesh", "aliases": ["andhra pradesh"]},
                       {"state_code": 10, "name": "Bihar", "aliases": ["bihar"]}],
            "districts": [district(502, 28, "Anantapur", ["anantapur"], ["anantapur"]), district(504, 28, "Y.S.R.", ["ysr"], ["cuddapah"]),
                          district(203, 10, "Patna", ["patna"], ["patna"])]}


def gold_rows(spec):
    """(state_code, state, unit, gold_code) -> gold rows the live check reads."""
    return pd.DataFrame([{"era": "2001_2014", "state": s, "state_code": str(sc), "unit": u, "gold_code": "" if g is None else str(g),
                          "proposed_stage": "llm"} for sc, s, u, g in spec])


def test_select_takes_whole_states_most_names_first_and_cuts_the_last_at_n():
    g = gold_rows([(10, "bihar", f"b{i}", 203) for i in range(5)] + [(28, "andhra pradesh", f"a{i}", 502) for i in range(3)]
                  + [(36, "telangana", "t0", 522)])
    g.loc[len(g)] = {**g.iloc[0].to_dict(), "unit": "fuzzy one", "proposed_stage": "fuzzy"}      # not a model answer: never asked
    picked = live_check.select(g, 6)
    assert picked["state_code"].tolist() == [10] * 5 + [28] and picked["unit"].tolist()[-1] == "a0"
    assert live_check.select(g, 50)["state_code"].tolist() == [10] * 5 + [28] * 3 + [36]
    assert live_check.select(g, 2)["unit"].tolist() == ["b0", "b1"]


class FakeClient:
    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def complete(self, prompt, schema, *, tag, system=None):
        self.calls.append(tag)
        asked = [line[2:].split(" (seen")[0] for line in prompt.splitlines() if line.startswith("- ")]
        items = [ResolveItem(unit=u, district_code=self.answers.get(u), kind="district" if self.answers.get(u) else "not_a_district",
                             confidence=0.9, reason="because") for u in asked]
        return LLMResult(parsed=ResolveBatch(items=items), provider="fake", model="fake-model", cached=False, prompt_tokens=1,
                         completion_tokens=1, latency_ms=1, tries=1, cache_key="k")


def test_ask_scores_each_answer_against_the_gold_code_with_livecheck_tags():
    rows = gold_rows([(28, "andhra pradesh", "cudapah", 504), (28, "andhra pradesh", "caw", None), (10, "bihar", "patna rly", 203)])
    unit_map = {("2001_2014", r.state, r.unit): {"first_year": "2001", "last_year": "2014"} for r in rows.itertuples()}
    rows["state_code"] = rows["state_code"].astype(int)
    client = FakeClient({"cudapah": 504, "caw": None, "patna rly": 502})          # 502 is not a Bihar candidate: rejected, unresolved
    out = live_check.ask(client, rows, unit_map, REGISTRY, resolver.build_pools(REGISTRY))
    assert client.calls == ["livecheck:10", "livecheck:28"]
    by_unit = {r["unit"]: r for r in out}
    assert by_unit["cudapah"]["match"] and by_unit["caw"]["match"] and by_unit["caw"]["answer"] is None
    assert not by_unit["patna rly"]["match"] and by_unit["patna rly"]["kind"] == "unknown"
    assert "live check: 2 of 3 answers match the gold set (66.7%)" in live_check.render(out)
