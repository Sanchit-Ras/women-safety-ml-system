"""The framework comparison's machinery without a live model: the input groups, the turn-taking runner with a failing
state, the scores, the agreement counts, the pacer, and the code-line counter."""
import json

import pandas as pd
import pytest

pytest.importorskip("langgraph")

from agents import resolver, resolver_graph  # noqa: E402
from evals import compare_frameworks as cf  # noqa: E402
from test_resolver import REGISTRY, FakeClient, batch  # noqa: E402
from test_resolver_graph import FakeStructured  # noqa: E402


def gold_rows():
    rows = [("2001_2014", "andhra pradesh", 28, "kadapa", 504, "llm"), ("2001_2014", "andhra pradesh", 28, "nellur", 503, "llm"),
            ("2001_2014", "bihar", 10, "patna rly", "", "llm"), ("2001_2014", "bihar", 10, "patna", 203, "exact")]
    return pd.DataFrame([{"era": e, "state": s, "state_code": str(c), "unit": u, "gold_code": str(g), "proposed_stage": st}
                         for e, s, c, u, g, st in rows])


UNIT_MAP = {("2001_2014", s, u): {"first_year": "2001", "last_year": "2014"}
            for s, u in (("andhra pradesh", "kadapa"), ("andhra pradesh", "nellur"), ("bihar", "patna rly"))}


def fake_runners(scripts):
    """A, B and C on scripted models: A and B on FakeClient, C on the LangChain adapter with a fake connector."""
    clients = {b: FakeClient(list(scripts[b])) for b in ("A", "B")}
    graph_b = resolver_graph.build(resolver_graph.our_client_caller(clients["B"]))
    graph_c = resolver_graph.build(resolver_graph.langchain_caller("m", structured=FakeStructured(list(scripts["C"]))))

    def state_of(g, tag):
        return {"state_code": g["state_code"], "state_name": g["state_name"], "candidates": g["candidates"], "pending": g["pending"],
                "allowed": frozenset(g["pool"].codes), "tag": f"{tag}:{g['state_code']}"}

    def run_a(g, tag):
        if g["state_code"] == 10:
            raise RuntimeError("quota exhausted")                        # one state fails outright for A
        return resolver.resolve_llm(clients["A"], g["state_code"], g["state_name"], g["pending"], g["pool"], REGISTRY, tag=tag), []

    def run_b(g, tag):
        out = graph_b.invoke(state_of(g, tag))
        return out["results"], out["calls"]

    def run_c(g, tag):
        out = graph_c.invoke(state_of(g, tag))
        return out["results"], out["calls"]

    return {"A": run_a, "B": run_b, "C": run_c}


def test_groups_hold_the_models_names_by_file_and_state_with_the_key():
    gs = cf.groups(gold_rows(), UNIT_MAP, REGISTRY)
    assert [(g["state_code"], [p.unit for p in g["pending"]]) for g in gs] == [(10, ["patna rly"]), (28, ["kadapa", "nellur"])]
    assert gs[1]["key"] == {"kadapa": 504, "nellur": 503} and gs[0]["key"] == {"patna rly": None}
    assert gs[1]["state_name"] == "Andhra Pradesh" and gs[1]["candidates"]


def test_the_runner_takes_turns_and_scores_each_build():
    gs = cf.groups(gold_rows(), UNIT_MAP, REGISTRY)
    good_bihar, good_ap = batch(**{"patna rly": (None, "railway")}), batch(kadapa=(504, "rename"), nellur=(503, "district"))
    scripts = {"A": [good_ap], "B": [good_bihar, good_ap], "C": [good_bihar, batch(kadapa=(504, "rename"), nellur=(999, "district"))]}
    pace = cf.Pacer(0.0)
    runs = cf.run(gs, lambda r: fake_runners(scripts), pace, rounds=1)
    a, b, c = (cf.score(runs[x][1], gs) for x in ("A", "B", "C"))
    assert (a["correct"], a["failed_states"], a["failed_names"]) == (2, 1, 1) and "quota exhausted" in a["errors"][0]
    assert (b["correct"], b["names"], b["calls"]) == (3, 3, 2) and b["ci95"][0] < 1.0 == b["precision"]
    assert (c["correct"], c["invented_rejected"], c["prompt_tokens_mean"]) == (2, 1, 120)
    assert cf.agreement(runs["B"][1], runs["C"][1], gs) == {"same": 2, "names": 3, "share": 2 / 3}
    assert all(o["seconds"] >= 0 for x in ("A", "B", "C") for o in runs[x][1])
    raw = cf.raw_answers(runs, gs)
    assert raw["B"]["1"]["answers"]["2001_2014|28|kadapa"] == ["llm", 504, "rename"]
    assert raw["A"]["1"]["answers"]["2001_2014|10|patna rly"] is None and "quota exhausted" in raw["A"]["1"]["errors"][0]
    assert raw["C"]["1"]["answers"]["2001_2014|28|nellur"][0] == "unresolved"
    m = {"date": "2026-11-01", "model": "m", "rounds": 1, "groups": len(gs), "names": 3, "spacing_s": 5.0,
         **cf.summarise(runs, gs, 1), "c_warnings": ["temperature will be ignored"],
         "static": {"control_code_lines": {"A": 14, "B": 35, "C": 51}, "diagram": "graph TD;",
                    "footprint": {"packages_added": ["langgraph"], "count": 1, "mb": 1.0}}}
    text = cf.render(m)
    assert "- temperature will be ignored" in text and "| answers matching the key, round by round | 2 of 3 | 3 of 3 | 2 of 3 |" in text
    assert "```mermaid" in text and "## Reading" in text and "the builds did not always agree" in text
    again = cf.render(json.loads(json.dumps(m)))                          # round keys become strings in the saved JSON
    assert again == text


def test_the_pacer_spaces_calls_and_counts_the_wait():
    t = {"now": 0.0}
    clock, waits = (lambda: t["now"]), []

    def sleep(s):
        waits.append(s)
        t["now"] += s

    pace = cf.Pacer(5.0, clock=clock, sleep=sleep)
    pace()
    t["now"] += 2.0
    pace()
    t["now"] += 7.0
    pace()
    assert waits == [3.0] and pace.waited == 3.0


def test_code_lines_leave_out_docstrings_comments_and_blank_lines():
    def f(x):
        """A docstring
        on two lines."""
        # a comment

        return x + 1
    assert cf.code_lines(f) == 2
