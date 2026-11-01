"""The LangGraph version of the resolver's model step against the hand-built loop, with a scripted model (no network):
the same calls and the same results for the same answers, one repair at most, and the LangChain connector's adapter."""
from dataclasses import asdict

import pytest

pytest.importorskip("langgraph")

from agents import resolver, resolver_graph  # noqa: E402
from agents.resolver import ResolveBatch, ResolveItem  # noqa: E402
from test_resolver import REGISTRY, FakeClient, batch  # noqa: E402

AP = 28


def setup():
    pool = resolver.build_pools(REGISTRY)[AP]
    pending = [resolver.PendingUnit(u, 2001, 2014, resolver.fuzzy_hints(u, pool)) for u in ("kadapa", "nellur", "guntakal rly")]
    return pool, pending


def run_graph(caller, pool, pending, tag="t"):
    graph = resolver_graph.build(caller)
    return graph.invoke({"state_code": AP, "state_name": "Andhra Pradesh", "candidates": resolver.candidate_lines(pool, REGISTRY),
                         "pending": pending, "allowed": frozenset(pool.codes), "tag": f"{tag}:{AP}"})


def scripted():
    first = ResolveBatch(items=[ResolveItem(unit="kadapa", district_code=504, kind="rename", confidence=0.9, reason="because"),
                                ResolveItem(unit="kadapa two", district_code=502, kind="district", confidence=0.5, reason="invented")])
    second = batch(nellur=(999, "district"), **{"guntakal rly": (502, "railway")})   # 999 is no candidate: rejected
    return [first, second]


def test_the_graph_makes_the_same_calls_and_returns_the_same_results_as_the_hand_built_loop():
    pool, pending = setup()
    a_client, b_client = FakeClient(scripted()), FakeClient(scripted())
    a = resolver.resolve_llm(a_client, AP, "Andhra Pradesh", pending, pool, REGISTRY, tag="t")
    out = run_graph(resolver_graph.our_client_caller(b_client), pool, pending)
    assert a_client.calls == b_client.calls                                   # tags, prompts and system text, call by call
    assert [c[0] for c in a_client.calls] == [f"t:{AP}", f"t:{AP}:repair"]
    assert {u: asdict(r) for u, r in a.items()} == {u: asdict(r) for u, r in out["results"].items()}
    assert out["results"]["kadapa"].district_code == 504 and out["results"]["nellur"].stage == "unresolved"
    assert "not a candidate" in out["results"]["nellur"].reason and len(out["calls"]) == 2


def test_the_graph_asks_again_at_most_once():
    pool, pending = setup()
    client = FakeClient([batch(kadapa=(504, "rename")), batch(kadapa=(504, "rename"))])   # two names missing, twice
    out = run_graph(resolver_graph.our_client_caller(client), pool, pending)
    assert len(client.calls) == 2 and out["repairs"] == 1
    assert {u for u, r in out["results"].items() if r.reason == "not answered by the model"} == {"nellur", "guntakal rly"}


def test_no_repair_when_every_name_comes_back():
    pool, pending = setup()
    client = FakeClient([batch(kadapa=(504, "rename"), nellur=(None, "unknown"), **{"guntakal rly": (502, "railway")})])
    out = run_graph(resolver_graph.our_client_caller(client), pool, pending)
    assert len(client.calls) == 1 and out["repairs"] == 0 and out["results"]["guntakal rly"].stage == "llm"


class FakeStructured:
    """Stands in for LangChain's structured-output runnable: returns raw message, parsed object, parsing error."""

    def __init__(self, outputs):
        self.outputs, self.messages = list(outputs), []

    def invoke(self, messages):
        from langchain_core.messages import AIMessage
        self.messages.append(messages)
        parsed = self.outputs.pop(0)
        raw = AIMessage(content="{}", usage_metadata={"input_tokens": 120, "output_tokens": 30, "total_tokens": 150})
        return {"raw": raw, "parsed": parsed, "parsing_error": None if parsed is not None else ValueError("bad json")}


def test_the_langchain_adapter_reports_usage_and_turns_a_bad_reply_into_a_repair():
    pool, pending = setup()
    good = batch(kadapa=(504, "rename"), nellur=(None, "unknown"), **{"guntakal rly": (502, "railway")})
    fake = FakeStructured([None, good])                                       # the first reply does not parse
    out = run_graph(resolver_graph.langchain_caller("m", structured=fake), pool, pending)
    first, second = out["calls"]
    assert (first["ok"], second["ok"], second["prompt_tokens"], second["completion_tokens"], second["model"]) == (False, True, 120, 30, "m")
    assert "bad json" in first["error"] and out["repairs"] == 1 and out["results"]["kadapa"].district_code == 504
    system, human = fake.messages[0]
    assert system.content == resolver_graph.PROJECT_CONTEXT and "Names to resolve" in human.content
    assert "Missing names that MUST be included" in fake.messages[1][1].content


def test_the_diagram_names_every_step():
    text = resolver_graph.diagram(resolver_graph.build(lambda p, s, t: (ResolveBatch(items=[]), {})))
    assert all(step in text for step in ("prompt", "call", "check", "repair", "finish"))
