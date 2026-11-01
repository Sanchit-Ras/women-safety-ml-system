"""The resolver's model step built with LangGraph, for a measured comparison with the hand-built loop in
``resolver.resolve_llm``. Nothing in the pipeline imports this module; ``evals/compare_frameworks.py`` does.

The graph is the same loop, drawn as steps:

    prompt -> call -> check --(names missing or invented, repair not used yet)--> repair -> call
                            \\--(every name back, or the one repair used)----------> finish

Every step except ``call`` is the resolver's own code (its prompt builder and its guards), so a comparison measures
the framework, not a rewrite. ``call`` takes any function (prompt, system, tag) -> (ResolveBatch, usage): our model
client (``our_client_caller``) or LangChain's Gemini connector (``langchain_caller``).

Needs requirements-frameworks.txt.
"""
from __future__ import annotations

import time
from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from agents import resolver
from agents.llm_client import SEED, TEMPERATURE
from agents.prompts import PROJECT_CONTEXT
from data_ingestion.normalize import norm

Usage = dict[str, Any]          # prompt_tokens, completion_tokens, latency_ms, model, ok, error
Caller = Callable[[str, str, str], tuple[resolver.ResolveBatch, Usage]]
MAX_REPAIRS = 1                 # as in resolve_llm: one more call when names are missing or invented, never two


class State(TypedDict, total=False):
    state_code: int
    state_name: str
    candidates: list[str]
    pending: list[resolver.PendingUnit]
    allowed: frozenset[int]
    tag: str
    prompt: str
    answers: dict[str, resolver.ResolveItem]
    missing: set[str]
    unknown: set[str]
    repairs: int
    calls: list[Usage]
    results: dict[str, resolver.Resolution]


def build(caller: Caller, max_repairs: int = MAX_REPAIRS):
    """The compiled graph. Invoke it with state_code, state_name, candidates, pending, allowed and tag; read
    ``results`` (one Resolution per pending name) and ``calls`` (one usage record per model call)."""

    def prompt(s: State) -> dict:
        return {"prompt": resolver.build_prompt(s["state_name"], s["state_code"], s["candidates"], s["pending"]),
                "answers": {}, "repairs": 0, "calls": []}

    def call(s: State) -> dict:
        batch, usage = caller(s["prompt"], PROJECT_CONTEXT, s["tag"] + (":repair" if s["repairs"] else ""))
        return {"answers": {**s["answers"], **{norm(i.unit): i for i in batch.items}}, "calls": [*s["calls"], usage]}

    def check(s: State) -> dict:
        want, got = {p.unit for p in s["pending"]}, set(s["answers"])
        return {"missing": want - got, "unknown": got - want}

    def route(s: State) -> str:
        return "repair" if (s["missing"] or s["unknown"]) and s["repairs"] < max_repairs else "finish"

    def repair(s: State) -> dict:
        return {"repairs": s["repairs"] + 1,
                "prompt": resolver.build_prompt(s["state_name"], s["state_code"], s["candidates"], s["pending"],
                                                missing=s["missing"], unknown=s["unknown"])}

    def finish(s: State) -> dict:
        model = s["calls"][-1].get("model") or "unknown"
        return {"results": resolver.answers_to_resolutions(s["pending"], s["answers"], s["allowed"], f"llm:{model}")}

    g = StateGraph(State)
    for name, fn in (("prompt", prompt), ("call", call), ("check", check), ("repair", repair), ("finish", finish)):
        g.add_node(name, fn)
    g.add_edge(START, "prompt")
    g.add_edge("prompt", "call")
    g.add_edge("call", "check")
    g.add_conditional_edges("check", route, {"repair": "repair", "finish": "finish"})
    g.add_edge("repair", "call")
    g.add_edge("finish", END)
    return g.compile()


def our_client_caller(client) -> Caller:
    """Version B: the graph calls our own model client (its cache, rate limits, retries and cost records)."""

    def call(prompt: str, system: str, tag: str):
        r = client.complete(prompt, resolver.ResolveBatch, tag=tag, system=system)
        return r.parsed, {"prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
                          "latency_ms": r.latency_ms, "model": r.model, "ok": True, "error": None}

    return call


def langchain_caller(model: str, api_key: str | None = None, *, max_output_tokens: int = 8192, timeout_s: int = 90,
                     structured=None) -> Caller:
    """Version C: the graph calls LangChain's Gemini connector, set up the standard way (structured output with a JSON
    schema), with our client's settings where the connector has them: temperature, seed, output cap, timeout and the
    four safety filters off. ``structured`` replaces the connector in tests. A reply that does not parse comes back as
    an empty batch, so the graph's check asks again, as it would for any missing names."""
    if structured is None:
        from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
        safety = {c: HarmBlockThreshold.BLOCK_NONE for c in (HarmCategory.HARM_CATEGORY_HARASSMENT, HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                                             HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                                             HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT)}
        llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=TEMPERATURE, seed=SEED,
                                     max_output_tokens=max_output_tokens, timeout=timeout_s, safety_settings=safety)
        structured = llm.with_structured_output(resolver.ResolveBatch, method="json_schema", include_raw=True)
    from langchain_core.messages import HumanMessage, SystemMessage

    def call(prompt: str, system: str, tag: str):
        started = time.perf_counter()
        out = structured.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
        latency_ms = int((time.perf_counter() - started) * 1000)
        raw, parsed = out.get("raw"), out.get("parsed")
        um = getattr(raw, "usage_metadata", None) or {}
        usage = {"prompt_tokens": um.get("input_tokens"), "completion_tokens": um.get("output_tokens"), "latency_ms": latency_ms,
                 "model": model, "ok": parsed is not None,
                 "error": None if parsed is not None else str(out.get("parsing_error") or "no parsed output")[:200]}
        return (parsed if parsed is not None else resolver.ResolveBatch(items=[])), usage

    return call


def diagram(graph) -> str:
    """The graph as Mermaid text, which GitHub draws inside a markdown report."""
    return graph.get_graph().draw_mermaid()
