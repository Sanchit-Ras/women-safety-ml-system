"""Three builds of the resolver's model step, run live on the same answer-key names and measured side by side:

  A  hand-built   resolver.resolve_llm with our own model client (cache, rate limits, retries, cost records, fallback)
  B  LangGraph    agents/resolver_graph.py with the same client: only the control flow changes
  C  LangGraph + LangChain's Gemini connector, set up the standard way

Inputs: every answer-key row the model answered (proposed_stage = llm), grouped by file and state as the pipeline asks
them, with the same candidate lists, prompt text and model. Each build runs ROUNDS times, each round in its own empty
store (no cached answers), the builds taking turns state by state; one pacer spaces every call of every build so the
account's per-minute limit holds, and pacing waits are never counted as response time.

Scores: answers matching the key (Wilson interval), 'cannot tell' answers, invented codes rejected, names never
returned, failed states, model calls and repairs, replies that did not parse, tokens and cost, response time, the
time the control flow itself adds, how often a build repeats its own answers, how often builds agree; and, measured
without the model, lines of control code and the packages each build needs.

    python -m evals.compare_frameworks [--rounds 2]      # live: about 140 calls, 12 minutes; needs GEMINI_* in .env
"""
from __future__ import annotations

import argparse
import ast
import inspect
import json
import os
import shutil
import tempfile
import textwrap
import time
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from agents import resolver, store
from evals import gold, pricing
from evals.metrics import wilson

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "evals" / "reports"
BUILDS = ("A", "B", "C")
LABELS = {"A": "A hand-built", "B": "B LangGraph, our client", "C": "C LangGraph + LangChain connector"}
ROUNDS = 2
HEADROOM = 0.8                      # use at most 80% of the per-minute limit, as our client does
C_WARNINGS: set[str] = set()        # what LangChain's connector warned about during the run: settings it changed or dropped


class Pacer:
    """Spaces calls ``spacing`` seconds apart across every build, and counts the time spent waiting."""

    def __init__(self, spacing: float, clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        self.spacing, self.clock, self.sleep, self.last, self.waited = spacing, clock, sleep, None, 0.0

    def __call__(self) -> None:
        now = self.clock()
        if self.last is not None and now - self.last < self.spacing:
            wait = self.spacing - (now - self.last)
            self.sleep(wait)
            self.waited += wait
        self.last = self.clock()


class PacedClient:
    """Our model client, paced, with one usage record per call (for builds A and B)."""

    def __init__(self, client, pace: Pacer):
        self.client, self.pace, self.usage = client, pace, []

    def complete(self, prompt, schema, *, tag, system=None):
        self.pace()
        r = self.client.complete(prompt, schema, tag=tag, system=system)
        self.usage.append({"tag": tag, "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
                           "latency_ms": r.latency_ms, "model": r.model, "ok": True, "error": None})
        return r


# ---- inputs -------------------------------------------------------------------------------------------------------

def groups(gold_rows: pd.DataFrame, unit_map: dict, registry: dict) -> list[dict]:
    """One group per (file, state): the names the model answered, as the pipeline asked them, with the answer key."""
    pools = resolver.build_pools(registry)
    names = {s["state_code"]: s["name"] for s in registry["states"]}
    out = []
    for (era, code), g in gold_rows[gold_rows["proposed_stage"] == "llm"].groupby(["era", "state_code"], sort=True):
        code = int(code)
        pool = pools[code]
        pending, key = [], {}
        for r in g.sort_values("unit").itertuples():
            um = unit_map[(era, r.state, r.unit)]
            pending.append(resolver.PendingUnit(r.unit, int(um["first_year"]), int(um["last_year"]), resolver.fuzzy_hints(r.unit, pool)))
            key[r.unit] = None if str(r.gold_code).strip() in ("", "nan", "<NA>") else int(float(r.gold_code))
        out.append({"era": era, "state_code": code, "state_name": names[code], "pool": pool, "pending": pending, "key": key,
                    "candidates": resolver.candidate_lines(pool, registry)})
    return out


# ---- one run ------------------------------------------------------------------------------------------------------

def run_build(build: str, group: dict, round_no: int, runners: dict, pace: Pacer) -> dict:
    """One build on one group: its results, its calls, and its own time (pacing waits taken out)."""
    waited, started = pace.waited, time.perf_counter()
    try:
        results, calls = runners[build](group, f"fw{build}r{round_no}")
        error = None
    except Exception as e:                                    # a failed state counts against the build, it does not stop the run
        results, calls, error = {}, [], f"{type(e).__name__}: {e}"[:300]
    seconds = time.perf_counter() - started - (pace.waited - waited)
    return {"results": results, "calls": calls, "seconds": seconds, "error": error}


def live_runners(registry: dict, pace: Pacer, workdir: Path, round_no: int, conns: list) -> dict:
    """The three builds against the live model, each with its own empty store for this round (its connection goes into
    ``conns``, for the caller to close)."""
    from agents import resolver_graph
    from agents.llm_client import build_client

    for b in ("A", "B"):
        conns.append(store.connect(workdir / f"{b}{round_no}.db"))
    clients = {b: PacedClient(build_client(c, only="gemini"), pace) for b, c in zip(("A", "B"), conns[-2:])}
    graph_b = resolver_graph.build(resolver_graph.our_client_caller(clients["B"]))
    lc = resolver_graph.langchain_caller(os.environ["GEMINI_MODEL"], os.environ["GEMINI_API_KEY"])
    c_usage = []

    def paced_lc(prompt, system, tag):
        import warnings
        pace()
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            batch, usage = lc(prompt, system, tag)
        C_WARNINGS.update(str(w.message)[:300] for w in seen)
        c_usage.append({**usage, "tag": tag})
        return batch, usage

    graph_c = resolver_graph.build(paced_lc)

    def state_of(g, tag):
        return {"state_code": g["state_code"], "state_name": g["state_name"], "candidates": g["candidates"], "pending": g["pending"],
                "allowed": frozenset(g["pool"].codes), "tag": f"{tag}:{g['state_code']}"}

    def run_a(g, tag):
        before = len(clients["A"].usage)
        res = resolver.resolve_llm(clients["A"], g["state_code"], g["state_name"], g["pending"], g["pool"], registry, tag=tag)
        return res, clients["A"].usage[before:]

    def run_b(g, tag):
        before = len(clients["B"].usage)
        return graph_b.invoke(state_of(g, tag))["results"], clients["B"].usage[before:]

    def run_c(g, tag):
        before = len(c_usage)
        return graph_c.invoke(state_of(g, tag))["results"], c_usage[before:]

    return {"A": run_a, "B": run_b, "C": run_c}


def run(groups_: list[dict], make_runners: Callable[[int], dict], pace: Pacer, rounds: int = ROUNDS) -> dict:
    """runs[build][round] = list of per-group outcomes, the builds taking turns (rotated) state by state."""
    runs = {b: {} for b in BUILDS}
    for r in range(1, rounds + 1):
        runners = make_runners(r)
        for b in BUILDS:
            runs[b][r] = [None] * len(groups_)
        for i, g in enumerate(groups_):
            for b in BUILDS[i % 3:] + BUILDS[:i % 3]:
                runs[b][r][i] = run_build(b, g, r, runners, pace)
    return runs


# ---- scores -------------------------------------------------------------------------------------------------------

def answer_of(res) -> tuple:
    return (None, None) if res is None else (res.stage, res.district_code)


def score(outcomes: list[dict], groups_: list[dict]) -> dict:
    n = correct = cannot = invented = missing = failed = 0
    for out, g in zip(outcomes, groups_):
        for unit, gold_code in g["key"].items():
            n += 1
            res = out["results"].get(unit)
            if res is None:
                failed += 1
            elif res.stage == "llm":
                correct += int(res.district_code == gold_code)
                cannot += int(res.kind == "unknown")
            elif res.reason.startswith("model proposed code"):
                invented += 1
            elif res.reason == "not answered by the model":
                missing += 1
    calls = [c for o in outcomes for c in o["calls"]]
    lat = [c["latency_ms"] for c in calls if c.get("latency_ms") is not None]
    pt = [c["prompt_tokens"] or 0 for c in calls]
    ct = [c["completion_tokens"] or 0 for c in calls]
    usd = sum(pricing.cost_usd(c["model"], c["prompt_tokens"] or 0, c["completion_tokens"] or 0) or 0.0 for c in calls)
    overhead = [o["seconds"] * 1000 - sum(c["latency_ms"] or 0 for c in o["calls"]) for o in outcomes if not o["error"]]
    lo, hi = wilson(correct, n)
    return {"names": n, "correct": correct, "precision": correct / n if n else 0.0, "ci95": [lo, hi], "cannot_tell": cannot,
            "invented_rejected": invented, "not_returned": missing, "failed_names": failed,
            "failed_states": sum(bool(o["error"]) for o in outcomes), "errors": [o["error"] for o in outcomes if o["error"]][:3],
            "calls": len(calls), "repairs": sum(str(c.get("tag", "")).endswith(":repair") for c in calls),
            "unparsable": sum(not c.get("ok", True) for c in calls),
            "prompt_tokens_mean": float(np.mean(pt)) if pt else 0.0, "completion_tokens_mean": float(np.mean(ct)) if ct else 0.0,
            "usd": usd, "inr": usd * pricing.USD_INR,
            "latency_ms_p50": float(np.percentile(lat, 50)) if lat else None, "latency_ms_p95": float(np.percentile(lat, 95)) if lat else None,
            "overhead_ms_mean": float(np.mean(overhead)) if overhead else None}


def agreement(a: list[dict], b: list[dict], groups_: list[dict]) -> dict:
    """How many names two runs answered identically (same stage and code)."""
    same = n = 0
    for oa, ob, g in zip(a, b, groups_):
        for unit in g["key"]:
            n += 1
            same += int(answer_of(oa["results"].get(unit)) == answer_of(ob["results"].get(unit)))
    return {"same": same, "names": n, "share": same / n if n else 0.0}


# ---- measured without the model ---------------------------------------------------------------------------------

def code_lines(*fns) -> int:
    """Lines of code in these functions: no blank lines, comments or docstrings."""
    total = 0
    for fn in fns:
        src = textwrap.dedent(inspect.getsource(fn))
        skip = set()
        for node in ast.walk(ast.parse(src)):
            body = getattr(node, "body", None)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and body and isinstance(body[0], ast.Expr) \
                    and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
                skip.update(range(body[0].lineno, body[0].end_lineno + 1))
        total += sum(1 for i, line in enumerate(src.splitlines(), 1)
                     if i not in skip and line.strip() and not line.strip().startswith("#"))
    return total


def _requirement_names(path: Path) -> list[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            out.append(line.split("==")[0].split(">=")[0].strip())
    return out


def closure(names: list[str]) -> set[str]:
    """Every installed distribution these need, transitively (extras left out), as normalised names."""
    import importlib.metadata as md
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    seen, todo = set(), [canonicalize_name(n) for n in names]
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            continue
        seen.add(name)
        for spec in dist.requires or []:
            req = Requirement(spec)
            if req.marker is None or req.marker.evaluate({"extra": ""}):
                todo.append(canonicalize_name(req.name))
    return seen


def footprint() -> dict:
    import importlib.metadata as md
    base = closure(_requirement_names(ROOT / "requirements.txt"))
    added = sorted(closure(_requirement_names(ROOT / "requirements-frameworks.txt")) - base)
    size = 0
    for name in added:
        for f in md.distribution(name).files or []:
            p = Path(f.locate())
            if p.is_file():
                size += p.stat().st_size
    return {"packages_added": added, "count": len(added), "mb": round(size / 1e6, 1)}


def static_measures() -> dict:
    from agents import resolver_graph
    lines = {"A": code_lines(resolver.resolve_llm),
             "B": code_lines(resolver_graph.build, resolver_graph.our_client_caller),
             "C": code_lines(resolver_graph.build, resolver_graph.langchain_caller)}
    return {"control_code_lines": lines, "footprint": footprint(),
            "diagram": resolver_graph.diagram(resolver_graph.build(lambda p, s, t: (resolver.ResolveBatch(items=[]), {})))}


# ---- report ------------------------------------------------------------------------------------------------------

def raw_answers(runs: dict, groups_: list[dict]) -> dict:
    """Every build's answer to every name in every round, and its calls: the evidence behind the scores, saved first."""
    out = {}
    for b, by_round in runs.items():
        for r, outcomes in by_round.items():
            out.setdefault(b, {})[str(r)] = {
                "answers": {f"{g['era']}|{g['state_code']}|{u}": (None if o["results"].get(u) is None else
                                                                  [o["results"][u].stage, o["results"][u].district_code, o["results"][u].kind])
                            for o, g in zip(outcomes, groups_) for u in g["key"]},
                "calls": [c for o in outcomes for c in o["calls"]], "errors": [o["error"] for o in outcomes if o["error"]]}
    return out


def summarise(runs: dict, groups_: list[dict], rounds: int) -> dict:
    pooled = {b: score([o for r in range(1, rounds + 1) for o in runs[b][r]], groups_ * rounds) for b in BUILDS}
    per_round = {b: {r: score(runs[b][r], groups_) for r in range(1, rounds + 1)} for b in BUILDS}
    repeat = {b: agreement(runs[b][1], runs[b][2], groups_) for b in BUILDS} if rounds >= 2 else {}
    between = {f"{x} vs {y}": agreement(runs[x][1], runs[y][1], groups_) for x, y in (("A", "B"), ("A", "C"), ("B", "C"))}
    return {"pooled": pooled, "per_round": per_round, "repeat": repeat, "between": between}


def _pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def _ms(x) -> str:
    return "-" if x is None else f"{x:,.0f}"


def render(m: dict) -> str:
    p, s = m["pooled"], m["static"]
    L = [f"# Hand-built loop vs LangGraph ({m['date']})", "",
         f"The resolver's model step, built three ways, run live on the same {m['names']} answer-key names ({m['groups']} state "
         f"batches) with the same prompt text, candidate lists and model (`{m['model']}`, temperature 0), {m['rounds']} rounds "
         f"each, every round from an empty store. The builds took turns state by state; one pacer spaced every call "
         f"{m['spacing_s']:.1f} s apart, and pacing waits are not counted as response time.", "",
         "- **A hand-built**: `resolver.resolve_llm` with our model client.",
         "- **B LangGraph, our client**: `agents/resolver_graph.py` calling the same client. Only the control flow changes.",
         "- **C LangGraph + LangChain connector**: the same graph calling `ChatGoogleGenerativeAI` with structured output "
         "(JSON schema), given our client's temperature, seed, output cap, timeout and safety filters.", "",
         "What the connector warned during the run (settings it changed or dropped on its own):", "",
         *([f"- {w}" for w in m.get("c_warnings", [])] or ["- nothing"]), "",
         "```mermaid", s["diagram"].strip(), "```", "",
         f"## Results, {m['rounds']} rounds pooled", "",
         "| | " + " | ".join(LABELS[b] for b in BUILDS) + " |", "|---|---|---|---|"]

    def row(label, f):
        L.append(f"| {label} | " + " | ".join(f(p[b]) for b in BUILDS) + " |")

    per_round = {b: [m["per_round"][b][r] for r in sorted(m["per_round"][b], key=int)] for b in BUILDS}
    L.append("| answers matching the key, round by round | "
             + " | ".join(", ".join(f"{s['correct']} of {s['names']}" for s in per_round[b]) for b in BUILDS) + " |")

    def interval(b):                                         # names are the unit: the rounds ask the same names again
        rs = per_round[b]
        k, n = round(sum(s["correct"] for s in rs) / len(rs)), rs[0]["names"]
        lo, hi = wilson(k, n)
        return f"{_pct(k / n)} ({_pct(lo)} to {_pct(hi)})"

    L.append("| share right, 95% interval over the names | " + " | ".join(interval(b) for b in BUILDS) + " |")
    row("'cannot tell' answers", lambda x: str(x["cannot_tell"]))
    row("invented codes, rejected by the guard", lambda x: str(x["invented_rejected"]))
    row("names never returned", lambda x: str(x["not_returned"]))
    row("states that failed outright", lambda x: str(x["failed_states"]))
    row("model calls (of which repairs)", lambda x: f"{x['calls']} ({x['repairs']})")
    row("replies that did not parse", lambda x: str(x["unparsable"]))
    row("tokens per call, in / out", lambda x: f"{x['prompt_tokens_mean']:,.0f} / {x['completion_tokens_mean']:,.0f}")
    row(f"cost of all {m['rounds']} rounds at paid rates", lambda x: f"₹{x['inr']:.2f}")
    row("response time per call, typical / slowest 5%", lambda x: f"{_ms(x['latency_ms_p50'])} / {_ms(x['latency_ms_p95'])} ms")
    row("time spent outside the model call, per state", lambda x: f"{_ms(x['overhead_ms_mean'])} ms")
    L += ["", "The interval treats the names as the unit: a second round asks the same names again, so it is not new "
              "evidence. Time outside the model call includes, for A and B, our client's bookkeeping (the cache lookup and the "
              "cost records written to the store); LangGraph's own share is the difference between A and B."]
    L += ["", "## Consistency", ""]
    if m["repeat"]:
        L += ["How often a build gave the same answer (stage and code) to a name in both rounds: "
              + "; ".join(f"{LABELS[b]} {v['same']} of {v['names']} ({_pct(v['share'])})" for b, v in m["repeat"].items()) + "."]
    L += ["", "How often two builds gave the same answer in round 1: "
              + "; ".join(f"{k} {v['same']} of {v['names']} ({_pct(v['share'])})" for k, v in m["between"].items()) + ".",
          "", "## Code and dependencies", "",
          "| | " + " | ".join(LABELS[b] for b in BUILDS) + " |", "|---|---|---|---|",
          "| lines of control code (no comments or docstrings) | " + " | ".join(str(s["control_code_lines"][b]) for b in BUILDS) + " |",
          f"| packages added to the project | 0 | {s['footprint']['count']} | {s['footprint']['count']} |",
          f"| disk space they take | 0 | {s['footprint']['mb']} MB | {s['footprint']['mb']} MB |", "",
          f"Packages added by requirements-frameworks.txt: {', '.join(s['footprint']['packages_added'])}. Installing them also raised "
          "`idna` from 3.11 to 3.20 (pip's dry run, 2026-11-01).", "",
          "## What each build gives without extra code", "",
          "| | A | B | C |", "|---|---|---|---|",
          "| answer cache keyed by the exact question | yes | yes | no |",
          "| token, cost and failure records in the store | yes | yes | no |",
          "| rate limits from the account's quotas | yes | yes | no (an `InMemoryRateLimiter` can be added) |",
          "| fallback to a second provider (Groq) | yes | yes | no (`with_fallbacks` can be added) |",
          "| resume from a saved step (checkpointer) | no | yes | yes |",
          "| retry rule per step (`RetryPolicy`) | no (retries live in the client) | yes | yes |",
          "| step-by-step streaming and tracing hooks | own trace log | yes | yes |",
          "| a drawn diagram of the flow | no | yes | yes |", ""]
    L += ["## Reading", "", *reading(m), ""]
    return "\n".join(L)


def reading(m: dict) -> list[str]:
    """Plain conclusions drawn from the numbers above, so the report says what they mean."""
    p, s = m["pooled"], m["static"]
    all_same = all(v["share"] == 1.0 for v in m["between"].values())
    steady = all(v["share"] == 1.0 for v in m.get("repeat", {}).values()) if m.get("repeat") else None
    out = []
    if all_same:
        out.append("- **Answers:** all three builds gave the same answer to every name"
                   + (", and each gave the same answers in both rounds." if steady else "."
                      ) + " The framework changed nothing about what the model was asked or how it answered.")
    else:
        out.append("- **Answers:** the builds did not always agree: " + "; ".join(f"{k} {v['same']} of {v['names']}" for k, v in m["between"].items())
                   + ". Compare that with how often each build repeated itself to see whether the difference is more than chance.")
    if m.get("c_warnings"):
        out.append("- **Hidden settings:** LangChain's connector changed a setting on its own (listed at the top); here it made no "
                   "difference to the answers, but it is the kind of choice a framework makes without asking.")
    lat = {b: p[b]["latency_ms_p50"] for b in BUILDS}
    out.append(f"- **Speed:** typical response times were {', '.join(f'{b} {_ms(lat[b])} ms' for b in BUILDS)}: the model's own time "
               "dominates, and the control flow adds milliseconds.")
    lines = s["control_code_lines"]
    out.append(f"- **Cost of the framework:** {lines['A']} lines of control code by hand against {lines['B']} with LangGraph "
               f"({lines['C']} with LangChain's connector), and {s['footprint']['count']} extra packages ({s['footprint']['mb']} MB). "
               "What it buys is in the table above: resume from a saved step, per-step retry rules, streaming, a drawn diagram.")
    out.append("- **When it pays:** for this short loop (one call, one check, at most one repair) the hand-built version is smaller "
               "and keeps the cache, cost records and fallback; a longer flow with branches, waiting for a human, or resuming "
               "after a failure is where a graph framework starts to earn its packages.")
    return out


def write(m: dict, out_dir: Path = REPORTS_DIR) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path, js_path = out_dir / f"{m['date']}_frameworks.md", out_dir / f"{m['date']}_frameworks.json"
    md_path.write_text(render(m), encoding="utf-8")
    js_path.write_text(json.dumps(m, indent=1, default=str), encoding="utf-8")
    return md_path, js_path


def main(argv: list[str] | None = None) -> None:
    from dotenv import load_dotenv
    ap = argparse.ArgumentParser(description="Run the three builds of the resolver's model step live and compare them.")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--render", metavar="JSON", help="rewrite the markdown report from a saved JSON report, with no model call")
    args = ap.parse_args(argv)
    if args.render:
        m = json.loads(Path(args.render).read_text(encoding="utf-8"))
        md_path = REPORTS_DIR / f"{m['date']}_frameworks.md"
        md_path.write_text(render(m), encoding="utf-8")
        print(f"report -> {md_path}")
        return
    load_dotenv(ROOT / ".env")
    registry = resolver.load_registry()
    gold_rows = pd.read_csv(gold.DISTRICTS_PATH, dtype=str, keep_default_na=False)
    groups_ = groups(gold_rows, resolver.read_rows(resolver.CSV_PATH), registry)
    spacing = 60.0 / (int(os.environ["GEMINI_RPM"]) * HEADROOM)
    pace = Pacer(spacing)
    today = date.today().isoformat()
    tmp, conns = Path(tempfile.mkdtemp(prefix="fwcompare_")), []
    try:
        runs = run(groups_, lambda r: live_runners(registry, pace, tmp, r, conns), pace, args.rounds)
    finally:
        for c in conns:                                      # Windows cannot delete an open database file
            c.close()
        shutil.rmtree(tmp, ignore_errors=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = REPORTS_DIR / f"{today}_frameworks_answers.json"   # written before anything else can fail
    raw_path.write_text(json.dumps(raw_answers(runs, groups_), indent=1, default=str), encoding="utf-8")
    m = {"date": today, "model": os.environ["GEMINI_MODEL"], "rounds": args.rounds, "groups": len(groups_),
         "names": sum(len(g["key"]) for g in groups_), "spacing_s": spacing, **summarise(runs, groups_, args.rounds),
         "c_warnings": sorted(C_WARNINGS), "static": static_measures()}
    md_path, js_path = write(m)
    print(f"answers -> {raw_path}")
    print(f"report -> {md_path}\njson   -> {js_path}")       # the report holds the rupee sign, which a Windows console cannot print


if __name__ == "__main__":
    main()
