"""Ask the live model about gold names and score it against the key: does today's model still answer the way the
cached answers did? Manual, not part of the gate: it costs a few paise and needs the GEMINI_* settings. The names
are the model-reviewed rows of the district gold set, whole states at a time (most names first) until ``n`` is
reached, so a run is one call per state, three or four in all. Calls are tagged ``livecheck:<state>`` and cached
in a store of their own, so they never mix with the pipeline's calls or its metrics.

    python -m evals.live_check [--n 20] [--floor 0.8] [--db PATH]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd

from agents import resolver, store
from agents.llm_client import build_client
from evals import gold


def select(g: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Model-reviewed gold rows, whole states by descending count (ties by state code) until n rows; the last
    state is cut at n, its names in alphabetical order."""
    llm = g[g["proposed_stage"] == "llm"].copy()
    llm["state_code"] = llm["state_code"].astype(int)
    counts = llm.groupby("state_code").size().reset_index(name="n").sort_values(["n", "state_code"], ascending=[False, True])
    picked, total = [], 0
    for code in counts["state_code"]:
        rows = llm[llm["state_code"] == code].sort_values("unit")
        picked.append(rows.head(n - total))
        total += len(picked[-1])
        if total >= n:
            break
    return pd.concat(picked).reset_index(drop=True) if picked else llm.head(0)


def ask(client, rows: pd.DataFrame, unit_map: dict, registry: dict, pools: dict) -> list[dict]:
    """One resolver call per state with the gold names as the pending units; each answer against the gold code."""
    names = {s["state_code"]: s["name"] for s in registry["states"]}
    out = []
    for code, g in rows.groupby("state_code", sort=True):
        code, pool = int(code), pools[int(code)]
        pending = []
        for r in g.itertuples():
            um = unit_map[(r.era, r.state, r.unit)]
            pending.append(resolver.PendingUnit(r.unit, int(um["first_year"]), int(um["last_year"]), resolver.fuzzy_hints(r.unit, pool)))
        answers = resolver.resolve_llm(client, code, names[code], pending, pool, registry, tag="livecheck")
        for r in g.itertuples():
            res = answers[r.unit]
            gold_code = None if r.gold_code in ("", None) or pd.isna(r.gold_code) else int(r.gold_code)
            out.append({"state": r.state, "unit": r.unit, "gold_code": gold_code, "answer": res.district_code, "kind": res.kind,
                        "confidence": res.confidence, "match": res.district_code == gold_code})
    return out


def render(results: list[dict]) -> str:
    lines = [f"{'ok ' if r['match'] else 'NO '} {r['state']:<18} {r['unit']:<24} gold {str(r['gold_code']):<5} "
             f"model {str(r['answer']):<5} {r['kind']:<16} {r['confidence']:.2f}" for r in results]
    k = sum(r["match"] for r in results)
    lines.append(f"\nlive check: {k} of {len(results)} answers match the gold set ({k / len(results):.1%})" if results else "no names")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Ask the live model about gold names and score it against the key.")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--floor", type=float, default=0.8, help="exit non-zero below this share of matches")
    ap.add_argument("--db", help="store for the calls' cache (default: a temporary file, so nothing mixes with the pipeline's)")
    args = ap.parse_args(argv)
    g = pd.read_csv(gold.DISTRICTS_PATH, dtype=str, keep_default_na=False)
    rows = select(g, args.n)
    registry = resolver.load_registry()
    unit_map = resolver.read_rows(resolver.CSV_PATH)
    with tempfile.TemporaryDirectory() as tmp:
        conn = store.connect(args.db or Path(tmp) / "livecheck.db")
        try:
            results = ask(build_client(conn, only="gemini"), rows, unit_map, registry, resolver.build_pools(registry))
        finally:
            conn.close()
    print(render(results))
    share = sum(r["match"] for r in results) / len(results) if results else 0.0
    sys.exit(1 if share < args.floor else 0)


if __name__ == "__main__":
    main()
