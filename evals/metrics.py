"""Metrics, each a function of the store and the gold sets. Nothing here calls a model or changes the store.

coverage            share of source rows that land on a district, per era, next to the notebook's name-equality rule
resolution          precision of the automatic stages on the district gold set, by stage
calibration         the model's self-reported confidence against its precision, in bins, with Wilson intervals
hallucination       records with no source row (structural, expected 0) and model answers naming a code outside
                    the allowed list (resolver: not a candidate for the state; lineage: outside the state's codes)
escalation          how much human work remained: units needing review, records flagged by the validator
operations          per model step: calls, p50/p95 latency, tokens, cost at published rates
rebasing_sensitivity  stable-unit groups at thresholds 0.1 / 0.2 / 0.3
prompt_fingerprint  sha256 over the prompt text and builders, so a report says which prompts it measured
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from agents import lineage, prompts, resolver
from agents.resolver import AUTO_VERIFIED_STAGES
from data_ingestion import crosswalk
from data_ingestion.normalize import norm
from evals import pricing

ROOT = Path(__file__).resolve().parents[1]
CONFIDENCE_BINS = (0.0, 0.8, 0.9, 0.95, 0.99, 1.0001)       # [0, .8) [.8, .9) [.9, .95) [.95, .99) [.99, 1]
THRESHOLDS = (0.1, 0.2, 0.3)
SPEC_NOTEBOOK_COVERAGE = 0.727                              # the figure quoted in the build spec for the original notebook


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a proportion with few samples (the plain k/n ± 1.96·sd misbehaves near 0 and 1)."""
    if n == 0:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


# ---- coverage ---------------------------------------------------------------------------------------------------

def coverage(conn) -> dict:
    """Records with a district code over all records, per era and overall; a NULL code is a state's unallocated
    bucket (railways, crime cells), which is a decision, not a failure, so it is shown separately."""
    df = pd.read_sql_query("SELECT era, district_code IS NOT NULL AS has_code, COUNT(*) AS n FROM records "
                           "GROUP BY era, has_code", conn)
    out = {}
    for era, g in df.groupby("era"):
        total, with_code = int(g["n"].sum()), int(g[g["has_code"] == 1]["n"].sum())
        out[era] = {"records": total, "with_district": with_code, "unallocated": total - with_code, "share": with_code / total}
    total, with_code = int(df["n"].sum()), int(df[df["has_code"] == 1]["n"].sum())
    out["all"] = {"records": total, "with_district": with_code, "unallocated": total - with_code, "share": with_code / total}
    return out


def notebook_coverage(unit_map_csv: Path = resolver.CSV_PATH, registry: dict | None = None) -> dict:
    """The rule the original notebook used, re-run on the same names: a 2001-2014 unit counts as matched when its
    normalised name equals a current district's name or alias. Computed, not quoted."""
    registry = registry or resolver.load_registry()
    names = {norm(d["name"]) for d in registry["districts"]} | {a for d in registry["districts"] for a in d.get("aliases", [])}
    units = [u for (era, _s, u) in resolver.read_rows(unit_map_csv) if era == "2001_2014"]
    matched = sum(norm(u) in names for u in units)
    return {"units": len(units), "matched": matched, "share": matched / len(units) if units else 0.0,
            "spec_figure": SPEC_NOTEBOOK_COVERAGE}


# ---- resolution precision ------------------------------------------------------------------------------------

def resolution(gold: pd.DataFrame) -> dict:
    """Per automatic stage: answered = rows where the stage proposed a code or 'no district'; correct = the proposal
    matched the verified decision; precision = correct / answered. 'unresolved' rows (escalated with no answer) and
    'human' rows (nothing automatic) are counted, not scored."""
    out = {}
    scored = gold[~gold["proposed_stage"].isin(("human", "unresolved"))]
    for stage, g in scored.groupby("proposed_stage"):
        n, correct = len(g), int((g["corrected"] == 0).sum())
        lo, hi = wilson(correct, n)
        out[stage] = {"answered": n, "correct": correct, "precision": correct / n, "ci95": [lo, hi]}
    n, correct = len(scored), int((scored["corrected"] == 0).sum())
    lo, hi = wilson(correct, n)
    out["all_automatic"] = {"answered": n, "correct": correct, "precision": correct / n if n else 0.0, "ci95": [lo, hi]}
    out["unresolved"] = int((gold["proposed_stage"] == "unresolved").sum())
    out["human_only"] = int((gold["proposed_stage"] == "human").sum())
    out["gold_rows"] = int(len(gold))
    return out


def calibration(gold: pd.DataFrame, stage: str = "llm", bins: tuple = CONFIDENCE_BINS) -> list[dict]:
    """Rows of the reliability curve: confidence bin -> answered, correct, precision, interval."""
    g = gold[gold["proposed_stage"] == stage].copy()
    g["conf"] = pd.to_numeric(g["proposed_confidence"], errors="coerce")
    g = g.dropna(subset=["conf"])
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        b = g[(g["conf"] >= lo) & (g["conf"] < hi)]
        n, correct = len(b), int((b["corrected"] == 0).sum())
        rows.append({"bin": f"[{lo:.2f}, {min(hi, 1.0):.2f}{']' if hi > 1 else ')'}", "answered": n, "correct": correct,
                     "precision": correct / n if n else None, "ci95": list(wilson(correct, n)) if n else None,
                     "mean_confidence": float(b["conf"].mean()) if n else None})
    return rows


# ---- hallucination -------------------------------------------------------------------------------------------

def hallucination(conn) -> dict:
    orphan = conn.execute("SELECT COUNT(*) FROM records r LEFT JOIN raw_rows w ON w.id = r.raw_row_id WHERE w.id IS NULL").fetchone()[0]
    records = conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    pools = resolver.build_pools(resolver.load_registry())
    asked = outside = 0
    for tag, response in conn.execute("SELECT tag, response_json FROM llm_calls WHERE tag LIKE 'resolve:%'"):
        pool = pools.get(int(tag.split(":")[1]))
        for item in json.loads(response).get("items", []):
            asked += 1
            code = item.get("district_code")
            if code is not None and (pool is None or code not in pool.codes):
                outside += 1
    lineage_rows = conn.execute("SELECT COUNT(*) FROM llm_calls WHERE tag LIKE 'lineage:%'").fetchone()[0]
    proposals = sum(len(json.loads(r[0]).get("items", [])) for r in conn.execute("SELECT response_json FROM llm_calls WHERE tag LIKE 'lineage:%'"))
    bad_lineage = sum(1 for (v,) in conn.execute("SELECT violations_json FROM escalations WHERE kind = 'unit'")
                      if any(x.get("rule") == "lineage_code_not_in_state" for x in json.loads(v)))
    return {"records": records, "records_without_source_row": int(orphan),
            "resolver_answers": asked, "resolver_codes_outside_candidates": outside,
            "resolver_rate": outside / asked if asked else 0.0,
            "lineage_calls": lineage_rows, "lineage_proposals": proposals, "lineage_codes_outside_state": bad_lineage,
            "lineage_rate": bad_lineage / proposals if proposals else 0.0}


# ---- escalation ---------------------------------------------------------------------------------------------------

def escalation(conn, unit_map_csv: Path = resolver.CSV_PATH) -> dict:
    rows = resolver.read_rows(unit_map_csv)
    out = {}
    for era in sorted({k[0] for k in rows}):
        own = [r for (e, _s, _u), r in rows.items() if e == era and not r["proposed_by"].startswith("carried")]
        review = [r for r in own if r["stage"] not in AUTO_VERIFIED_STAGES]
        out[era] = {"units": len(own), "needed_review": len(review), "rate": len(review) / len(own) if own else 0.0}
    flagged = conn.execute("SELECT COUNT(*) FROM records WHERE validation_status = 'escalated'").fetchone()[0]
    records = conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    out["records"] = {"records": records, "flagged": flagged, "rate": flagged / records if records else 0.0}
    out["open_escalations"] = {k: n for k, n in conn.execute("SELECT kind, COUNT(*) FROM escalations WHERE status = 'open' GROUP BY kind")}
    return out


# ---- operations ---------------------------------------------------------------------------------------------------

def operations(conn) -> dict:
    df = pd.read_sql_query("SELECT tag, model, prompt_tokens, completion_tokens, latency_ms FROM llm_attempts WHERE outcome = 'ok'", conn)
    if df.empty:
        return {}
    df["step"] = df["tag"].str.split(":").str[0]
    answers = {}
    for tag, response in conn.execute("SELECT tag, response_json FROM llm_calls"):
        answers[tag.split(":")[0]] = answers.get(tag.split(":")[0], 0) + len(json.loads(response).get("items", []))
    out = {}
    for step, g in df.groupby("step"):
        usd = sum(pricing.cost_usd(m, p, c) or 0.0 for m, p, c in zip(g["model"], g["prompt_tokens"].fillna(0), g["completion_tokens"].fillna(0)))
        items = answers.get(step, 0)
        out[step] = {"calls": int(len(g)), "models": sorted(g["model"].unique()),
                     "latency_ms_p50": float(np.percentile(g["latency_ms"], 50)), "latency_ms_p95": float(np.percentile(g["latency_ms"], 95)),
                     "prompt_tokens_mean": float(g["prompt_tokens"].mean()), "completion_tokens_mean": float(g["completion_tokens"].mean()),
                     "answers": items, "usd_total": usd, "inr_total": usd * pricing.USD_INR,
                     "inr_per_1000_answers": usd * pricing.USD_INR / items * 1000 if items else None}
    out["pricing"] = {"rates_usd_per_m": pricing.RATES_USD_PER_M, "as_of": pricing.RATES_AS_OF, "usd_inr": pricing.USD_INR}
    return out


# ---- re-basing sensitivity ------------------------------------------------------------------------------------------

def rebasing_sensitivity(units_path: Path = lineage.UNITS_PATH, thresholds: tuple = THRESHOLDS) -> dict:
    su = json.loads(Path(units_path).read_text(encoding="utf-8"))
    edges, members = su["meta"]["edges"], [m for u in su["units"] for m in u["members"]]
    nodes = [{"district_code": m, "name": str(m)} for m in members]
    out = {}
    for t in thresholds:
        rows = [{"verified": "1", "kind": e["kind"], "parent_code": str(e["parent_code"]), "child_code": str(e["child_code"]), "year": str(e["year"])}
                for e in edges if (e["share"] >= t if e["share"] is not None else e["grouped"])]
        units = lineage.build_units(rows, nodes, {})
        out[str(t)] = {"grouping_edges": len(rows), "groups": len(units), "largest": max(len(u["members"]) for u in units)}
    return out


# ---- prompt fingerprint -------------------------------------------------------------------------------------------

def prompt_fingerprint() -> dict:
    parts = {"project_context": prompts.PROJECT_CONTEXT, "resolver": inspect.getsource(resolver.build_prompt),
             "lineage": inspect.getsource(lineage.build_prompt), "crosswalk": inspect.getsource(crosswalk.build_prompt)}
    return {k: hashlib.sha256(v.encode("utf-8")).hexdigest()[:12] for k, v in parts.items()}
