"""Metrics, each a function of the store and the gold sets. Nothing here calls a model or changes the store.

coverage            share of source rows that land on a district, per era, next to the notebook's name-equality rule
extraction          the stored records against the verified gold records: field and record exact-match
resolution          precision of the automatic stages on the district gold set, by stage
calibration         the model's self-reported confidence against its precision, in bins, with Wilson intervals
acceptance          per scored stage: what accepting answers at or above a cut takes from the human and lets through
                    wrong, at every observed score; the rule's cut against the cut in the resolver; a drop-one check
hallucination       records with no source row (structural, expected 0) and model answers naming a code outside
                    the allowed list (resolver: not a candidate for the state; lineage: outside the state's codes)
escalation          how much human work remained: units needing review (and the same names under the resolver's
                    cuts, with the wrong answers those cuts let through), records flagged by the validator
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
from agents.schemas import CORE_CATEGORIES
from data_ingestion import crosswalk
from data_ingestion.normalize import norm
from evals import pricing

ROOT = Path(__file__).resolve().parents[1]
CONFIDENCE_BINS = (0.0, 0.8, 0.9, 0.95, 0.99, 1.0001)       # [0, .8) [.8, .9) [.9, .95) [.95, .99) [.99, 1]
THRESHOLDS = (0.1, 0.2, 0.3)
SCORED_STAGES = ("llm", "fuzzy", "geometry")                 # the stages whose answers carry a score a cut can use
ACCEPT_MIN_ANSWERS = 20          # the rule behind resolver.ACCEPT_WITHOUT_REVIEW: at least this many reviewed answers
ACCEPT_MIN_PRECISION = 0.95      # at or above the cut, and at least this share of them right
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
    """Before and after on the same 2001-2014 names, counted two ways (names, and source rows = a name's years).
    Before: the old approach, a unit is placed when its normalised name equals a current district's name or alias.
    After: this pipeline's verified unit map places it on a district. Both computed here, neither quoted."""
    registry = registry or resolver.load_registry()
    names = {norm(d["name"]) for d in registry["districts"]} | {a for d in registry["districts"] for a in d.get("aliases", [])}
    rows = [(u, r) for (era, _s, u), r in resolver.read_rows(unit_map_csv).items() if era == "2001_2014"]
    weight = [int(r.get("n_years") or 1) for _u, r in rows]
    same = [norm(u) in names for u, _r in rows]
    placed = [bool(str(r.get("district_code", "")).strip()) for _u, r in rows]
    n, w = len(rows), sum(weight)

    def share(flags, weights=None):
        if not n:
            return 0.0
        return sum(flags) / n if weights is None else sum(x for x, f in zip(weights, flags) if f) / w

    return {"units": n, "matched": sum(same), "share": share(same), "pipeline_units": sum(placed), "pipeline_share": share(placed),
            "no_district_units": n - sum(placed), "rows": w, "rows_share": share(same, weight),
            "pipeline_rows_share": share(placed, weight), "spec_figure": SPEC_NOTEBOOK_COVERAGE}


def extraction(conn, records_gold: Path) -> dict:
    """The stored records against the verified gold records. Each gold row is matched to the record built from the
    same source row (era, file, row number) and its seven counts are compared with the gold counts. Field exact-match =
    equal counts / counts compared; record exact-match = records with all seven equal / gold rows. A gold row whose
    record no longer exists counts as missing, its seven counts as unequal."""
    g = pd.read_csv(records_gold, dtype=str, keep_default_na=False)
    checked_by = g["checked_by"].value_counts().to_dict() if "checked_by" in g.columns else {}
    g = g[g["verified"] == "1"]
    cols = ", ".join(f"r.{c}" for c in CORE_CATEGORIES)
    current = {(era, f, int(sr)): vals for era, f, sr, *vals in conn.execute(
        f"SELECT w.era, w.source_file, w.source_row, {cols} FROM records r JOIN raw_rows w ON w.id = r.raw_row_id")}
    tally: dict[str, list[int]] = {}
    for row in g.to_dict("records"):
        t = tally.setdefault(row["era"], [0, 0, 0, 0])              # gold rows, equal counts, fully equal records, missing
        t[0] += 1
        rec = current.get((row["era"], row["source_file"], int(row["source_row"])))
        if rec is None:
            t[3] += 1
            continue
        equal = sum(int(row[c]) == int(v) for c, v in zip(CORE_CATEGORIES, rec))
        t[1] += equal
        t[2] += int(equal == len(CORE_CATEGORIES))
    n = len(CORE_CATEGORIES)

    def summary(rows, equal, full, missing):
        return {"gold_rows": rows, "counts_compared": rows * n, "counts_equal": equal, "missing": missing,
                "field_exact_match": equal / (rows * n) if rows else 0.0, "record_exact_match": full / rows if rows else 0.0}

    total = [sum(t[i] for t in tally.values()) for i in range(4)]
    return {"checked_by": checked_by, "by_era": {era: summary(*t) for era, t in sorted(tally.items())}, "all": summary(*total)}


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


# ---- accepting answers without review ---------------------------------------------------------------------------

def _scored(gold: pd.DataFrame, stage: str) -> pd.DataFrame:
    a = gold[gold["proposed_stage"] == stage].copy()
    a["score"] = pd.to_numeric(a["proposed_confidence"], errors="coerce")
    a["right"] = a["corrected"] == 0
    return a.dropna(subset=["score"])


def _takes(a: pd.DataFrame, stage: str, cut: float | None) -> pd.Series:
    """Which answers the resolver's own acceptance test takes at this cut (None: no cut, nothing taken)."""
    cuts = {} if cut is None else {stage: cut}
    return pd.Series([resolver.accepts_without_review(stage, k, s, cuts) for k, s in zip(a["proposed_kind"], a["score"])],
                     index=a.index, dtype=bool)


def pick_cut(a: pd.DataFrame, stage: str, min_answers: int = ACCEPT_MIN_ANSWERS,
             min_precision: float = ACCEPT_MIN_PRECISION) -> float | None:
    """The rule: the lowest observed score with at least min_answers answers at or above it, at least min_precision
    of them right. None when no score qualifies: too few answers, or too many of them wrong."""
    for cut in sorted(a["score"].unique()):
        took = a[_takes(a, stage, cut)]
        if len(took) >= min_answers and took["right"].mean() >= min_precision:
            return float(cut)
    return None


def _blank(x):
    return "" if pd.isna(x) else x


def acceptance(gold: pd.DataFrame, stage: str, in_code: float | None) -> dict:
    """What accepting a stage's answers without review would take and cost, on its reviewed answers. At every
    observed score: answers taken, the wrong ones among them, the rest still sent to a human. Then the rule's cut,
    whether the cut in code makes the same calls, and two checks on choosing a cut from the answers it is graded on:
    the rule re-run without each answer in turn (does a single answer move it?), and each answer graded by the cut
    chosen without it (an estimate for names the cut has not seen)."""
    a = _scored(gold, stage)
    points = []
    for cut in sorted(a["score"].unique(), reverse=True):
        took = a[_takes(a, stage, cut)]
        n, ok = len(took), int(took["right"].sum())
        points.append({"cut": float(cut), "accepted": n, "wrong": n - ok, "to_human": len(a) - n,
                       "precision": ok / n if n else None, "ci95": list(wilson(ok, n)) if n else None})
    pick = pick_cut(a, stage)
    moves, held_out = [], []
    for i in a.index:
        rest = a.drop(index=i)
        cut = pick_cut(rest, stage)
        if not (_takes(rest, stage, cut) == _takes(rest, stage, pick)).all():
            moves.append({"unit": a.at[i, "unit"], "score": float(a.at[i, "score"]), "right": bool(a.at[i, "right"]), "cut": cut})
        if _takes(a.loc[[i]], stage, cut).iloc[0]:
            held_out.append(bool(a.at[i, "right"]))
    took = a[_takes(a, stage, in_code)]
    wrong = [{"state": r["state"], "unit": r["unit"], "score": float(r["score"]), "proposed_kind": _blank(r["proposed_kind"]),
              "proposed_code": None if pd.isna(r["proposed_code"]) else int(r["proposed_code"]),
              "gold_code": None if pd.isna(r["gold_code"]) else int(r["gold_code"]), "gold_name": _blank(r["gold_name"])}
             for r in took[~took["right"]].to_dict("records")]
    k = sum(held_out)
    return {"answers": len(a), "right": int(a["right"].sum()), "points": points, "pick": pick, "in_code": in_code,
            "same_calls": bool((_takes(a, stage, pick) == _takes(a, stage, in_code)).all()),
            "accepted": len(took), "wrong_accepted": wrong,
            "drop_one_runs": len(a), "drop_one_same_calls": len(a) - len(moves), "drop_one_moves": moves,
            "held_out_accepted": len(held_out), "held_out_right": k,
            "held_out_precision": k / len(held_out) if held_out else None,
            "held_out_ci95": list(wilson(k, len(held_out))) if held_out else None}


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

def escalation(conn, unit_map_csv: Path = resolver.CSV_PATH, gold: pd.DataFrame | None = None) -> dict:
    """Per era: the names that went to a human (every stage but the identity lookups, less answers the cut accepted).
    With the gold set, which holds exactly the names a human reviewed: the same names under the cuts in
    resolver.ACCEPT_WITHOUT_REVIEW, how many the cuts take from the human and how many of those were wrong."""
    rows = resolver.read_rows(unit_map_csv)
    out = {}
    for era in sorted({k[0] for k in rows}):
        own = [r for (e, _s, _u), r in rows.items() if e == era and not r["proposed_by"].startswith("carried")]
        review = [r for r in own if r["stage"] not in AUTO_VERIFIED_STAGES and not resolver.accepted_without_review(r)]
        v = out[era] = {"units": len(own), "needed_review": len(review), "rate": len(review) / len(own) if own else 0.0}
        if gold is not None:
            g = gold[gold["era"] == era]
            took = [resolver.accepts_without_review(s, k, c) for s, k, c in
                    zip(g["proposed_stage"], g["proposed_kind"], pd.to_numeric(g["proposed_confidence"], errors="coerce"))]
            wrong = sum(t and c != 0 for t, c in zip(took, g["corrected"]))
            to_human = len(review) - sum(took)
            v.update({"accepted_by_cut": int(sum(took)), "wrong_accepted": int(wrong), "with_cut": to_human,
                      "rate_with_cut": to_human / len(own) if own else 0.0})
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
