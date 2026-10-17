"""Run every metric against the store and the gold sets; write a dated report and its JSON twin; append one
line to the history so the curve across reports can be drawn.

    python -m evals.run_evals [--out evals/reports]
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path

import pandas as pd

from agents import resolver, store
from evals import gold, metrics

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "evals" / "reports"
HISTORY_PATH = ROOT / "evals" / "history.csv"
# escalation_units_2001_2014: the share of the 2001-2014 names the pipeline of that date sends to a human, so a
# line written after the cuts went in counts them (and wrong_accepted_without_review says what they let through).
HISTORY_COLUMNS = ["date", "coverage_all", "extraction_field_exact_match", "resolution_precision_automatic",
                   "resolution_precision_llm", "hallucination_records",
                   "hallucination_resolver_rate", "escalation_units_2001_2014", "llm_accept_confidence",
                   "wrong_accepted_without_review", "escalation_records", "resolver_p95_ms",
                   "inr_per_1000_resolver_answers", "prompt_fingerprint"]


def compute(conn, districts_path: Path = gold.DISTRICTS_PATH, records_path: Path = gold.RECORDS_PATH) -> dict:
    g = pd.read_csv(districts_path, dtype={"proposed_confidence": str}).fillna({"proposed_confidence": ""})
    return {"date": date.today().isoformat(), "gold_districts": str(districts_path.relative_to(ROOT)),
            "coverage": metrics.coverage(conn), "notebook_coverage": metrics.notebook_coverage(),
            "extraction": metrics.extraction(conn, records_path) if records_path.exists() else None,
            "resolution": metrics.resolution(g), "calibration_llm": metrics.calibration(g, "llm"),
            "calibration_fuzzy": metrics.calibration(g, "fuzzy", bins=(0.0, 0.9, 0.95, 1.0001)),
            "acceptance": {s: metrics.acceptance(g, s, resolver.ACCEPT_WITHOUT_REVIEW.get(s)) for s in metrics.SCORED_STAGES},
            "hallucination": metrics.hallucination(conn), "escalation": metrics.escalation(conn, gold=g),
            "operations": metrics.operations(conn), "rebasing_sensitivity": metrics.rebasing_sensitivity(),
            "prompts": metrics.prompt_fingerprint()}


def _pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def _ci(ci) -> str:
    return "-" if not ci else f"{100 * ci[0]:.0f}–{100 * ci[1]:.0f}%"


def _extraction_lines(x) -> list[str]:
    if not x:
        return []
    who = ", ".join(f"{k} {v}" for k, v in x["checked_by"].items()) or "not recorded"
    lines = ["", "## Extraction: stored records against the verified gold records", "",
             "| era | gold records | counts compared | counts equal | records fully equal | missing |", "|---|---|---|---|---|---|"]
    for era, v in {**x["by_era"], "all": x["all"]}.items():
        lines.append(f"| {era} | {v['gold_rows']} | {v['counts_compared']} | {v['counts_equal']} | {_pct(v['record_exact_match'])} | {v['missing']} |")
    lines += ["", f"Field exact-match: {_pct(x['all']['field_exact_match'])}. Gold rows confirmed by: {who}. 'script' is an "
                  "independent re-read of the source files that shares no code with the pipeline; 'hand' is a person."]
    return lines


def _cut(x) -> str:
    return "none" if x is None else f"{x:.2f}"


def _acceptance_lines(acc) -> list[str]:
    if not acc:
        return []
    L = ["", "## Accepting answers without review: the cut", "",
         f"Rule (`evals.metrics.pick_cut`): the lowest score with at least {metrics.ACCEPT_MIN_ANSWERS} reviewed answers at or "
         f"above it, at least {_pct(metrics.ACCEPT_MIN_PRECISION)} of them right. An answer at or above its stage's cut in "
         "`resolver.ACCEPT_WITHOUT_REVIEW` is accepted without a human; below it, it waits for review.", "",
         "| stage | reviewed answers | right | rule's cut | cut in code | same calls | accepted by the cut in code | wrong among them |",
         "|---|---|---|---|---|---|---|---|"]
    for stage, a in acc.items():
        L.append(f"| {stage} | {a['answers']} | {a['right']} | {_cut(a['pick'])} | {_cut(a['in_code'])} | "
                 f"{'yes' if a['same_calls'] else '**no**'} | {a['accepted']} | {len(a['wrong_accepted'])} |")
    llm = acc.get("llm")
    if llm and llm["points"]:
        L += ["", "Model answers, cut by cut:", "",
              "| accept if confidence >= | accepted without review | wrong but accepted | still sent to a human | precision of accepted | 95% interval |",
              "|---|---|---|---|---|---|"]
        for p in llm["points"]:
            mark = " (in code)" if llm["in_code"] is not None and abs(p["cut"] - llm["in_code"]) < 1e-9 else ""
            L.append(f"| {p['cut']:.2f}{mark} | {p['accepted']} | {p['wrong']} | {p['to_human']} | {_pct(p['precision'])} | {_ci(p['ci95'])} |")
    wrong = [(stage, w) for stage, a in acc.items() for w in a["wrong_accepted"]]
    if wrong:
        L += ["", "Wrong answers the cuts in code let through:", ""]
        for stage, w in wrong:
            said = "no district" if w["proposed_code"] is None else f"code {w['proposed_code']}"
            L.append(f"- {stage}, {w['unit']} ({w['state']}), score {w['score']:.2f}: said {said} ({w['proposed_kind']}); "
                     f"the right answer is {w['gold_code'] if w['gold_code'] is not None else 'no district'} {w['gold_name']}".rstrip() + ".")
    L += ["", "Choosing a cut and grading it on the same answers flatters it, so two checks:", ""]
    for stage, a in acc.items():
        if a["pick"] is None and a["in_code"] is None:
            continue
        moved = "; ".join(f"without {mv['unit']} ({mv['score']:.2f}, {'right' if mv['right'] else 'wrong'}) the cut is {_cut(mv['cut'])}"
                          for mv in a["drop_one_moves"])
        L.append(f"- {stage}: the rule re-run without each answer in turn makes the same calls in {a['drop_one_same_calls']} of "
                 f"{a['drop_one_runs']} runs" + (f" ({moved})" if moved else "") + ". Each answer graded by the cut chosen without it: "
                 f"{a['held_out_right']} of {a['held_out_accepted']} accepted answers right ({_pct(a['held_out_precision'])}, "
                 f"{_ci(a['held_out_ci95'])}), the estimate for names the cut has not seen.")
    return L


def render(m: dict) -> str:
    L = [f"# Evaluation report ({m['date']})", "",
         f"Gold set: `{m['gold_districts']}`, {m['resolution']['gold_rows']} rows. Prompts measured: "
         + ", ".join(f"{k} `{v}`" for k, v in m["prompts"].items()) + ".", "",
         "## Coverage: source rows that land on a district", "", "| era | records | with district | unallocated bucket | coverage |", "|---|---|---|---|---|"]
    for era, c in m["coverage"].items():
        L.append(f"| {era} | {c['records']:,} | {c['with_district']:,} | {c['unallocated']:,} | {_pct(c['share'])} |")
    nb = m["notebook_coverage"]
    L += ["", f"## Before and after on the same {nb['units']} names (2001–2014)", "",
          "| | names placed on a district | rows placed on a district |", "|---|---|---|",
          f"| name equality against today's district names (the old approach) | {_pct(nb['share'])} | {_pct(nb['rows_share'])} |",
          f"| this pipeline | {_pct(nb['pipeline_share'])} | {_pct(nb['pipeline_rows_share'])} |",
          "", f"Same names and the same code for both lines. The pipeline's remaining {nb['no_district_units']} names are railway police "
              f"and crime cells, sent to no district on purpose. The build spec quotes {_pct(nb['spec_figure'])} for the original notebook, "
              "which matched against the old map; that figure is quoted, not reproduced.",
          *_extraction_lines(m.get("extraction")),
          "", "## Resolution precision on the gold set", "", "| stage | answered | correct | precision | 95% interval |", "|---|---|---|---|---|"]
    r = m["resolution"]
    for stage, v in r.items():
        if isinstance(v, dict):
            L.append(f"| {stage} | {v['answered']} | {v['correct']} | {_pct(v['precision'])} | {_ci(v['ci95'])} |")
    L += ["", f"Not scored: {r['unresolved']} rows the stages escalated without an answer, {r['human_only']} rows decided by hand with no automatic proposal.",
          "", "## Calibration: the model's confidence against its precision", "", "| confidence bin | answered | correct | precision | 95% interval | mean confidence |", "|---|---|---|---|---|---|"]
    for b in m["calibration_llm"]:
        L.append(f"| {b['bin']} | {b['answered']} | {b['correct']} | {_pct(b['precision'])} | {_ci(b['ci95'])} | "
                 f"{'-' if b['mean_confidence'] is None else f'{b['mean_confidence']:.2f}'} |")
    L += ["", "Fuzzy stage (WRatio / 100):", "", "| score bin | answered | correct | precision | 95% interval |", "|---|---|---|---|---|"]
    for b in m["calibration_fuzzy"]:
        L.append(f"| {b['bin']} | {b['answered']} | {b['correct']} | {_pct(b['precision'])} | {_ci(b['ci95'])} |")
    L += _acceptance_lines(m.get("acceptance"))
    h = m["hallucination"]
    L += ["", "## Hallucination", "",
          f"- Records with no source row: **{h['records_without_source_row']}** of {h['records']:,} (every record keys on its raw row).",
          f"- Resolver answers naming a code that is not a candidate for the state: **{h['resolver_codes_outside_candidates']}** of "
          f"{h['resolver_answers']} ({_pct(h['resolver_rate'])}); such an answer is escalated, never stored.",
          f"- Lineage proposals naming a code outside the state's list: **{h['lineage_codes_outside_state']}** of {h['lineage_proposals']} "
          f"({_pct(h['lineage_rate'])}); dropped and escalated.",
          "", "## Escalation: the human work that remained", "",
          "| era | units | needed review | rate | with the cuts in code | rate | wrong answers the cuts let through |",
          "|---|---|---|---|---|---|---|"]
    e = m["escalation"]
    for era, v in e.items():
        if era not in ("records", "open_escalations"):
            cut = (f" {v['with_cut']} | {_pct(v['rate_with_cut'])} | {v['wrong_accepted']} |" if "with_cut" in v else " - | - | - |")
            L.append(f"| {era} | {v['units']} | {v['needed_review']} | {_pct(v['rate'])} |" + cut)
    L += [f"| records flagged by the validator | {e['records']['records']:,} | {e['records']['flagged']} | {_pct(e['records']['rate'])} | - | - | - |",
          "", "Needed review: the names that went to a human. With the cuts in code: the same names had the cuts in "
              "`resolver.ACCEPT_WITHOUT_REVIEW` been in place, measured on the gold set, which holds exactly the names a human reviewed.",
          "", f"Open escalations: {e['open_escalations']}.", "",
          "## Operations: model calls, latency, tokens, cost", "",
          "| step | calls | p50 ms | p95 ms | prompt tokens | completion tokens | answers | INR total | INR per 1,000 answers |", "|---|---|---|---|---|---|---|---|---|"]
    o = m["operations"]
    for step, v in o.items():
        if step != "pricing":
            L.append(f"| {step} | {v['calls']} | {v['latency_ms_p50']:.0f} | {v['latency_ms_p95']:.0f} | {v['prompt_tokens_mean']:.0f} | "
                     f"{v['completion_tokens_mean']:.0f} | {v['answers']} | {v['inr_total']:.2f} | "
                     f"{'-' if v['inr_per_1000_answers'] is None else f'{v['inr_per_1000_answers']:.0f}'} |")
    p = o.get("pricing", {})
    L += ["", f"Cost is computed, not spent: {p.get('rates_usd_per_m')} USD per million tokens as of {p.get('as_of')}, at {p.get('usd_inr')} INR per USD; "
              "every run so far stayed inside the free tier.",
          "", "## Re-basing: stable units by grouping threshold", "", "| threshold | grouping edges | groups | largest group |", "|---|---|---|---|"]
    for t, v in m["rebasing_sensitivity"].items():
        L.append(f"| {t} | {v['grouping_edges']} | {v['groups']} | {v['largest']} |")
    return "\n".join(L) + "\n"


def history_row(m: dict) -> dict:
    r, h, e, o = m["resolution"], m["hallucination"], m["escalation"], m["operations"]
    x, acc = m.get("extraction"), m.get("acceptance")
    era_a = e.get("2001_2014", {})
    return {"date": m["date"], "coverage_all": round(m["coverage"]["all"]["share"], 4),
            "extraction_field_exact_match": round(x["all"]["field_exact_match"], 4) if x else "",
            "resolution_precision_automatic": round(r["all_automatic"]["precision"], 4),
            "resolution_precision_llm": round(r.get("llm", {}).get("precision", float("nan")), 4),
            "hallucination_records": h["records_without_source_row"], "hallucination_resolver_rate": round(h["resolver_rate"], 4),
            "escalation_units_2001_2014": round(era_a.get("rate_with_cut", era_a.get("rate", float("nan"))), 4),
            "llm_accept_confidence": "" if not acc or acc["llm"]["in_code"] is None else acc["llm"]["in_code"],
            "wrong_accepted_without_review": sum(len(a["wrong_accepted"]) for a in acc.values()) if acc else "",
            "escalation_records": round(e["records"]["rate"], 4),
            "resolver_p95_ms": round(o.get("resolve", {}).get("latency_ms_p95", float("nan"))),
            "inr_per_1000_resolver_answers": round(o.get("resolve", {}).get("inr_per_1000_answers") or float("nan"), 1),
            "prompt_fingerprint": "/".join(m["prompts"].values())}


def write(m: dict, out_dir: Path = REPORTS_DIR, history: Path = HISTORY_PATH) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md, js = out_dir / f"{m['date']}.md", out_dir / f"{m['date']}.json"
    md.write_text(render(m), encoding="utf-8")
    js.write_text(json.dumps(m, indent=1, default=str), encoding="utf-8")
    rows = []
    if history.exists():                                              # one line per report date; old columns kept, new ones blank
        with open(history, newline="", encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r.get("date") != m["date"]]
    rows.append(history_row(m))
    with open(history, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=HISTORY_COLUMNS, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return md, js


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Compute every metric and write a dated report.")
    ap.add_argument("--out", default=str(REPORTS_DIR))
    args = ap.parse_args(argv)
    m = compute(store.connect())
    md, js = write(m, Path(args.out))
    print(render(m))
    print(f"report -> {md}\njson   -> {js}\nhistory -> {HISTORY_PATH}")


if __name__ == "__main__":
    main()
