"""The deterministic eval gate. CI runs it on every push: it rebuilds the store from the committed files alone (no
model call, about half a minute), computes the metrics that need no model, checks them against floors, and exits
non-zero when any check fails. It writes no report and no history line; those come from ``run_evals`` on a machine
that holds the model cache.

The checks, and what each one stops:
  coverage, extraction, records without a source row, resolution precision   a change that quietly worsens the
                                                                              pipeline's answers
  the cut rule agrees with the cut in code                                    lowering ACCEPT_WITHOUT_REVIEW without
                                                                              the evidence (evals.metrics.pick_cut)
  the latest report measured the prompt versions in code                      a new prompt version pushed before a
                                                                              report has measured it
  the committed panels equal the ones rebuilt from the committed inputs       data/panel_*.parquet drifting from the
                                                                              decision files that produce them

    python -m evals.gate [--keep DIR]        # --keep leaves the rebuilt store and panels in DIR for inspection
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from agents import orchestrator, prompts, store
from evals import run_evals

ROOT = Path(__file__).resolve().parents[1]
PANELS = {"panel_observed": ROOT / "data" / "panel_observed.parquet", "panel_rebased": ROOT / "data" / "panel_rebased.parquet"}
FLOORS = {"coverage_all": 0.99, "resolution_precision_automatic": 0.90}


def latest_report(reports_dir: Path = run_evals.REPORTS_DIR) -> tuple[str, dict] | None:
    files = sorted(reports_dir.glob("*.json"))
    if not files:
        return None
    return files[-1].stem, json.loads(files[-1].read_text(encoding="utf-8"))


def frames_equal(a: pd.DataFrame, b: pd.DataFrame, atol: float = 1e-6) -> tuple[bool, str]:
    """Same columns, same rows (sorted by the key columns), numbers equal within atol, everything else equal."""
    if list(a.columns) != list(b.columns):
        return False, "columns differ"
    if len(a) != len(b):
        return False, f"{len(a)} rows committed, {len(b)} rebuilt"
    keys = [c for c in ("unit_id", "state_code", "district_code", "year") if c in a.columns]
    a, b = a.sort_values(keys).reset_index(drop=True), b.sort_values(keys).reset_index(drop=True)
    for c in a.columns:
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
            same = np.allclose(a[c].astype(float).to_numpy(), b[c].astype(float).to_numpy(), atol=atol, equal_nan=True)
        else:
            same = a[c].fillna("").astype(str).equals(b[c].fillna("").astype(str))
        if not same:
            return False, f"column {c} differs"
    return True, f"{len(a)} rows equal"


def rebuild(workdir: Path):
    """The orchestrator's build into a store of its own, every output under workdir: nothing committed is touched."""
    conn = store.connect(workdir / "gate.db")
    orchestrator.build(conn, trace_dir=workdir / "traces", parquet_path=workdir / "panel_observed.parquet",
                       rebase_paths={"parquet_path": workdir / "panel_rebased.parquet", "units_path": workdir / "stable_units.json",
                                     "reports_dir": workdir / "reports"})
    return conn


def checks(m: dict, versions: dict, report: tuple[str, dict] | None, panels: dict[str, tuple[bool, str]]) -> list[dict]:
    out = []

    def check(name, ok, detail):
        out.append({"check": name, "ok": bool(ok), "detail": detail})

    cov = m["coverage"]["all"]["share"]
    check(f"coverage, all eras >= {FLOORS['coverage_all']:.0%}", cov >= FLOORS["coverage_all"], f"{cov:.2%}")
    x = m.get("extraction")
    check("extraction field exact-match = 100%", bool(x) and x["all"]["field_exact_match"] == 1.0,
          f"{x['all']['field_exact_match']:.2%}" if x else "no records gold set")
    orphans = m["hallucination"]["records_without_source_row"]
    check("records without a source row = 0", orphans == 0, str(orphans))
    p = m["resolution"]["all_automatic"]["precision"]
    check(f"resolution precision, automatic stages >= {FLOORS['resolution_precision_automatic']:.0%}",
          p >= FLOORS["resolution_precision_automatic"], f"{p:.1%}")
    for stage, a in m.get("acceptance", {}).items():
        fmt = lambda c: "none" if c is None else f"{c:.2f}"   # noqa: E731
        check(f"cut rule agrees with the cut in code: {stage}", a["same_calls"], f"rule {fmt(a['pick'])}, code {fmt(a['in_code'])}")
    if report is None:
        check("the latest report measured the prompt versions in code", False, "no report in evals/reports")
    else:
        name, r = report
        measured = r.get("prompt_versions")
        check("the latest report measured the prompt versions in code", measured == versions,
              f"report {name}: {measured if measured is not None else 'records no prompt versions'}; code: {versions}")
    for label, (ok, detail) in panels.items():
        check(f"committed {label} equals the one rebuilt from the committed inputs", ok, detail)
    return out


def render(results: list[dict]) -> str:
    width = max(len(r["check"]) for r in results)
    lines = [f"{'PASS' if r['ok'] else 'FAIL'}  {r['check']:<{width}}  {r['detail']}" for r in results]
    failed = sum(not r["ok"] for r in results)
    lines.append(f"\ngate: {len(results)} checks, {failed} failed")
    return "\n".join(lines)


def run(workdir: Path) -> list[dict]:
    committed = {k: pd.read_parquet(p) for k, p in PANELS.items() if p.exists()}
    conn = rebuild(workdir)
    try:
        m = run_evals.compute(conn)
    finally:
        conn.close()
    panels = {k: frames_equal(df, pd.read_parquet(workdir / f"{k}.parquet")) for k, df in committed.items()}
    return checks(m, dict(prompts.VERSIONS), latest_report(), panels)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Rebuild from the committed files and check the deterministic metrics against their floors.")
    ap.add_argument("--keep", help="directory to leave the rebuilt store and panels in (default: a temporary one)")
    args = ap.parse_args(argv)
    if args.keep:
        results = run(Path(args.keep))
    else:
        with tempfile.TemporaryDirectory() as tmp:
            results = run(Path(tmp))
    print(render(results))
    sys.exit(1 if any(not r["ok"] for r in results) else 0)


if __name__ == "__main__":
    main()
