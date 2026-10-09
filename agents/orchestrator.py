"""One command builds the panel from the raw files and the verified decision records, and writes a trace.

    python -m agents.orchestrator build [--fixes verified|proposed|none] [--no-registry]

Order, each step rerunnable:
  1. wipe the derived tables (records, extended counts, unit map, lineage, crosswalk, panel) - raw rows,
     the LLM cache, the receipts and the escalation queue are kept;
  2. registry: states, districts (plus the map-only codes), circles from the 2016+ files;
  3. crosswalk, unit map and lineage: verified rows from the three CSVs;
  4. raw rows for the three eras; 5. records for the three eras; 6. validate; 7. panel.

The trace (``data/traces/<run_id>.jsonl``, also copied to ``latest.jsonl``) has one line per step with
what went in (file hashes), what came out (counts) and how long it took. It is the record of a run:
open it to answer "what did the pipeline do to the 2017 file?" without re-running anything.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import time
from pathlib import Path

from agents import lineage, loaders, panel, store, validator
from agents import resolver
from data_ingestion import crosswalk
from data_ingestion.crosswalk import ERA_FILES

ROOT = Path(__file__).resolve().parents[1]
TRACES_DIR = ROOT / "data" / "traces"
ERAS = store.ERAS                                            # time order, from the eras table's seed list
DERIVED = ("panel_observed", "extended_counts", "records", "unit_map", "district_lineage", "crosswalk")


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


class Trace:
    def __init__(self, run_id: str, path: Path):
        self.run_id, self.path, self.lines = run_id, path, []
        path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, step: str, result: dict, seconds: float, **extra) -> None:
        line = {"run_id": self.run_id, "step": step, "seconds": round(seconds, 3), "result": result, **extra}
        self.lines.append(line)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")

    def run(self, step: str, fn, **extra) -> dict:
        t0 = time.perf_counter()
        result = fn()
        self.record(step, result, time.perf_counter() - t0, **extra)
        return result


def wipe_derived(conn) -> dict:
    with conn:
        for table in DERIVED:
            conn.execute(f"DELETE FROM {table}")
    return {"cleared": list(DERIVED)}


def build(conn, *, fixes: str = "verified", registry: bool = True, raw_files: dict | None = None,
          csv_paths: dict | None = None, trace_dir: Path = TRACES_DIR, parquet_path: Path | None = None,
          registry_dict: dict | None = None) -> dict:
    """Build the observed panel. ``raw_files``/``csv_paths``/``registry_dict`` exist so tests can run the
    whole chain on fixture files; the CLI uses the repo's data."""
    raw_files = {**ERA_FILES, **(raw_files or {})}
    paths = {"crosswalk": crosswalk.CSV_PATH, "column_fixes": crosswalk.FIXES_PATH, "unit_map": resolver.CSV_PATH,
             "lineage": lineage.CSV_PATH, **(csv_paths or {})}
    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    trace = Trace(run_id, Path(trace_dir) / f"{run_id}.jsonl")
    inputs = {name: file_hash(Path(p)) for name, p in {**{f"raw:{e}": p for e, p in raw_files.items()},
                                                       **{f"csv:{k}": p for k, p in paths.items()}}.items() if Path(p).exists()}
    trace.record("inputs", {"fixes": fixes, "files": inputs}, 0.0)

    trace.run("wipe_derived", lambda: wipe_derived(conn))
    if registry:
        from data_ingestion.registry import build_registry, load_keys, load_lgd_list, write_registry
        trace.run("registry", lambda: write_registry(conn, build_registry(load_keys()), load_lgd_list()))
    trace.run("crosswalk_apply", lambda: crosswalk.apply(conn, csv_path=paths["crosswalk"]))
    trace.run("unit_map_apply", lambda: resolver.apply(conn, csv_path=paths["unit_map"]))
    trace.run("lineage_apply", lambda: lineage.apply(conn, csv_path=paths["lineage"]))
    for era in ERAS:
        trace.run(f"raw_rows:{era}", lambda e=era: loaders.load_raw_rows(conn, e, raw_files[e]))
    for era in ERAS:
        trace.run(f"records:{era}", lambda e=era: loaders.build_records(conn, e, fixes=fixes, fixes_path=paths["column_fixes"],
                                                                           registry=registry_dict))
    trace.run("validate", lambda: validator.validate(conn))
    result = trace.run("panel", lambda: panel.build_panel(conn, parquet_path))
    shutil.copyfile(trace.path, Path(trace_dir) / "latest.jsonl")
    return {"run_id": run_id, "trace": str(trace.path), "steps": len(trace.lines), "panel": result}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build the observed panel from the raw files and the verified decision records.")
    ap.add_argument("command", choices=["build"])
    ap.add_argument("--fixes", choices=["verified", "proposed", "none"], default="verified")
    ap.add_argument("--no-registry", action="store_true", help="keep the districts/circles already in the store")
    args = ap.parse_args(argv)
    conn = store.connect()
    out = build(conn, fixes=args.fixes, registry=not args.no_registry)
    for line in open(out["trace"], encoding="utf-8"):
        rec = json.loads(line)
        print(f"{rec['step']:<22} {rec['seconds']:>7.2f}s  {json.dumps(rec['result'], default=str)[:150]}")
    print(f"\ntrace -> {out['trace']}")


if __name__ == "__main__":
    main()
