"""Gold sets: frozen files of answers a human certified, kept apart from the pipeline so they cannot drift.

``evals/gold/districts.csv``  every unit-map row that went through human review (the automatic identity stages
    exact/circle/rule/successor and answers accepted without review by the cut are not in it): the verified code
    and kind, the stage that proposed, and the
    original automatic proposal where it differed, recovered from the LLM cache or the row's own reason.
    ``corrected = 1`` marks the rows where the automatic stage was wrong: those are the hard ones.
``evals/gold/records.csv``    100 source rows sampled across the four files with the record the pipeline built from
    each and the source cells that fed every count, for a checker to confirm (``verified``) or correct;
    ``checked_by`` says who confirmed it: ``script`` (an independent re-read of the source files that shares no code
    with the pipeline) and ``hand`` where a person confirmed the row.

Both files are generated once and then frozen: ``build`` refuses to overwrite unless ``--force`` is given.

    python -m evals.gold build [--force]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import pandas as pd

from agents import loaders, resolver, store
from agents.resolver import AUTO_VERIFIED_STAGES
from agents.schemas import CORE_CATEGORIES
from data_ingestion import crosswalk
from data_ingestion.normalize import norm

ROOT = Path(__file__).resolve().parents[1]
GOLD_DIR = ROOT / "evals" / "gold"
DISTRICTS_PATH = GOLD_DIR / "districts.csv"
RECORDS_PATH = GOLD_DIR / "records.csv"
DISTRICT_COLUMNS = ["era", "state", "state_code", "unit", "gold_code", "gold_name", "gold_kind", "proposed_stage",
                    "proposed_code", "proposed_kind", "proposed_confidence", "corrected", "reviewer_note"]
RECORD_COLUMNS = ["era", "source_file", "source_row", "year", "state", "unit", "circle", "district_code", *CORE_CATEGORIES,
                  "breakdown", "verified", "checked_by", "reviewer_note"]
SAMPLE = {"2001_2014": 40, "2015": 15, "2016": 10, "2017_plus": 35}
SEED = 7
_FUZZY_RE = re.compile(r"'(.+?)' ~ (.+?)(?:;|$)")


def _stage_of(proposed_by: str) -> str:
    if proposed_by.startswith("llm"):
        return "llm"
    if proposed_by.startswith("carried"):
        return "carried"
    return proposed_by or "human"


def cached_llm_answers(conn) -> dict[tuple[int, str], dict]:
    """(state_code, normalised unit) -> the model's original item, from the resolver's saved answers."""
    out: dict[tuple[int, str], dict] = {}
    for tag, response, created in conn.execute("SELECT tag, response_json, created_at FROM llm_calls WHERE tag LIKE 'resolve:%' "
                                               "ORDER BY created_at"):
        state_code = int(tag.split(":")[1])
        for item in json.loads(response).get("items", []):
            out[(state_code, norm(item["unit"]))] = item                     # a later call (a repair) overrides
    return out


def build_districts(conn, unit_map_csv: Path = resolver.CSV_PATH) -> pd.DataFrame:
    rows = resolver.read_rows(unit_map_csv)
    registry = resolver.load_registry()
    names = {d["district_code"]: d["name"] for d in registry["districts"]}
    for code, name in conn.execute("SELECT district_code, name FROM districts"):     # map-only codes such as 526
        names.setdefault(int(code), name)
    by_name = {norm(d["name"]): d["district_code"] for d in registry["districts"]}
    cache = cached_llm_answers(conn)
    out = []
    for (era, state, unit), r in rows.items():
        if (r["verified"] != "1" or r["stage"] in AUTO_VERIFIED_STAGES or r["proposed_by"].startswith("carried")
                or resolver.accepted_without_review(r)):            # the answer key holds human decisions only
            continue
        stage = _stage_of(r["proposed_by"])
        gold = int(r["district_code"]) if r["district_code"].strip() else None
        p_code, p_kind, p_conf = gold, r["kind"], r["confidence"]           # accepted as proposed unless shown otherwise
        if stage == "llm":
            item = cache.get((int(r["state_code"]), norm(unit)))
            if item is not None:
                p_code, p_kind, p_conf = item.get("district_code"), item.get("kind", ""), item.get("confidence", "")
        elif stage in ("fuzzy", "geometry") and r["stage"] == "human":
            m = _FUZZY_RE.search(r["reason"])
            p_code = by_name.get(norm(m.group(2))) if m else None
        elif stage == "human":
            p_code, p_kind, p_conf = None, "", ""                            # nothing automatic was proposed
        if r["stage"] == "unresolved":
            p_code, p_kind = None, "unknown"
        corrected = int(stage != "human" and r["stage"] != "unresolved" and p_code != gold)
        out.append({"era": era, "state": state, "state_code": int(r["state_code"]), "unit": unit, "gold_code": gold,
                    "gold_name": names.get(gold, "") if gold is not None else "", "gold_kind": r["kind"],
                    "proposed_stage": stage if r["stage"] != "unresolved" else "unresolved", "proposed_code": p_code,
                    "proposed_kind": p_kind, "proposed_confidence": p_conf, "corrected": corrected,
                    "reviewer_note": r["reviewer_note"]})
    df = pd.DataFrame(out, columns=DISTRICT_COLUMNS)
    for c in ("gold_code", "proposed_code"):
        df[c] = df[c].astype("Int64")
    return df.sort_values(["era", "state", "unit"]).reset_index(drop=True)


def _breakdown(row, mapping: dict[str, str]) -> str:
    parts: dict[str, list[str]] = {c: [] for c in CORE_CATEGORIES}
    for col, cat in mapping.items():
        core = crosswalk.core_of(cat)
        if core and not pd.isna(row[col]):
            parts[core].append(f"{col}={int(row[col])}")
    return " | ".join(f"{c}: {' + '.join(p) if p else 'none'}" for c, p in parts.items())


def build_records(conn, sample: dict[str, int] = SAMPLE, seed: int = SEED) -> pd.DataFrame:
    """A stratified sample of records with the source cells that produced every count."""
    out = []
    for era, n in sample.items():
        recs = pd.read_sql_query("SELECT * FROM records WHERE era = ? ORDER BY id", conn, params=(era,))
        if recs.empty:
            continue
        picked = recs.sample(n=min(n, len(recs)), random_state=seed)
        frame = loaders.raw_frame(conn, era)
        mapping = crosswalk.verified_mapping(conn, era)
        for col in mapping:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if era != "2001_2014":                                            # the loaders apply the verified column fixes here too
            frame["year"] = pd.to_numeric(frame["YEAR" if "YEAR" in frame.columns else "year"]).astype(int)
            frame = crosswalk.apply_column_fixes(frame, crosswalk.read_fixes(only_verified=True), era)
        frame = frame.set_index("raw_row_id")
        for rec in picked.itertuples():
            raw = frame.loc[rec.raw_row_id]
            out.append({"era": era, "source_file": rec.source_file if hasattr(rec, "source_file") else "", "source_row": "",
                        "year": rec.year, "state": rec.state, "unit": rec.district, "circle": rec.registration_circle or "",
                        "district_code": rec.district_code, **{c: getattr(rec, c) for c in CORE_CATEGORIES},
                        "breakdown": _breakdown(raw, mapping), "verified": "", "checked_by": "", "reviewer_note": ""})
            out[-1]["source_file"], out[-1]["source_row"] = raw["source_file"], int(raw["source_row"])
    df = pd.DataFrame(out, columns=RECORD_COLUMNS)
    df["district_code"] = df["district_code"].astype("Int64")
    return df.sort_values(["era", "source_file", "source_row"]).reset_index(drop=True)


def write(df: pd.DataFrame, path: Path, force: bool = False) -> bool:
    if path.exists() and not force:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    return True


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Generate the gold sets once; they are frozen afterwards.")
    ap.add_argument("command", choices=["build"])
    ap.add_argument("--force", action="store_true", help="overwrite a frozen gold file")
    args = ap.parse_args(argv)
    conn = store.connect()
    d, r = build_districts(conn), build_records(conn)
    for df, path in ((d, DISTRICTS_PATH), (r, RECORDS_PATH)):
        written = write(df, path, args.force)
        print(f"{path.relative_to(ROOT)}: {len(df)} rows {'written' if written else 'kept (frozen; use --force to regenerate)'}")
    print(f"districts: {int(d['corrected'].sum())} corrected by hand, by stage {d['proposed_stage'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
