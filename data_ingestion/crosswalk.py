"""Category crosswalk: which source column feeds which of the seven core categories.

Era A (2001-2014) has exactly the seven columns, so it *defines* the core
categories and is mapped without any model. Eras B (2016) and C (2017+) are
proposed by the model, one call per era, and land in ``data/crosswalk.csv``
with ``verified = 0``. A human edits that one file: fix ``core_category`` if
it is wrong, set ``verified = 1``. ``apply`` then loads only verified rows into
the database. Loaders refuse anything unverified, so the pipeline cannot run
on an unchecked mapping.

    python -m data_ingestion.crosswalk propose   # model proposals -> data/crosswalk.csv (verified rows untouched)
    python -m data_ingestion.crosswalk apply     # data/crosswalk.csv (verified rows) -> crosswalk table
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field

from agents import store
from agents.llm_client import build_client
from agents.prompts import PROJECT_CONTEXT
from agents.schemas import CORE_CATEGORIES
from data_ingestion.profile import KEY_COLS_BC

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CSV_PATH = ROOT / "data" / "crosswalk.csv"
ERA_FILES = {"2016": RAW / "districtwise-crime-against-women-2016.csv",
             "2017_plus": RAW / "districtwise-crime-against-women-2017-onwards.csv"}
ERA_LABEL = {"2016": "2016", "2017_plus": "2017-2022"}
ERA_ORDER = {"2001_2014": 0, "2016": 1, "2017_plus": 2}
CSV_COLUMNS = ["era", "source_column", "core_category", "reason", "ambiguous", "proposed_by", "verified", "reviewer_note"]

# The seven core categories ARE the 2001-2014 columns. The law sections are the
# standard NCRB heads and are what the model is told to map by.
ERA_A_DEFINITIONS: dict[str, tuple[str, str]] = {
    "Rape": ("rape", "IPC 376"),
    "Kidnapping and Abduction": ("kidnapping_abduction", "IPC 363-373, of women and girls"),
    "Dowry Deaths": ("dowry_deaths", "IPC 304B"),
    "Assault on women with intent to outrage her modesty": ("assault_on_women", "IPC 354"),
    "Insult to modesty of Women": ("insult_to_modesty", "IPC 509"),
    "Cruelty by Husband or his Relatives": ("cruelty_by_husband", "IPC 498A"),
    "Importation of Girls": ("importation_of_girls", "IPC 366B, from a foreign country"),
}

CoreCategory = Literal[CORE_CATEGORIES]


class ColumnMapping(BaseModel):
    source_column: str
    core_category: Optional[CoreCategory] = Field(
        description="one of the seven core categories, or null when the offence is not one of them")
    reason: str = Field(min_length=3, description="one line, citing the law section")
    ambiguous: bool = Field(default=False, description="true when reasonable statisticians could disagree")


class CrosswalkProposal(BaseModel):
    mappings: list[ColumnMapping]


def era_columns(era: str) -> list[str]:
    """The category columns of one era's file, in file order (header only, no rows read)."""
    return [c for c in pd.read_csv(ERA_FILES[era], nrows=0).columns if c not in KEY_COLS_BC]


def build_prompt(era: str, columns: list[str], missing=(), unknown=()) -> str:
    core = "\n".join(f'  {cat:<21} = "{col}" ({law})' for col, (cat, law) in ERA_A_DEFINITIONS.items())
    lines = [
        "Task: build a category crosswalk for one era of NCRB district-level 'crimes against women' data.",
        "",
        "The seven CORE categories are defined by the 2001-2014 NCRB table columns:",
        core,
        "",
        f"Below are the {len(columns)} category columns of the {ERA_LABEL[era]} file. For EVERY column decide which "
        "single core category it feeds, or null when the offence is not one of the seven (for example acid attack, "
        "Dowry Prohibition Act, Immoral Traffic (Prevention) Act, Protection of Women from Domestic Violence Act, "
        "POCSO Act, cyber crimes).",
        "",
        "Rules:",
        "1. Every column appears exactly once in your answer, spelled exactly as given.",
        "2. A column maps to at most one core category. Several columns may map to the same category (their counts "
        "will be added together). Never split one column across categories.",
        "3. Map by legal meaning (the IPC section), not by similar words: a column belongs to a core category only if "
        "the 2001-2014 column would have counted the same cases.",
        "4. Set ambiguous = true when reasonable statisticians could disagree (attempts; POCSO child-rape cases that "
        "before 2013 were counted under IPC 376; kidnapping sub-heads), and say why in the reason.",
        "5. reason: one line, cite the law section.",
        "",
        "Columns:",
        *[f"- {c}" for c in columns],
    ]
    if missing or unknown:
        lines += ["", "Your previous answer was incomplete."]
        if missing:
            lines.append(f"Missing columns that MUST be included: {sorted(missing)}")
        if unknown:
            lines.append(f"Unknown columns that must NOT appear: {sorted(unknown)}")
        lines.append(f"Return all {len(columns)} columns exactly once.")
    return "\n".join(lines)


def propose_era(client, era: str, columns: list[str]) -> tuple[list[ColumnMapping], dict]:
    """One model call for one era, plus one repair call if the column set does not match.
    That set check is something the JSON schema cannot express, so it lives here."""
    want = set(columns)
    result = client.complete(build_prompt(era, columns), CrosswalkProposal, tag=f"crosswalk:{era}",
                             system=PROJECT_CONTEXT)
    got = {m.source_column for m in result.parsed.mappings}
    if got != want:
        result = client.complete(build_prompt(era, columns, missing=want - got, unknown=got - want),
                                 CrosswalkProposal, tag=f"crosswalk:{era}:repair", system=PROJECT_CONTEXT)
    by_col = {m.source_column: m for m in result.parsed.mappings if m.source_column in want}
    info = {"provider": result.provider, "model": result.model, "cached": result.cached, "tries": result.tries,
            "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
            "missing": sorted(want - set(by_col))}
    return [by_col[c] for c in columns if c in by_col], info


# ---- the one file that holds the human's decisions --------------------------------------

def read_rows(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        return {(r["era"], r["source_column"]): r for r in csv.DictReader(fh)}


def write_rows(path: Path, rows: dict[tuple[str, str], dict], order: dict[tuple[str, str], int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for key in sorted(rows, key=lambda k: (ERA_ORDER[k[0]], order.get(k, 10**6), k[1])):
            writer.writerow({c: rows[key].get(c, "") for c in CSV_COLUMNS})


def propose(client, conn, *, csv_path: Path = CSV_PATH, columns_by_era: dict[str, list[str]] | None = None) -> dict:
    """Write Era A by definition; ask the model about every unverified column of Eras B and C.
    Rows a human has already verified are never touched."""
    columns_by_era = columns_by_era or {era: era_columns(era) for era in ERA_FILES}
    rows, order, summary = read_rows(csv_path), {}, {}

    for i, (col, (cat, law)) in enumerate(ERA_A_DEFINITIONS.items()):
        key = ("2001_2014", col)
        order[key] = i
        rows[key] = {"era": "2001_2014", "source_column": col, "core_category": cat,
                     "reason": f"defines the core category ({law})", "ambiguous": "0", "proposed_by": "definition",
                     "verified": "1", "reviewer_note": rows.get(key, {}).get("reviewer_note", "")}

    for era, columns in columns_by_era.items():
        for i, c in enumerate(columns):
            order[(era, c)] = i
        todo = [c for c in columns if rows.get((era, c), {}).get("verified") != "1"]
        if not todo:
            summary[era] = {"proposed": 0, "kept_verified": len(columns)}
            continue
        mappings, info = propose_era(client, era, columns)          # ask about all columns; overwrite only unverified
        by_col = {m.source_column: m for m in mappings}
        for c in todo:
            m = by_col.get(c)
            rows[(era, c)] = {"era": era, "source_column": c,
                              "core_category": (m.core_category or "") if m else "",
                              "reason": m.reason if m else "NOT PROPOSED: see escalation queue",
                              "ambiguous": "1" if (m and m.ambiguous) else "0",
                              "proposed_by": f"llm:{info['model']}" if m else "none",
                              "verified": "0", "reviewer_note": rows.get((era, c), {}).get("reviewer_note", "")}
        if info["missing"]:
            store.add_escalation(conn, kind="crosswalk", ref={"era": era, "columns": info["missing"]},
                                 violations=[{"rule": "columns_not_proposed",
                                              "message": f"the model left out {len(info['missing'])} column(s) even "
                                                         "after one repair try"}])
        summary[era] = {"proposed": len(todo), "kept_verified": len(columns) - len(todo), **info}

    write_rows(csv_path, rows, order)
    return summary


def apply(conn, *, csv_path: Path = CSV_PATH) -> dict:
    """Load verified rows into the crosswalk table. Unverified rows stay out."""
    loaded, pending = {}, {}
    for (era, col), r in read_rows(csv_path).items():
        if r.get("verified") != "1":
            pending[era] = pending.get(era, 0) + 1
            continue
        cat = r["core_category"].strip() or None
        if cat is not None and cat not in CORE_CATEGORIES:
            raise ValueError(f"{era}/{col}: '{cat}' is not a core category; allowed: {CORE_CATEGORIES}")
        conn.execute("INSERT OR REPLACE INTO crosswalk (era, source_column, core_category, reason, proposed_by, verified) "
                     "VALUES (?, ?, ?, ?, ?, 1)", (era, col, cat, r["reason"], r["proposed_by"]))
        loaded[era] = loaded.get(era, 0) + 1
    conn.commit()
    return {"loaded": loaded, "still_unverified": pending}


def verified_mapping(conn, era: str) -> dict[str, str | None]:
    """For the loaders: source column -> core category (None = not one of the seven)."""
    return {r["source_column"]: r["core_category"]
            for r in conn.execute("SELECT source_column, core_category FROM crosswalk WHERE era = ? AND verified = 1", (era,))}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Category crosswalk: propose with the model, apply after human review.")
    ap.add_argument("command", choices=["propose", "apply"])
    args = ap.parse_args(argv)
    conn = store.connect()
    if args.command == "propose":
        for era, s in propose(build_client(conn), conn).items():
            print(f"{era}: {s}")
        rows = read_rows(CSV_PATH)
        print(f"\n{'era':<10} {'source_column':<52} {'proposed':<21} amb  reason")
        for key in sorted(rows, key=lambda k: (ERA_ORDER[k[0]], k[1])):
            r = rows[key]
            if r["era"] != "2001_2014":
                print(f"{r['era']:<10} {r['source_column']:<52} {r['core_category'] or '-':<21} {r['ambiguous']:<4} {r['reason'][:80]}")
        print(f"\nreview file -> {CSV_PATH}")
    else:
        print(apply(conn))


if __name__ == "__main__":
    main()
