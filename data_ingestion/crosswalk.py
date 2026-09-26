"""Category crosswalk: which source column feeds which harmonised category.

Two tiers, but only ONE decision per column:

* ``category``       the fine, harmonised category (``agents.schemas.EXTENDED_CATEGORIES``).
                     Every column of every era gets exactly one. Categories that only
                     exist from 2016 (acid attack, POCSO, cyber, ...) are kept and shown
                     for those years, never stretched back to 2001.
* ``core_category``  derived, never typed: the same name when ``category`` is one of the
                     seven that exist in every era (the 2001-2022 series), else empty.

Era A (2001-2014) has exactly the seven core columns, so it is mapped by definition.
Eras B (2016) and C (2017+) are proposed by the model, one call per era, into
``data/crosswalk.csv`` with ``verified = 0``. A human edits that one file (fix
``category`` if wrong, set ``verified = 1``); ``apply`` loads only verified rows.

Two evidence tools decide the hard rows, so verification is arithmetic, not opinion:

* ``fidelity``    adds the district file up to national totals and compares them with
                  the figures NCRB itself published (``data/reference/``). This is how the
                  2018+ column-alignment defect was found; its correction lives in
                  ``data/column_fixes.csv`` (also human-verified) and is shown side by side.
* ``continuity``  for every column that feeds a core category, what the national total
                  does at the era boundary with and without that column. A correct mapping
                  does not jump exactly when the file format changes.

    python -m data_ingestion.crosswalk propose      # model proposals -> data/crosswalk.csv (verified rows untouched)
    python -m data_ingestion.crosswalk fidelity     # file vs NCRB-published national figures, with and without fixes
    python -m data_ingestion.crosswalk continuity   # boundary evidence -> printed table + the 'evidence' column
    python -m data_ingestion.crosswalk apply        # verified rows -> crosswalk table
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from agents import store
from agents.llm_client import build_client
from agents.prompts import PROJECT_CONTEXT
from agents.schemas import CORE_CATEGORIES, EXTENDED_CATEGORIES
from data_ingestion.profile import KEY_COLS_BC

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CSV_PATH = ROOT / "data" / "crosswalk.csv"
FIXES_PATH = ROOT / "data" / "column_fixes.csv"
REFERENCE_PATH = ROOT / "data" / "reference" / "ncrb_published_national.csv"
ERA_FILES = {"2001_2014": RAW / "crimes_against_women_2001-2014.csv",
             "2016": RAW / "districtwise-crime-against-women-2016.csv",
             "2017_plus": RAW / "districtwise-crime-against-women-2017-onwards.csv"}
MODEL_ERAS = ("2016", "2017_plus")
ERA_LABEL = {"2016": "2016", "2017_plus": "2017-2022"}
ERA_ORDER = {"2001_2014": 0, "2016": 1, "2017_plus": 2}
BOUNDARY = {"2016": (2014, 2016), "2017_plus": (2016, 2017)}     # last year of the previous era -> first year of this one
CSV_COLUMNS = ["era", "source_column", "category", "core_category", "reason", "ambiguous", "evidence",
               "proposed_by", "verified", "reviewer_note"]
FIDELITY_TOLERANCE = 0.005      # a reconstructed national total must land within 0.5% of NCRB's published figure

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

EXTENDED_NOTES: dict[str, str] = {
    "attempt_to_rape": "IPC 376 read with 511; a separate NCRB head since 2014",
    "murder_with_rape": "IPC 376A; 2017+ only",
    "acid_attack": "IPC 326A",
    "attempt_to_acid_attack": "IPC 326B",
    "abetment_of_suicide": "IPC 306",
    "miscarriage_offences": "IPC 312-318",
    "human_trafficking": "IPC 370 / 370A",
    "unnatural_offences": "IPC 377; 2016 file only",
    "dowry_prohibition_act": "Dowry Prohibition Act, 1961 (not IPC 304B)",
    "domestic_violence_act": "Protection of Women from Domestic Violence Act, 2005",
    "immoral_traffic_act": "Immoral Traffic (Prevention) Act, 1956, all sections",
    "indecent_representation_act": "Indecent Representation of Women (Prohibition) Act, 1986",
    "pocso_act": "Protection of Children from Sexual Offences Act, 2012, all heads; 2017+ only",
    "cyber_crimes": "IT Act and other women-centric cyber crimes; 2017+ only",
}

Category = Literal[EXTENDED_CATEGORIES]


class ColumnMapping(BaseModel):
    source_column: str
    category: Category = Field(description="exactly one harmonised category from the allowed list")
    reason: str = Field(min_length=3, description="one line, citing the section or act")
    ambiguous: bool = Field(default=False, description="true when reasonable statisticians could disagree")


class CrosswalkProposal(BaseModel):
    mappings: list[ColumnMapping]


def core_of(category: str) -> str:
    """The core name when the category is one of the seven, else empty. Derived, never typed."""
    return category if category in CORE_CATEGORIES else ""


def era_columns(era: str) -> list[str]:
    """The category columns of one era's file, in file order (header only, no rows read)."""
    return [c for c in pd.read_csv(ERA_FILES[era], nrows=0).columns if c not in KEY_COLS_BC]


def build_prompt(era: str, columns: list[str], missing=(), unknown=()) -> str:
    core = "\n".join(f'  {cat:<22} = "{col}" ({law})' for col, (cat, law) in ERA_A_DEFINITIONS.items())
    extended = "\n".join(f"  {name:<28} {note}" for name, note in EXTENDED_NOTES.items())
    lines = [
        "Task: assign every category column of one era of NCRB district-level 'crimes against women' data to "
        "exactly one harmonised category.",
        "",
        "CORE categories (exist in every year 2001-2022; defined by the 2001-2014 table columns):",
        core,
        "",
        "EXTENDED categories (exist only from 2016 or 2017; kept for those years, never stretched back):",
        extended,
        "",
        f"Below are the {len(columns)} category columns of the {ERA_LABEL[era]} file.",
        "",
        "Rules:",
        "1. Every column appears exactly once, spelled exactly as given, with exactly one category from the lists above.",
        "2. A column may take a CORE category ONLY if NCRB's published head for the 2001-2014 table would have counted "
        "the same cases under it. Attempts, POCSO Act offences, special-act offences and cyber offences are separate "
        "NCRB heads: give them their own EXTENDED category, never a core one.",
        "3. Age-split IPC columns ('above_18' / 'below_18') belong to their IPC head regardless of the victim's age. "
        "Only columns that name the POCSO Act itself or a POCSO-only offence (child rape, sexual assault of children, "
        "child sexual harassment, use of a child for pornography) are pocso_act.",
        "4. Any IPC section from 363 to 373 belongs to kidnapping_abduction, except 366B (importation_of_girls) and "
        "370/370A (human_trafficking), which NCRB reports as separate heads.",
        "5. Several columns may share a category (their counts are added together). Never split one column across "
        "categories.",
        "6. Map by legal meaning (the section or act), not by similar words.",
        "7. Set ambiguous = true when reasonable statisticians could disagree, and say why in the reason.",
        "8. reason: one line, cite the section or act.",
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
            r = {c: rows[key].get(c, "") for c in CSV_COLUMNS}
            r["core_category"] = core_of(r["category"])           # derived, never typed: recomputed on every write
            writer.writerow(r)


def propose(client, conn, *, csv_path: Path = CSV_PATH, columns_by_era: dict[str, list[str]] | None = None) -> dict:
    """Write Era A by definition; ask the model about every unverified column of Eras B and C.
    Rows a human has already verified are never touched."""
    columns_by_era = columns_by_era or {era: era_columns(era) for era in MODEL_ERAS}
    rows, order, summary = read_rows(csv_path), {}, {}

    for i, (col, (cat, law)) in enumerate(ERA_A_DEFINITIONS.items()):
        key = ("2001_2014", col)
        order[key] = i
        rows[key] = {"era": "2001_2014", "source_column": col, "category": cat, "core_category": cat,
                     "reason": f"defines the core category ({law})", "ambiguous": "0", "evidence": "",
                     "proposed_by": "definition", "verified": "1",
                     "reviewer_note": rows.get(key, {}).get("reviewer_note", "")}

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
                              "category": m.category if m else "",
                              "core_category": core_of(m.category) if m else "",
                              "reason": m.reason if m else "NOT PROPOSED: see escalation queue",
                              "ambiguous": "1" if (m and m.ambiguous) else "0",
                              "evidence": "",
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


# ---- column fixes: the source file put some values under the wrong header ----------------

def read_fixes(path: Path = FIXES_PATH, *, only_verified: bool = False) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        fixes = list(csv.DictReader(fh))
    return [f for f in fixes if f.get("verified") == "1"] if only_verified else fixes


def apply_column_fixes(df: pd.DataFrame, fixes: list[dict], era: str) -> pd.DataFrame:
    """Move values from the column the file put them in to the column they belong to,
    for the year range of each fix. All moves of one era are applied at once, from a
    copy: a swap or a rotation must never be done one column at a time."""
    out = df.copy()
    for f in fixes:
        if f["era"] != era or f["file_column"] not in df.columns or f["actual_column"] not in df.columns:
            continue
        in_range = (df["year"] >= int(f["year_from"])) & (df["year"] <= int(f["year_to"]))
        out.loc[in_range, f["actual_column"]] = df.loc[in_range, f["file_column"]].values
    return out


def load_era_frames(fixes: str = "proposed") -> dict[str, pd.DataFrame]:
    """fixes: 'none' = the files exactly as they are; 'proposed' = every row of column_fixes.csv;
    'verified' = only rows a human has verified (what the loaders will use)."""
    a = pd.read_csv(ERA_FILES["2001_2014"])
    a = a[~a["DISTRICT"].str.upper().str.contains("TOTAL")]      # state total rows would double count
    frames = {"2001_2014": a, "2016": pd.read_csv(ERA_FILES["2016"]), "2017_plus": pd.read_csv(ERA_FILES["2017_plus"])}
    if fixes != "none":
        fx = read_fixes(only_verified=(fixes == "verified"))
        for era in MODEL_ERAS:
            frames[era] = apply_column_fixes(frames[era], fx, era)
    return frames


# ---- evidence 1: does the file add up to what NCRB published? ---------------------------------

def category_totals(rows: dict, frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """National totals per category per year under the current mapping, plus each
    source column's own yearly totals so the what-ifs can subtract it back out."""
    totals: dict[str, dict[int, int]] = {}
    col_totals: dict[tuple[str, str], dict[int, int]] = {}
    a = frames["2001_2014"].groupby("Year")[list(ERA_A_DEFINITIONS)].sum()
    for col, (cat, _) in ERA_A_DEFINITIONS.items():
        for y, v in a[col].items():
            totals.setdefault(cat, {})
            totals[cat][int(y)] = totals[cat].get(int(y), 0) + int(v)
    for era in MODEL_ERAS:
        by_year = frames[era].groupby("year").sum(numeric_only=True)
        for (e, col), r in rows.items():
            if e != era or col not in by_year.columns:
                continue
            col_totals[(era, col)] = {int(y): int(v) for y, v in by_year[col].items()}
            cat = r.get("category", "")
            if cat:
                totals.setdefault(cat, {})
                for y, v in col_totals[(era, col)].items():
                    totals[cat][y] = totals[cat].get(y, 0) + v
    table = pd.DataFrame(totals).T.sort_index(axis=1).fillna(0).astype(int)
    return table, col_totals


def fidelity(rows: dict, reference: pd.DataFrame | None = None, frames_by_mode: dict | None = None) -> pd.DataFrame:
    """Reconstructed national totals vs NCRB-published, for every head-year with a published figure,
    with the file as it is and with the proposed column fixes applied."""
    reference = reference if reference is not None else pd.read_csv(REFERENCE_PATH)
    frames_by_mode = frames_by_mode or {"none": load_era_frames("none"), "proposed": load_era_frames("proposed")}
    tables = {mode: category_totals(rows, frames)[0] for mode, frames in frames_by_mode.items()}

    def lookup(table: pd.DataFrame, category: str, year: int):
        if year not in table.columns:
            return None
        if category == "total":
            return int(table[year].sum())
        return int(table.at[category, year]) if category in table.index else None

    out = []
    for r in reference.itertuples():
        published, y = int(r.cases), int(r.year)
        as_is, fixed = lookup(tables["none"], r.category, y), lookup(tables["proposed"], r.category, y)
        out.append({"head": r.head, "category": r.category, "year": y, "published": published,
                    "file_as_is": as_is, "with_fixes": fixed,
                    "diff": (fixed - published) / published if fixed is not None else None})
    return pd.DataFrame(out)


def fidelity_hints(rows: dict, table: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> dict[tuple, str]:
    """For every published figure the file misses, name the single column whose own total
    closes the gap. A short total points at a column mapped elsewhere; an excess points at a
    column that should not be in the category."""
    _, col_totals = category_totals(rows, frames)
    found: dict[tuple, list[tuple[int, int, int, str]]] = {}          # (year, gap, column total, category)
    for r in table.itertuples():
        if r.category == "total" or r.with_fixes is None or pd.isna(r.with_fixes):
            continue
        gap = int(r.published - r.with_fixes)
        if abs(gap) <= FIDELITY_TOLERANCE * r.published:
            continue
        era = "2016" if r.year == 2016 else ("2017_plus" if r.year >= 2017 else None)
        for (e, col), totals in col_totals.items():
            if e != era:
                continue
            same = rows[(e, col)].get("category") == r.category
            v = totals.get(int(r.year), 0)
            if v and ((gap > 0 and not same and abs(v - gap) <= FIDELITY_TOLERANCE * r.published)
                      or (gap < 0 and same and abs(v + gap) <= FIDELITY_TOLERANCE * r.published)):
                found.setdefault((e, col), []).append((int(r.year), gap, v, r.category))
    hints = {}
    for key, items in found.items():
        years = ", ".join(f"{y}: {v:,}" for y, _, v, _ in items)
        gap0, target = items[0][1], items[0][3]
        current = rows[key]["category"]
        if gap0 > 0:
            hints[key] = (f"auto-fidelity: {target} falls short of NCRB's published figure by exactly this column's total "
                          f"({years}). Mapping it to {target} would match; it is currently {current}.")
        else:
            hints[key] = (f"auto-fidelity: {target} exceeds NCRB's published figure by exactly this column's total "
                          f"({years}). Removing it from {target} would match.")
    return hints


def merge_evidence(rows: dict, new: dict[tuple, str], prefix: str) -> None:
    """Write one tool's notes into the evidence column, replacing only that tool's old notes."""
    for key, r in rows.items():
        if r["era"] not in MODEL_ERAS:
            continue
        if key in new:
            r["evidence"] = new[key]
        elif r.get("evidence", "").startswith(prefix):
            r["evidence"] = ""


# ---- evidence 2: does the mapping create a jump exactly at the file-format change? ---------

def typical_change(series: pd.Series, years: list[int]) -> float:
    """Median size of the year-over-year change inside one era: the scale a boundary jump is judged against."""
    vals = [series[y] for y in years if y in series.index]
    changes = [abs(b - a) / a for a, b in zip(vals[:-1], vals[1:]) if a > 0]
    return float(np.median(changes)) if changes else float("nan")


def continuity(rows: dict, frames: dict[str, pd.DataFrame] | None = None) -> tuple[pd.DataFrame, dict[tuple, str]]:
    frames = frames or load_era_frames("proposed")
    table, col_totals = category_totals(rows, frames)
    evidence: dict[tuple, str] = {}
    for (era, col), r in rows.items():
        cat = core_of(r.get("category", ""))
        if era not in BOUNDARY or not cat or (era, col) not in col_totals:
            continue
        y0, y1 = BOUNDARY[era]
        if y0 not in table.columns or y1 not in table.columns or table.at[cat, y0] <= 0:
            continue
        base, with_col = table.at[cat, y0], table.at[cat, y1]
        alone = col_totals[(era, col)].get(y1, 0)
        in_era = [y for y in table.columns if y <= 2014] if era == "2016" else [y for y in table.columns if y >= 2017]
        gap = " (two-year gap, 2015 missing)" if era == "2016" else ""
        evidence[(era, col)] = (f"auto: {cat} {y0}->{y1}{gap} is {with_col / base - 1:+.0%} with this column, "
                                f"{(with_col - alone) / base - 1:+.0%} without; typical in-era yearly change "
                                f"{typical_change(table.loc[cat], sorted(in_era)):.0%}; this column alone = {alone:,} in {y1}")
    return table.loc[[c for c in CORE_CATEGORIES if c in table.index]], evidence


def annotate(csv_path: Path = CSV_PATH, frames: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Run the continuity check and write its findings into the 'evidence' column (never into reviewer_note)."""
    rows = read_rows(csv_path)
    table, evidence = continuity(rows, frames)
    merge_evidence(rows, evidence, "auto:")
    write_rows(csv_path, rows, {})
    return table


# ---- loading the human's decisions ------------------------------------------------------------

def apply(conn, *, csv_path: Path = CSV_PATH) -> dict:
    """Load verified rows into the crosswalk table. Unverified rows stay out."""
    loaded, pending = {}, {}
    for (era, col), r in read_rows(csv_path).items():
        if r.get("verified") != "1":
            pending[era] = pending.get(era, 0) + 1
            continue
        category = r["category"].strip()
        if category not in EXTENDED_CATEGORIES:
            raise ValueError(f"{era}/{col}: '{category}' is not a known category; allowed: {EXTENDED_CATEGORIES}")
        core = core_of(category)                                  # the CSV's core column is ignored: it is derived
        conn.execute("INSERT OR REPLACE INTO crosswalk (era, source_column, category, core_category, reason, proposed_by, "
                     "verified) VALUES (?, ?, ?, ?, ?, ?, 1)", (era, col, category, core or None, r["reason"], r["proposed_by"]))
        loaded[era] = loaded.get(era, 0) + 1
    conn.commit()
    return {"loaded": loaded, "still_unverified": pending}


def verified_mapping(conn, era: str) -> dict[str, str]:
    """For the loaders: source column -> harmonised category. Use core_of() for the seven-category series."""
    return {r["source_column"]: r["category"]
            for r in conn.execute("SELECT source_column, category FROM crosswalk WHERE era = ? AND verified = 1", (era,))}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Category crosswalk: propose, check fidelity and continuity, apply after review.")
    ap.add_argument("command", choices=["propose", "fidelity", "continuity", "apply"])
    args = ap.parse_args(argv)
    pd.set_option("display.width", 220)
    if args.command == "propose":
        conn = store.connect()
        for era, s in propose(build_client(conn), conn).items():
            print(f"{era}: {s}")
        rows = read_rows(CSV_PATH)
        print(f"\n{'era':<10} {'source_column':<52} {'category':<28} {'core':<21} amb  reason")
        for key in sorted(rows, key=lambda k: (ERA_ORDER[k[0]], k[1])):
            r = rows[key]
            if r["era"] != "2001_2014":
                print(f"{r['era']:<10} {r['source_column']:<52} {r['category']:<28} {r['core_category'] or '-':<21} "
                      f"{r['ambiguous']:<4} {r['reason'][:70]}")
        print(f"\nreview file -> {CSV_PATH}")
    elif args.command == "fidelity":
        rows = read_rows(CSV_PATH)
        table = fidelity(rows)
        hints = fidelity_hints(rows, table, load_era_frames("proposed"))
        merge_evidence(rows, hints, "auto-fidelity:")
        write_rows(CSV_PATH, rows, {})
        shown = table.copy()
        shown["status"] = shown["diff"].map(lambda d: "" if d is None or pd.isna(d) else ("OK" if abs(d) <= FIDELITY_TOLERANCE else "MISMATCH"))
        shown["diff"] = shown["diff"].map(lambda d: "" if d is None or pd.isna(d) else f"{d:+.2%}")
        print("file (district rows added up nationally) vs NCRB-published figures; fixes = data/column_fixes.csv, all rows:\n")
        print(shown.drop(columns=["head"]).to_string(index=False))
        n_ok, n_bad = int((shown["status"] == "OK").sum()), int((shown["status"] == "MISMATCH").sum())
        print(f"\nwith fixes: {n_ok} OK, {n_bad} MISMATCH (tolerance {FIDELITY_TOLERANCE:.1%}); 'total' is informational "
              "(NCRB's total includes heads not in this list)")
        if hints:
            print("\nhints written to the 'evidence' column:")
            for (era, col), h in sorted(hints.items()):
                print(f"  {era:<10} {col:<48} {h}")
    elif args.command == "continuity":
        table = annotate()
        print("national totals per core category under the CURRENT mapping (verified or not), with proposed column fixes:\n")
        print(table.to_string())
        print("\nevidence written to the 'evidence' column for every column that feeds a core category:")
        for key, r in sorted(read_rows(CSV_PATH).items(), key=lambda kv: (ERA_ORDER[kv[0][0]], kv[0][1])):
            if r["evidence"]:
                print(f"  {r['era']:<10} {r['source_column']:<52} {r['evidence']}")
    else:
        print(apply(store.connect()))


if __name__ == "__main__":
    main()
