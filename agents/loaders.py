"""Era loaders: every source row into ``raw_rows`` unchanged, then one record per row in the seven
core categories, built only from verified decisions.

Two steps per era, both rerunnable:

* ``load_raw_rows``  copies the file into ``raw_rows`` as JSON, one row per line, once. The table's
  UNIQUE (source_file, source_row) makes a second run a no-op. The file is never edited: column
  fixes and the crosswalk are applied when *reading* the raw rows, so a rebuild always starts from
  the bytes that came from the portal. A CSV is read with the csv module; NCRB's xlsx workbook
  through ``data_ingestion.ncrb_xlsx`` (``source_row`` is then the Excel row number).
* ``build_records``  deletes the era's records and rebuilds them from ``raw_rows``:
    - 2001-2014: the seven columns are the seven categories by definition; the district comes from
      the verified unit map (``(state, unit) -> district_code``) and a unit mapped to no district
      (railway police, a crime cell) keeps ``district_code = NULL`` on purpose: it counts in the
      state's unallocated bucket, which the validator accepts as resolved.
    - 2015: the workbook's 60 heads go through the verified crosswalk (totals counted, their parts
      'excluded'); the district comes from the unit map like 2001-14, because the workbook has no codes.
    - 2016 and 2017+: verified column fixes first (the 2018-22 column rotation), then the verified
      crosswalk folds source columns into categories; core categories go to ``records``, the other
      fourteen to ``extended_counts``. Codes come from the file unless a verified unit-map row for
      that era overrides the circle (the Sipahijala pattern).

A count cell that is empty in a column the file does not carry at all that year (``kidnapping_for_ransom``
from 2019) is "not reported" and contributes nothing, counted in the summary. An empty cell in a column
that other rows of the same year do fill is a violation: the record is stored with 0 for that cell,
marked ``escalated`` and queued, so nothing is dropped and nothing is silently changed.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from agents import store
from agents.resolver import load_registry, map_states
from agents.schemas import CORE_CATEGORIES, CrimeRecord
from data_ingestion import ncrb_xlsx
from data_ingestion.crosswalk import ERA_A_DEFINITIONS, ERA_FILES, apply_column_fixes, core_of, read_fixes, verified_mapping
from data_ingestion.normalize import norm, norm_state

ERA_A = "2001_2014"
NAME_KEYED_ERAS = ("2001_2014", "2015")          # no codes in the file: state + unit name resolve through the unit map
# A verified mapping to one of these kinds says "this unit has no district": the state's unallocated bucket.
UNALLOCATED_KINDS = ("railway", "not_a_district", "unknown")

_INSERT_RECORD = (
    "INSERT INTO records (raw_row_id, era, state, district, registration_circle, state_code, district_code, year, "
    + ", ".join(CORE_CATEGORIES) + ", validation_status) VALUES (" + ", ".join("?" * (8 + len(CORE_CATEGORIES) + 1)) + ")"
)


# ---- raw rows ----------------------------------------------------------------------------------------------

def load_raw_rows(conn, era: str, path: Path | None = None) -> dict:
    """Copy one source file into raw_rows, one JSON object per row, exactly as the file spells it."""
    path = Path(path or ERA_FILES[era])
    if path.suffix.lower() == ".xlsx":
        rows = [(era, path.name, r["source_row"], json.dumps(r, ensure_ascii=False)) for r in ncrb_xlsx.read_district_rows(path)]
    else:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            rows = [(era, path.name, i, json.dumps(r, ensure_ascii=False)) for i, r in enumerate(csv.DictReader(fh))]
    before = conn.execute("SELECT COUNT(*) FROM raw_rows WHERE era = ?", (era,)).fetchone()[0]
    with conn:
        conn.executemany("INSERT OR IGNORE INTO raw_rows (era, source_file, source_row, payload_json) VALUES (?, ?, ?, ?)", rows)
    after = conn.execute("SELECT COUNT(*) FROM raw_rows WHERE era = ?", (era,)).fetchone()[0]
    return {"era": era, "file": path.name, "rows_in_file": len(rows), "inserted": after - before, "raw_rows": after}


def raw_frame(conn, era: str) -> pd.DataFrame:
    """The era's raw rows as a table: the payload columns plus raw_row_id, source_file, source_row."""
    rows = conn.execute("SELECT id, source_file, source_row, payload_json FROM raw_rows WHERE era = ? ORDER BY id",
                        (era,)).fetchall()
    df = pd.DataFrame([json.loads(r["payload_json"]) for r in rows])
    df.insert(0, "raw_row_id", [r["id"] for r in rows])
    df.insert(1, "source_file", [r["source_file"] for r in rows])
    if "source_row" in df.columns:                                       # the xlsx reader already wrote it into the payload
        df = df.drop(columns=["source_row"])
    df.insert(2, "source_row", [r["source_row"] for r in rows])
    return df


# ---- records -----------------------------------------------------------------------------------------------

def _clear_era(conn, era: str) -> None:
    with conn:
        conn.execute("DELETE FROM extended_counts WHERE record_id IN (SELECT id FROM records WHERE era = ?)", (era,))
        conn.execute("DELETE FROM records WHERE era = ?", (era,))


def _unit_map(conn, era: str) -> dict[tuple[str, str], tuple[int | None, str]]:
    return {(r["state"], r["unit"]): (r["district_code"], r["kind"] or "")
            for r in conn.execute("SELECT state, unit, district_code, kind FROM unit_map WHERE era = ? AND verified = 1", (era,))}


def _insert(conn, rec: CrimeRecord, status: str = "pending") -> int:
    cur = conn.execute(_INSERT_RECORD, (rec.raw_row_id, rec.era, rec.state, rec.district, rec.registration_circle,
                                        rec.state_code, rec.district_code, rec.year,
                                        *(getattr(rec, c) for c in CORE_CATEGORIES), status))
    return cur.lastrowid


class _Record(CrimeRecord):
    """CrimeRecord plus the raw row it came from, which is what the records table keys on."""
    raw_row_id: int


def build_records_era_a(conn, *, registry: dict | None = None) -> dict:
    """2001-2014: seven columns, seven categories; district from the verified unit map."""
    df = raw_frame(conn, ERA_A)
    if df.empty:
        return {"era": ERA_A, "records": 0}
    registry = registry or load_registry()
    is_total = df["DISTRICT"].str.upper().str.contains("TOTAL", na=False)       # state sums, not units
    state_norm = df["STATE/UT"].map(norm_state)
    unit_norm = df["DISTRICT"].map(norm)
    state_codes = map_states(sorted(state_norm[~is_total].unique()), registry)
    unit_map = _unit_map(conn, ERA_A)
    summary = {"era": ERA_A, "records": 0, "state_total_rows_skipped": int(is_total.sum()), "unallocated": 0,
               "unmapped_units": set()}
    _clear_era(conn, ERA_A)
    with conn:
        for i, row in df.iterrows():
            if is_total[i]:
                continue
            code, kind = unit_map.get((state_norm[i], unit_norm[i]), (None, None))
            if kind is None:
                summary["unmapped_units"].add(f"{state_norm[i]}/{unit_norm[i]}")
            elif code is None:
                summary["unallocated"] += 1
            rec = _Record(raw_row_id=int(row["raw_row_id"]), era=ERA_A, source_file=row["source_file"],
                          source_row=int(row["source_row"]), state=row["STATE/UT"], district=row["DISTRICT"],
                          state_code=state_codes[state_norm[i]], district_code=code, year=int(row["Year"]),
                          **{cat: int(row[col]) for col, (cat, _) in ERA_A_DEFINITIONS.items()})
            _insert(conn, rec)
            summary["records"] += 1
    summary["unmapped_units"] = sorted(summary["unmapped_units"])
    return summary


def _fold_counts(row, mapping: dict[str, str], not_reported: pd.DataFrame, year: int) -> tuple[dict, dict, list[dict], int]:
    """One raw row's crime cells -> (core counts, extended counts, violations, not-reported cells)."""
    core = {c: 0 for c in CORE_CATEGORIES}
    extended: dict[str, int] = {}
    violations: list[dict] = []
    skipped = 0
    for col, cat in mapping.items():
        v = row[col]
        if pd.isna(v):
            if not_reported.loc[year, col]:
                skipped += 1
            else:
                violations.append({"rule": "missing_count", "field": col, "message": "empty cell, stored as 0"})
            continue
        if v != int(v):
            violations.append({"rule": "non_integer_count", "field": col, "message": f"{v} is not a whole number"})
        target = core_of(cat)
        if target:
            core[target] += int(v)
        else:
            extended[cat] = extended.get(cat, 0) + int(v)
    return core, extended, violations, skipped


def build_records_with_crosswalk(conn, era: str, *, fixes: str = "verified", fixes_path: Path | None = None,
                                 registry: dict | None = None) -> dict:
    """2015, 2016, 2017+: the verified crosswalk folds columns into categories. Codes come from the file
    (2016+, unit-map overrides win) or from the unit map by state and unit name (2015)."""
    df = raw_frame(conn, era)
    if df.empty:
        return {"era": era, "records": 0}
    mapping = verified_mapping(conn, era)                                   # source column -> category
    if not mapping:
        raise ValueError(f"no verified crosswalk rows for era {era}: run `python -m data_ingestion.crosswalk apply` first")
    for col in mapping:
        df[col] = pd.to_numeric(df[col], errors="coerce")                   # '' -> NaN, '20.0' -> 20.0
    df["year"] = pd.to_numeric(df["YEAR" if "YEAR" in df.columns else "year"]).astype(int)
    if fixes != "none":
        fx = read_fixes(**({"path": fixes_path} if fixes_path else {}), only_verified=(fixes == "verified"))
        df = apply_column_fixes(df, fx, era)
    not_reported = df.groupby("year")[list(mapping)].apply(lambda g: g.isna().all())   # year x column: file lacks it
    named = era in NAME_KEYED_ERAS
    if named:
        state_norm = df["state_name"].map(norm_state)
        unit_norm = df["district"].map(norm)
        state_codes = map_states(sorted(state_norm.unique()), registry or load_registry())
        unit_map = _unit_map(conn, era)
    else:
        overrides = {k: v[0] for k, v in _unit_map(conn, era).items()}    # a verified row for this era wins over the file
    summary = {"era": era, "records": 0, "extended_rows": 0, "not_reported_cells": 0, "overrides_applied": 0,
               "records_with_missing_counts": 0, "unallocated": 0, "unmapped_units": set()}
    escalations: list[tuple[dict, list[dict]]] = []
    _clear_era(conn, era)
    with conn:
        for i, row in df.iterrows():
            year = int(row["year"])
            if named:
                code, kind = unit_map.get((state_norm[i], unit_norm[i]), (None, None))
                if kind is None:
                    summary["unmapped_units"].add(f"{state_norm[i]}/{unit_norm[i]}")
                elif code is None:
                    summary["unallocated"] += 1
                state_code, district, circle = state_codes[state_norm[i]], row["district"], None
            else:
                key = (norm_state(row["state_name"]), norm(row["registration_circles"]))
                if key in overrides:                                        # may be NULL: sent to the unallocated bucket
                    code = overrides[key]
                    summary["overrides_applied"] += 1
                else:
                    code = int(row["district_code"])
                state_code, district, circle = int(row["state_code"]), row["district_name"], row["registration_circles"]
            core, extended, violations, skipped = _fold_counts(row, mapping, not_reported, year)
            summary["not_reported_cells"] += skipped
            rec = _Record(raw_row_id=int(row["raw_row_id"]), era=era, source_file=row["source_file"],
                          source_row=int(row["source_row"]), state=row["state_name"], district=district,
                          registration_circle=circle, state_code=state_code, district_code=code, year=year, **core)
            record_id = _insert(conn, rec, "escalated" if violations else "pending")
            conn.executemany("INSERT INTO extended_counts (record_id, category, count) VALUES (?, ?, ?)",
                             [(record_id, cat, n) for cat, n in extended.items()])
            summary["records"] += 1
            summary["extended_rows"] += len(extended)
            if violations:
                summary["records_with_missing_counts"] += 1
                escalations.append(({"era": era, "source_file": rec.source_file, "source_row": rec.source_row,
                                     "state": rec.state, "district": rec.district, "circle": rec.registration_circle,
                                     "year": year}, violations))
    for ref, violations in escalations:
        store.add_escalation(conn, kind="record", ref=ref, violations=violations)
    summary["escalations_queued"] = len(escalations)
    summary["unmapped_units"] = sorted(summary["unmapped_units"])
    if not named:
        del summary["unallocated"], summary["unmapped_units"]
    return summary


def build_records(conn, era: str, *, fixes: str = "verified", fixes_path: Path | None = None,
                  registry: dict | None = None) -> dict:
    if era == ERA_A:
        return build_records_era_a(conn, registry=registry)
    return build_records_with_crosswalk(conn, era, fixes=fixes, fixes_path=fixes_path, registry=registry)
