"""Rules a record must satisfy beyond what the schema already enforces.

The schema guarantees the cheap facts (counts >= 0, total = sum of parts, year in range, codes that
exist). This module checks what needs context, and writes the outcome to ``records.validation_status``:

* ``resolved``      a record must point at a district, or be sent to the state's unallocated bucket by a
                    verified unit-map row (railway police, a crime cell). NULL with no mapping stays
                    ``pending``: it is not wrong, it is undecided.
* ``year_in_era``   2001-2014 / 2016 / 2017-2022; a year outside its era's range is a file defect.
* ``duplicate_key`` two records for the same unit and year (the J&K railways 2010 pair). Both are kept,
                    both are flagged; the panel sums them.
* ``spike``         a category count at least SPIKE_MIN and more than SPIKE_FACTOR times the unit's median
                    for that category over its other years. Flag only: a real crime wave looks the same as
                    a typo, so a human decides.
* ``collapse``      the mirror image: a count below a tenth of a median that is itself at least SPIKE_MIN
                    (a district reporting 1 kidnapping after years of 140). A missing value coded as zero
                    looks exactly like this.

Flags never change a count. A flagged record is ``escalated`` (kept, visible in the queue); an unflagged,
resolved record is ``valid``. Records the loader already escalated (an empty count cell) stay escalated.
Escalations are keyed by the raw row (era, source_file, source_row), so a rebuild queues nothing twice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from agents import store
from agents.loaders import NAME_KEYED_ERAS, UNALLOCATED_KINDS
from agents.schemas import CORE_CATEGORIES
from data_ingestion.normalize import norm, norm_state

SPIKE_FACTOR = 10
SPIKE_MIN = 50
MIN_YEARS_FOR_SPIKE = 4                  # a median over fewer years says little


def era_years(conn) -> dict[str, tuple[int, int]]:
    """Each era's year range, from the eras table (one source of truth, seeded by the store)."""
    return {r["era"]: (int(r["first_year"]), int(r["last_year"])) for r in conn.execute("SELECT era, first_year, last_year FROM eras")}


def _records(conn, era: str | None) -> pd.DataFrame:
    sql = ("SELECT r.id, r.raw_row_id, r.era, r.state, r.district, r.registration_circle, r.state_code, r.district_code, "
           "r.year, r.validation_status, w.source_file, w.source_row, " + ", ".join(f"r.{c}" for c in CORE_CATEGORIES)
           + " FROM records r JOIN raw_rows w ON w.id = r.raw_row_id")
    params: tuple = ()
    if era:
        sql, params = sql + " WHERE r.era = ?", (era,)
    return pd.DataFrame([dict(r) for r in conn.execute(sql, params)])


def _unit_key(df: pd.DataFrame) -> pd.Series:
    """What 'the same reporting unit' means: the normalised (state, unit) name in 2001-14, the
    (district code, circle) pair in the 2016+ files."""
    keys = []
    for era, state, district, code, circle in zip(df["era"], df["state"], df["district"], df["district_code"],
                                                  df["registration_circle"]):
        if era in NAME_KEYED_ERAS:
            keys.append(f"{era}|{norm_state(state)}|{norm(district)}")
        else:
            keys.append(f"{era}|{'' if pd.isna(code) else int(code)}|{norm(circle or '')}")
    return pd.Series(keys, index=df.index, dtype=object)


def spikes(df: pd.DataFrame) -> dict[int, list[dict]]:
    """Leave-one-out median per unit and category. A point more than SPIKE_FACTOR times its unit's median
    over the other years is a spike; a point below a tenth of a median that is itself at least SPIKE_MIN is
    a collapse. Both are flags for a human: a wave, a reform, a typo and a missing value all look like this."""
    found: dict[int, list[dict]] = {}
    unit = _unit_key(df)
    for cat in CORE_CATEGORIES:
        for _, g in df.groupby(unit):
            if len(g) < MIN_YEARS_FOR_SPIKE:
                continue
            values = g[cat].to_numpy(dtype=float)
            for pos, (idx, v) in enumerate(zip(g.index, values)):
                median = float(np.median(np.delete(values, pos)))
                year = int(df.at[idx, "year"])
                if v >= SPIKE_MIN and v > SPIKE_FACTOR * max(median, 1.0):
                    found.setdefault(int(df.at[idx, "id"]), []).append(
                        {"rule": "spike", "field": cat, "message": f"{int(v)} in {year} vs a median of {median:g} over the unit's other years"})
                elif median >= SPIKE_MIN and v * SPIKE_FACTOR < median:
                    found.setdefault(int(df.at[idx, "id"]), []).append(
                        {"rule": "collapse", "field": cat, "message": f"{int(v)} in {year} vs a median of {median:g} over the unit's other years"})
    return found


def validate(conn, era: str | None = None) -> dict:
    df = _records(conn, era)
    if df.empty:
        return {"records": 0}
    violations: dict[int, list[dict]] = {}

    def add(record_id: int, v: dict) -> None:
        violations.setdefault(int(record_id), []).append(v)

    # resolved: NULL district is fine only when a verified mapping says the unit has no district
    bucket = {(r["era"], r["state"], r["unit"]) for r in conn.execute(
        "SELECT era, state, unit FROM unit_map WHERE verified = 1 AND district_code IS NULL AND kind IN "
        "(" + ", ".join("?" * len(UNALLOCATED_KINDS)) + ")", UNALLOCATED_KINDS)}
    unresolved = df["district_code"].isna()
    for idx in df.index[unresolved]:
        r = df.loc[idx]
        if (r["era"], norm_state(r["state"]), norm(r["district"])) not in bucket:
            add(r["id"], {"rule": "unresolved", "field": "district_code", "message": "no district and no verified bucket"})

    # year inside the era
    ranges = era_years(conn)
    for idx in df.index:
        lo, hi = ranges[df.at[idx, "era"]]
        if not lo <= df.at[idx, "year"] <= hi:
            add(df.at[idx, "id"], {"rule": "year_in_era", "field": "year", "message": f"{df.at[idx, 'year']} is outside {lo}-{hi}"})

    # duplicates: same unit, same year
    dup = pd.Series([f"{k}|{y}" for k, y in zip(_unit_key(df), df["year"])], index=df.index).duplicated(keep=False)
    for idx in df.index[dup]:
        add(df.at[idx, "id"], {"rule": "duplicate_key", "field": "year", "message": "another record has the same unit and year; both kept, the panel sums them"})

    for record_id, found in spikes(df).items():
        for v in found:
            add(record_id, v)

    # outcome per record
    by_rule: dict[str, int] = {}
    queued = 0
    status: dict[int, str] = {}
    for idx in df.index:
        rid, current = int(df.at[idx, "id"]), df.at[idx, "validation_status"]
        found = violations.get(rid, [])
        for v in found:
            by_rule[v["rule"]] = by_rule.get(v["rule"], 0) + 1
        if any(v["rule"] == "unresolved" for v in found):
            status[rid] = "pending"
        elif found or current == "escalated":
            status[rid] = "escalated"
        else:
            status[rid] = "valid"
        if found:
            ref = {"era": df.at[idx, "era"], "source_file": df.at[idx, "source_file"], "source_row": int(df.at[idx, "source_row"]),
                   "state": df.at[idx, "state"], "district": df.at[idx, "district"], "year": int(df.at[idx, "year"])}
            queued += store.add_escalation(conn, kind="record", ref=ref, violations=found)
    with conn:
        conn.executemany("UPDATE records SET validation_status = ? WHERE id = ?", [(s, rid) for rid, s in status.items()])
    counts = pd.Series(status).value_counts().to_dict()
    return {"records": int(len(df)), "valid": int(counts.get("valid", 0)), "escalated": int(counts.get("escalated", 0)),
            "pending": int(counts.get("pending", 0)), "violations_by_rule": by_rule, "escalations_queued": int(queued)}
