"""The observed panel: one row per reporting unit and year, the seven core counts summed over the
unit's records, as the files report them after mapping. No boundary arithmetic here; re-basing to
today's districts is a later step that reads this table.

A unit is a district (``unit_id = d<code>``) or a state's unallocated bucket (``s<state_code>u``): the
railway police and crime cells a verified mapping sends to no district. Records are included when they
are ``valid`` or ``escalated`` (flagged, kept); ``pending`` records are counted and left out.

Written twice: the ``panel_observed`` table in the store, and ``data/panel_observed.parquet`` for anything
outside the store (the model, a notebook, a reviewer). Parquet is a typed, compressed, column-oriented
file that pandas reads in one line.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from agents.schemas import CORE_CATEGORIES

ROOT = Path(__file__).resolve().parents[1]
PARQUET_PATH = ROOT / "data" / "panel_observed.parquet"
COLUMNS = ["unit_id", "state_code", "district_code", "year", "era", *CORE_CATEGORIES, "total", "n_records"]


def observed(conn) -> pd.DataFrame:
    sums = ", ".join(f"SUM({c}) AS {c}" for c in CORE_CATEGORIES)
    rows = conn.execute(
        f"SELECT era, state_code, district_code, year, {sums}, SUM(total) AS total, COUNT(*) AS n_records "
        "FROM records WHERE validation_status IN ('valid', 'escalated') "
        "GROUP BY era, state_code, district_code, year ORDER BY state_code, district_code, year").fetchall()
    df = pd.DataFrame([dict(r) for r in rows], columns=["era", "state_code", "district_code", "year", *CORE_CATEGORIES, "total", "n_records"])
    df["district_code"] = df["district_code"].astype("Int64")
    df["unit_id"] = [f"d{int(c)}" if pd.notna(c) else f"s{int(s)}u" for c, s in zip(df["district_code"], df["state_code"])]
    return df[COLUMNS]


def build_panel(conn, parquet_path: Path | None = None) -> dict:
    df = observed(conn)
    with conn:
        conn.execute("DELETE FROM panel_observed")
        conn.executemany(
            "INSERT INTO panel_observed (" + ", ".join(COLUMNS) + ") VALUES (" + ", ".join("?" * len(COLUMNS)) + ")",
            [tuple(None if pd.isna(v) else int(v) if not isinstance(v, str) else v for v in row) for row in df.itertuples(index=False)])
    path = Path(parquet_path or PARQUET_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    pending = conn.execute("SELECT COUNT(*) FROM records WHERE validation_status = 'pending'").fetchone()[0]
    return {"rows": int(len(df)), "districts": int(df["district_code"].dropna().nunique()),
            "unallocated_rows": int(df["district_code"].isna().sum()), "years": [int(df["year"].min()), int(df["year"].max())] if len(df) else [],
            "records_left_out_pending": int(pending), "parquet": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)}
