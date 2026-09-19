"""The build-time pipeline store: one SQLite file, schema defined here.

Everything the ingestion layer produces or consumes at build time lives in
this database: the LLM response cache and token ledger, immutable raw rows,
converted records, the unit -> district map, the category crosswalk and the
human escalation queue. Nothing at request time (the API) reads it; the API
reads committed artifacts.

Design rules the schema enforces, so that no module has to remember them:

* ``raw_rows`` is append-only provenance. Records point at the raw row they
  came from; a rebuild regenerates ``records`` *from* ``raw_rows``.
* Counts cannot be negative and ``total`` is a generated column, so a record
  whose total disagrees with its parts cannot exist.
* Anything an LLM produced lands with ``verified = 0``. Loaders refuse
  unverified rows; a human flips the bit.
* ``district_code`` on ``records`` is nullable on purpose: Era A records exist
  before the resolver has run. The validator refuses to mark a record valid
  while it is NULL.
* ``llm_calls`` holds saved answers (the cache). The cache key is a hash of the
  *request* (intended model, temperature, prompt, schema); the row records
  which provider actually answered, so fallback answers are cached for
  reproducibility but never mislabelled.
* ``llm_attempts`` is the receipt book: one row per network try, including
  failed ones, because a failed try still uses quota and tokens. Rate-limit
  accounting and cost reports read this table, never the cache.

Run ``python -m agents.store init | counts | reset`` to manage the file.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path
from typing import get_args

from dotenv import load_dotenv

from agents.schemas import CORE_CATEGORIES, Era

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 2
ERAS: tuple[str, ...] = get_args(Era)

_era_check = "(" + ", ".join(f"'{e}'" for e in ERAS) + ")"
_category_check = "(" + ", ".join(f"'{c}'" for c in CORE_CATEGORIES) + ")"
_count_columns = ",\n    ".join(f"{c} INTEGER NOT NULL CHECK ({c} >= 0)" for c in CORE_CATEGORIES)
_total_expr = " + ".join(CORE_CATEGORIES)

DDL = f"""
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Canonical geography. Codes are LGD codes taken from the 2016+ source files,
-- never minted here. Every other table points at these via foreign key.
CREATE TABLE IF NOT EXISTS states (
    state_code   INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,
    aliases_json TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS districts (
    district_code   INTEGER PRIMARY KEY,
    state_code      INTEGER NOT NULL REFERENCES states(state_code),
    name            TEXT NOT NULL,                 -- canonical = latest-year spelling
    aliases_json    TEXT NOT NULL DEFAULT '[]',    -- every spelling seen in any source
    first_seen_year INTEGER NOT NULL,
    last_seen_year  INTEGER NOT NULL
);

-- Boundary changes between eras (splits, merges, renames). Filled by the
-- resolver; lets an analyst know a 2001-2014 series covers a larger area.
CREATE TABLE IF NOT EXISTS district_lineage (
    parent_code INTEGER NOT NULL REFERENCES districts(district_code),
    child_code  INTEGER NOT NULL REFERENCES districts(district_code),
    kind        TEXT NOT NULL CHECK (kind IN ('split', 'merge', 'rename')),
    year        INTEGER,
    source      TEXT NOT NULL,                     -- 'llm:<model>' | 'human'
    verified    INTEGER NOT NULL DEFAULT 0 CHECK (verified IN (0, 1)),
    PRIMARY KEY (parent_code, child_code)
);

-- Immutable provenance: every source row exactly as it came, once.
CREATE TABLE IF NOT EXISTS raw_rows (
    id           INTEGER PRIMARY KEY,
    era          TEXT NOT NULL CHECK (era IN {_era_check}),
    source_file  TEXT NOT NULL,
    source_row   INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    UNIQUE (source_file, source_row)
);

-- One CrimeRecord per raw row, in the seven core categories.
CREATE TABLE IF NOT EXISTS records (
    id                  INTEGER PRIMARY KEY,
    raw_row_id          INTEGER NOT NULL UNIQUE REFERENCES raw_rows(id),
    era                 TEXT NOT NULL CHECK (era IN {_era_check}),
    state               TEXT NOT NULL,
    district            TEXT NOT NULL,
    registration_circle TEXT,
    state_code          INTEGER REFERENCES states(state_code),
    district_code       INTEGER REFERENCES districts(district_code),   -- NULL until resolved
    year                INTEGER NOT NULL CHECK (year BETWEEN 2001 AND 2030),
    {_count_columns},
    total               INTEGER GENERATED ALWAYS AS ({_total_expr}) STORED,
    validation_status   TEXT NOT NULL DEFAULT 'pending'
                        CHECK (validation_status IN ('pending', 'valid', 'escalated'))
);
CREATE INDEX IF NOT EXISTS ix_records_district_year ON records(district_code, year);
CREATE INDEX IF NOT EXISTS ix_records_era_status ON records(era, validation_status);

-- Resolver output: which canonical district a source unit name refers to,
-- and how we know. Committed to data/unit_map.json once verified.
CREATE TABLE IF NOT EXISTS unit_map (
    era           TEXT NOT NULL CHECK (era IN {_era_check}),
    state         TEXT NOT NULL,                   -- normalised source spelling
    unit          TEXT NOT NULL,                   -- normalised source spelling
    district_code INTEGER REFERENCES districts(district_code),   -- NULL = unresolved
    stage         TEXT NOT NULL CHECK (stage IN ('exact', 'alias', 'fuzzy', 'llm', 'human', 'unresolved')),
    confidence    REAL CHECK (confidence BETWEEN 0 AND 1),
    reason        TEXT,
    verified      INTEGER NOT NULL DEFAULT 0 CHECK (verified IN (0, 1)),
    PRIMARY KEY (era, state, unit)
);
CREATE INDEX IF NOT EXISTS ix_unit_map_district ON unit_map(district_code);

-- Category crosswalk: source column -> core category (NULL = not one of the seven).
CREATE TABLE IF NOT EXISTS crosswalk (
    era           TEXT NOT NULL CHECK (era IN {_era_check}),
    source_column TEXT NOT NULL,
    core_category TEXT CHECK (core_category IN {_category_check}),
    reason        TEXT NOT NULL,
    proposed_by   TEXT NOT NULL,                   -- 'llm:<model>' | 'human'
    verified      INTEGER NOT NULL DEFAULT 0 CHECK (verified IN (0, 1)),
    PRIMARY KEY (era, source_column)
);

-- Saved answers (the cache). One row per distinct question that got a valid answer.
CREATE TABLE IF NOT EXISTS llm_calls (
    cache_key         TEXT PRIMARY KEY,            -- sha256(intended model|temperature|prompt|schema)
    run_id            TEXT NOT NULL,
    tag               TEXT NOT NULL,               -- e.g. 'crosswalk:2017_plus', 'resolve:telangana:warangal'
    provider          TEXT NOT NULL,               -- who actually answered
    model             TEXT NOT NULL,
    temperature       REAL NOT NULL,
    prompt_hash       TEXT NOT NULL,
    schema_hash       TEXT NOT NULL,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    latency_ms        INTEGER,
    attempts          INTEGER NOT NULL DEFAULT 1,
    response_json     TEXT NOT NULL,
    created_at        TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_llm_calls_run ON llm_calls(run_id);
CREATE INDEX IF NOT EXISTS ix_llm_calls_tag ON llm_calls(tag);

-- Receipt book. One row per network try, successful or not. Append-only.
CREATE TABLE IF NOT EXISTS llm_attempts (
    id                INTEGER PRIMARY KEY,
    run_id            TEXT NOT NULL,
    tag               TEXT NOT NULL,
    cache_key         TEXT NOT NULL,
    provider          TEXT NOT NULL,
    model             TEXT NOT NULL,
    outcome           TEXT NOT NULL CHECK (outcome IN ('ok', 'invalid_output', 'transient_error', 'permanent_error')),
    detail            TEXT,                        -- short error text; never a prompt or a key
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    latency_ms        INTEGER,
    created_at        TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_llm_attempts_provider_time ON llm_attempts(provider, model, created_at);
CREATE INDEX IF NOT EXISTS ix_llm_attempts_run ON llm_attempts(run_id);

-- Human review queue. Nothing is silently dropped; it lands here with a reason.
CREATE TABLE IF NOT EXISTS escalations (
    id              INTEGER PRIMARY KEY,
    kind            TEXT NOT NULL CHECK (kind IN ('record', 'unit', 'crosswalk', 'extraction')),
    ref_json        TEXT NOT NULL,                 -- what is being escalated (ids / keys)
    violations_json TEXT NOT NULL,                 -- list of Violation dicts
    status          TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'resolved')),
    resolution      TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    resolved_at     TEXT
);
CREATE INDEX IF NOT EXISTS ix_escalations_status ON escalations(status, kind);
"""

TABLES = ("meta", "states", "districts", "district_lineage", "raw_rows", "records",
          "unit_map", "crosswalk", "llm_calls", "llm_attempts", "escalations")


def db_path(path: str | os.PathLike | None = None) -> Path:
    """Resolve the store location: explicit arg > PIPELINE_DB env > data/pipeline.db.
    Relative paths are anchored at the repo root, never the shell's cwd."""
    load_dotenv(ROOT / ".env")
    p = Path(path or os.environ.get("PIPELINE_DB", "data/pipeline.db"))
    return p if p.is_absolute() else ROOT / p


def connect(path: str | os.PathLike | None = None) -> sqlite3.Connection:
    """Open (creating if needed) the store with the schema applied.

    ``PRAGMA foreign_keys`` is per-connection in SQLite and off by default, so
    it is set here; opening the file any other way silently disables the
    foreign-key checks the schema relies on.
    """
    p = db_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(DDL)
    conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
    conn.commit()
    found = int(conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0])
    if found != SCHEMA_VERSION:
        conn.close()
        raise RuntimeError(
            f"{p} was built with schema v{found} but the code expects v{SCHEMA_VERSION}. "
            "The store is a rebuildable workbench: run `python -m agents.store reset --yes`."
        )
    return conn


def reset(path: str | os.PathLike | None = None) -> sqlite3.Connection:
    """Delete the store file (and any journal) and recreate it empty.

    All connections must be closed first: Windows refuses to delete an open
    file, and on other platforms an open connection would keep writing to a
    ghost of the old file.
    """
    p = db_path(path)
    for suffix in ("", "-journal", "-wal", "-shm"):
        f = Path(str(p) + suffix)
        if f.exists():
            f.unlink()
    return connect(p)


def table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES}


# ---- llm_calls helpers (consumer: agents/llm_client.py) --------------------------

def get_cached_call(conn: sqlite3.Connection, cache_key: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM llm_calls WHERE cache_key = ?", (cache_key,)).fetchone()


def put_call(conn: sqlite3.Connection, *, cache_key: str, run_id: str, tag: str, provider: str, model: str,
             temperature: float, prompt_hash: str, schema_hash: str, prompt_tokens: int | None,
             completion_tokens: int | None, latency_ms: int | None, attempts: int, response_json: str) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO llm_calls
           (cache_key, run_id, tag, provider, model, temperature, prompt_hash, schema_hash,
            prompt_tokens, completion_tokens, latency_ms, attempts, response_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (cache_key, run_id, tag, provider, model, temperature, prompt_hash, schema_hash,
         prompt_tokens, completion_tokens, latency_ms, attempts, response_json),
    )
    conn.commit()


def add_attempt(conn: sqlite3.Connection, *, run_id: str, tag: str, cache_key: str, provider: str, model: str,
                outcome: str, detail: str | None = None, prompt_tokens: int | None = None,
                completion_tokens: int | None = None, latency_ms: int | None = None) -> None:
    """Write one receipt. Called for every network try, whatever happened."""
    conn.execute(
        """INSERT INTO llm_attempts
           (run_id, tag, cache_key, provider, model, outcome, detail, prompt_tokens, completion_tokens, latency_ms)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_id, tag, cache_key, provider, model, outcome, detail, prompt_tokens, completion_tokens, latency_ms),
    )
    conn.commit()


def usage_last_24h(conn: sqlite3.Connection, provider: str, model: str) -> tuple[int, int]:
    """(tries, tokens) this provider+model has used in the last 24 hours, from the receipts."""
    row = conn.execute(
        """SELECT COUNT(*), COALESCE(SUM(COALESCE(prompt_tokens, 0) + COALESCE(completion_tokens, 0)), 0)
           FROM llm_attempts
           WHERE provider = ? AND model = ? AND created_at >= datetime('now', '-1 day')""",
        (provider, model),
    ).fetchone()
    return int(row[0]), int(row[1])


# ---- CLI ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Manage the pipeline store.")
    ap.add_argument("command", choices=["init", "counts", "reset"])
    ap.add_argument("--db", help="override PIPELINE_DB")
    ap.add_argument("--yes", action="store_true", help="required for reset")
    args = ap.parse_args(argv)

    if args.command == "reset":
        if not args.yes:
            ap.error("reset deletes the store; pass --yes to confirm")
        conn = reset(args.db)
    else:
        conn = connect(args.db)

    print(f"store: {db_path(args.db)}  (schema v{SCHEMA_VERSION}, sqlite {sqlite3.sqlite_version})")
    for table, n in table_counts(conn).items():
        print(f"  {table:<17} {n:>8}")


if __name__ == "__main__":
    main()
