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
* ``circles`` lists the police reporting units under each district (2016+).
  They are how old-file unit names such as ``guntakal rly.`` find their
  district, and they live apart from district aliases because a circle is
  *inside* a district, not another name for it.
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
import json
import os
import sqlite3
from pathlib import Path
from typing import get_args

from dotenv import load_dotenv

from agents.schemas import CORE_CATEGORIES, EXTENDED_CATEGORIES, Era

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 7

# The source eras in time order: one row each in the ``eras`` table. The five tables that carry an era
# column reference it, so a new source file is a new row here, not a schema change. ``Era`` in
# schemas.py mirrors this list for the pydantic side; the assert keeps the two from drifting.
ERA_ROWS: list[tuple] = [
    # era, first_year, last_year, source_file, format, notes
    ("2001_2014", 2001, 2014, "crimes_against_women_2001-2014.csv", "csv", "7 heads; police units, no codes"),
    ("2015", 2015, 2015, "ncrb_district_wise_crimes_against_women_2015.xlsx", "xlsx",
     "NCRB workbook; 60 heads with totals beside their parts; police units, no codes"),
    ("2016", 2016, 2016, "districtwise-crime-against-women-2016.csv", "csv", "19 heads; LGD codes"),
    ("2017_plus", 2017, 2022, "districtwise-crime-against-women-2017-onwards.csv", "csv",
     "42 heads; LGD codes; 2018-22 column rotation undone at load"),
]
ERAS: tuple[str, ...] = tuple(r[0] for r in ERA_ROWS)
assert ERAS == get_args(Era), "agents.schemas.Era and store.ERA_ROWS must list the same eras"

_category_check = "(" + ", ".join(f"'{c}'" for c in CORE_CATEGORIES) + ")"
_extended_check = "(" + ", ".join(f"'{c}'" for c in EXTENDED_CATEGORIES) + ")"
_crosswalk_category_check = "(" + ", ".join(f"'{c}'" for c in (*EXTENDED_CATEGORIES, "excluded")) + ")"
_extended_only_check = "(" + ", ".join(f"'{c}'" for c in EXTENDED_CATEGORIES if c not in CORE_CATEGORIES) + ")"
_count_columns = ",\n    ".join(f"{c} INTEGER NOT NULL CHECK ({c} >= 0)" for c in CORE_CATEGORIES)
_total_expr = " + ".join(CORE_CATEGORIES)


def _districts_ddl(name: str) -> str:
    """The districts table, by name so a migration can build it under a temporary name.
    ``first_seen_year``/``last_seen_year`` are NULL and ``in_files`` is 0 for a district that is on the
    LGD map but in none of the data files (Bengaluru Rural, Medchal-Malkajgiri): the map needs its code
    and the unit map and lineage may point at it, but no record ever will."""
    return f"""
CREATE TABLE IF NOT EXISTS {name} (
    district_code   INTEGER PRIMARY KEY,
    state_code      INTEGER NOT NULL REFERENCES states(state_code),
    name            TEXT NOT NULL,                 -- canonical = latest-year spelling
    aliases_json    TEXT NOT NULL DEFAULT '[]',    -- every spelling seen in any source
    first_seen_year INTEGER,                       -- NULL = never seen in a data file
    last_seen_year  INTEGER,
    in_files        INTEGER NOT NULL DEFAULT 1 CHECK (in_files IN (0, 1))
);"""


def _lineage_ddl(name: str) -> str:
    """Boundary changes over time. 'split': child was carved out of parent. 'merge': child absorbed parent.
    'recode': same district, the files switched to a new code. 'rename': same code, new name.
    'reported_together': the parent's crimes are filed inside the child's row. Proposed by data hints and the
    model, verified by a human; this table drives the stable-unit graph. One pair can carry two facts
    (a split and, later, reported_together), so the kind is part of the key."""
    return f"""
CREATE TABLE IF NOT EXISTS {name} (
    parent_code INTEGER NOT NULL REFERENCES districts(district_code),
    child_code  INTEGER NOT NULL REFERENCES districts(district_code),
    kind        TEXT NOT NULL CHECK (kind IN ('split', 'merge', 'recode', 'rename', 'reported_together')),
    year        INTEGER,
    confidence  REAL CHECK (confidence BETWEEN 0 AND 1),
    reason      TEXT,
    source      TEXT NOT NULL,                     -- 'llm:<model>' | 'human' | 'geometry' | 'rule'
    verified    INTEGER NOT NULL DEFAULT 0 CHECK (verified IN (0, 1)),
    PRIMARY KEY (parent_code, child_code, kind)
);"""


EXTENDED_DDL = f"""
-- The 14 categories the 2016+ files add beyond the seven core ones (acid attack, POCSO, ...).
-- One row per (record, category): kept for district summaries, never part of the comparable series.
CREATE TABLE IF NOT EXISTS extended_counts (
    record_id INTEGER NOT NULL REFERENCES records(id),
    category  TEXT NOT NULL CHECK (category IN {_extended_only_check}),
    count     INTEGER NOT NULL CHECK (count >= 0),
    PRIMARY KEY (record_id, category)
);"""

ERAS_DDL = """
-- The source eras. Every table with an era column points here; a new source file is a new row.
CREATE TABLE IF NOT EXISTS eras (
    era         TEXT PRIMARY KEY,
    first_year  INTEGER NOT NULL,
    last_year   INTEGER NOT NULL CHECK (last_year >= first_year),
    source_file TEXT NOT NULL,
    format      TEXT NOT NULL CHECK (format IN ('csv', 'xlsx')),
    notes       TEXT
);"""


def _seed_eras_sql() -> str:
    values = ", ".join("(" + ", ".join(f"'{str(v).replace(chr(39), chr(39) * 2)}'" for v in r) + ")" for r in ERA_ROWS)
    return f"INSERT OR IGNORE INTO eras (era, first_year, last_year, source_file, format, notes) VALUES {values};"


def _raw_rows_ddl(name: str) -> str:
    return f"""
-- Immutable provenance: every source row exactly as it came, once.
CREATE TABLE IF NOT EXISTS {name} (
    id           INTEGER PRIMARY KEY,
    era          TEXT NOT NULL REFERENCES eras(era),
    source_file  TEXT NOT NULL,
    source_row   INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    UNIQUE (source_file, source_row)
);"""


def _records_ddl(name: str) -> str:
    return f"""
-- One CrimeRecord per raw row, in the seven core categories.
CREATE TABLE IF NOT EXISTS {name} (
    id                  INTEGER PRIMARY KEY,
    raw_row_id          INTEGER NOT NULL UNIQUE REFERENCES raw_rows(id),
    era                 TEXT NOT NULL REFERENCES eras(era),
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
);"""


def _unit_map_ddl(name: str) -> str:
    return f"""
-- Resolver output: which canonical district a source unit name refers to, and how we know.
-- Loaded from data/unit_map.csv once verified.
CREATE TABLE IF NOT EXISTS {name} (
    era           TEXT NOT NULL REFERENCES eras(era),
    state         TEXT NOT NULL,                   -- normalised source spelling
    unit          TEXT NOT NULL,                   -- normalised source spelling
    district_code INTEGER REFERENCES districts(district_code),   -- NULL = unresolved
    kind          TEXT CHECK (kind IN ('district', 'circle', 'railway', 'commissionerate', 'headquarters_town',
                                       'rename', 'not_a_district', 'unknown')),
    stage         TEXT NOT NULL CHECK (stage IN ('exact', 'alias', 'circle', 'rule', 'successor', 'fuzzy', 'geometry',
                                                 'llm', 'human', 'unresolved')),
    confidence    REAL CHECK (confidence BETWEEN 0 AND 1),
    reason        TEXT,
    verified      INTEGER NOT NULL DEFAULT 0 CHECK (verified IN (0, 1)),
    PRIMARY KEY (era, state, unit)
);"""


def _crosswalk_ddl(name: str) -> str:
    return f"""
-- Category crosswalk: one harmonised category per source column (2015+ detail).
-- core_category is the same name when it is one of the seven that exist in every
-- era (the 2001-2022 series), else NULL. The CHECK makes that a rule, not a habit.
-- 'excluded' = a column that must not be counted: a part of a total listed beside it,
-- a grand total, or a head that is not a crime against women.
CREATE TABLE IF NOT EXISTS {name} (
    era           TEXT NOT NULL REFERENCES eras(era),
    source_column TEXT NOT NULL,
    category      TEXT NOT NULL CHECK (category IN {_crosswalk_category_check}),
    core_category TEXT CHECK (core_category IS NULL OR (core_category = category AND core_category IN {_category_check})),
    reason        TEXT NOT NULL,
    proposed_by   TEXT NOT NULL,                   -- 'llm:<model>' | 'human' | 'definition'
    verified      INTEGER NOT NULL DEFAULT 0 CHECK (verified IN (0, 1)),
    PRIMARY KEY (era, source_column)
);"""


def _panel_ddl(name: str) -> str:
    return f"""
-- The observed panel: one row per unit and year, summed over the unit's valid/escalated records.
-- Derived: rebuilt by agents.panel on every build, never edited. unit_id = 'd<code>' or 's<state>u'.
CREATE TABLE IF NOT EXISTS {name} (
    unit_id       TEXT NOT NULL,
    state_code    INTEGER NOT NULL REFERENCES states(state_code),
    district_code INTEGER REFERENCES districts(district_code),      -- NULL = the state's unallocated bucket
    year          INTEGER NOT NULL,
    era           TEXT NOT NULL REFERENCES eras(era),
    {_count_columns},
    total         INTEGER NOT NULL CHECK (total >= 0),
    n_records     INTEGER NOT NULL CHECK (n_records >= 1),
    PRIMARY KEY (unit_id, year)
);"""

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

{_districts_ddl('districts')}

-- Police reporting units ("registration circles") under each district, 2016+.
-- One district has many circles. Kept apart from name aliases on purpose.
CREATE TABLE IF NOT EXISTS circles (
    district_code   INTEGER NOT NULL REFERENCES districts(district_code),
    circle_norm     TEXT NOT NULL,                 -- normalised, what the resolver matches on
    circle_name     TEXT NOT NULL,                 -- latest spelling, for display
    first_seen_year INTEGER NOT NULL,
    last_seen_year  INTEGER NOT NULL,
    PRIMARY KEY (district_code, circle_norm)
);
CREATE INDEX IF NOT EXISTS ix_circles_norm ON circles(circle_norm);

-- Boundary changes over time; see _lineage_ddl for the kinds.
{_lineage_ddl('district_lineage')}
{ERAS_DDL}
{_raw_rows_ddl('raw_rows')}
{_records_ddl('records')}
CREATE INDEX IF NOT EXISTS ix_records_district_year ON records(district_code, year);
CREATE INDEX IF NOT EXISTS ix_records_era_status ON records(era, validation_status);
{EXTENDED_DDL}
{_panel_ddl('panel_observed')}
{_unit_map_ddl('unit_map')}
CREATE INDEX IF NOT EXISTS ix_unit_map_district ON unit_map(district_code);
{_crosswalk_ddl('crosswalk')}

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

TABLES = ("meta", "eras", "states", "districts", "circles", "district_lineage", "raw_rows", "records", "extended_counts",
          "panel_observed", "unit_map", "crosswalk", "llm_calls", "llm_attempts", "escalations")


# ---- migrations ---------------------------------------------------------------------------------------------
# One function per version step. A step runs inside one transaction and leaves every other table, the LLM
# cache and the receipts untouched; ``reset`` is for a broken file, not for a version bump.

def _migrate_5_to_6(conn: sqlite3.Connection) -> None:
    """v5 -> v6: districts may be map-only (seen-years nullable, ``in_files`` flag), district_lineage is keyed
    by kind as well, extended_counts is added. SQLite cannot drop NOT NULL or change a primary key in place,
    so each changed table is rebuilt the way the ALTER TABLE documentation prescribes: foreign keys off,
    copy into a new table, drop the old one, rename, check, all inside one transaction."""
    conn.executescript(f"""
PRAGMA foreign_keys = OFF;
BEGIN;
{_districts_ddl('districts_new')}
INSERT INTO districts_new (district_code, state_code, name, aliases_json, first_seen_year, last_seen_year, in_files)
    SELECT district_code, state_code, name, aliases_json, first_seen_year, last_seen_year, 1 FROM districts;
DROP TABLE districts;
ALTER TABLE districts_new RENAME TO districts;
{_lineage_ddl('district_lineage_new')}
INSERT INTO district_lineage_new (parent_code, child_code, kind, year, confidence, reason, source, verified)
    SELECT parent_code, child_code, kind, year, confidence, reason, source, verified FROM district_lineage;
DROP TABLE district_lineage;
ALTER TABLE district_lineage_new RENAME TO district_lineage;
{EXTENDED_DDL}
UPDATE meta SET value = '6' WHERE key = 'schema_version';
COMMIT;
PRAGMA foreign_keys = ON;
""")
    dangling = conn.execute("PRAGMA foreign_key_check").fetchall()
    if dangling:
        raise RuntimeError(f"migration v5->v6 left {len(dangling)} dangling foreign key(s): {[tuple(d) for d in dangling[:5]]}")


def _migrate_6_to_7(conn: sqlite3.Connection) -> None:
    """v6 -> v7: the era list moves from CHECK constraints typed into five tables to the ``eras`` table those
    tables reference, so a new era (2015) is a row, not a rebuild. Each of the five is rebuilt once, with
    its rows; the crosswalk's category check gains 'excluded'. Same recipe as v5 -> v6."""
    counts = ", ".join(CORE_CATEGORIES)
    rebuilds = [
        ("raw_rows", _raw_rows_ddl, "id, era, source_file, source_row, payload_json"),
        ("records", _records_ddl, "id, raw_row_id, era, state, district, registration_circle, state_code, district_code, "
                                  f"year, {counts}, validation_status"),
        ("unit_map", _unit_map_ddl, "era, state, unit, district_code, kind, stage, confidence, reason, verified"),
        ("crosswalk", _crosswalk_ddl, "era, source_column, category, core_category, reason, proposed_by, verified"),
        ("panel_observed", _panel_ddl, f"unit_id, state_code, district_code, year, era, {counts}, total, n_records"),
    ]
    script = ["PRAGMA foreign_keys = OFF;", "BEGIN;", ERAS_DDL, _seed_eras_sql()]
    for name, ddl, columns in rebuilds:
        script += [ddl(f"{name}_new"), f"INSERT INTO {name}_new ({columns}) SELECT {columns} FROM {name};",
                   f"DROP TABLE {name};", f"ALTER TABLE {name}_new RENAME TO {name};"]
    script += ["UPDATE meta SET value = '7' WHERE key = 'schema_version';", "COMMIT;", "PRAGMA foreign_keys = ON;"]
    conn.executescript("\n".join(script))
    conn.executescript(DDL)                                   # the indexes went with the dropped tables; recreate
    dangling = conn.execute("PRAGMA foreign_key_check").fetchall()
    if dangling:
        raise RuntimeError(f"migration v6->v7 left {len(dangling)} dangling foreign key(s): {[tuple(d) for d in dangling[:5]]}")


MIGRATIONS = {5: _migrate_5_to_6, 6: _migrate_6_to_7}


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
    if found > SCHEMA_VERSION:
        conn.close()
        raise RuntimeError(f"{p} was built with schema v{found} but this code expects v{SCHEMA_VERSION}: update the code.")
    for version in range(found, SCHEMA_VERSION):          # bring an older file forward one step at a time
        step = MIGRATIONS.get(version)
        if step is None:
            conn.close()
            raise RuntimeError(
                f"{p} was built with schema v{found} and there is no migration from v{version}. "
                "The store is a rebuildable workbench: run `python -m agents.store reset --yes`."
            )
        step(conn)
    conn.execute(_seed_eras_sql())                            # new eras in ERA_ROWS become rows; existing rows untouched
    conn.commit()
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


# ---- registry + escalation helpers (consumers: data_ingestion/registry.py, crosswalk.py) ----

def replace_registry(conn: sqlite3.Connection, states: list[tuple], districts: list[tuple], circles: list[tuple],
                     map_only: list[tuple] = ()) -> None:
    """Rewrite the canonical geography wholesale. The registry is *derived* from
    the 2016+ files, so it is rebuilt, never edited by hand. If records or
    mappings already point at a district that would vanish, the foreign keys
    make this fail instead of leaving dangling references.
    ``map_only``: (district_code, state_code, name) for districts on the LGD map
    that appear in no data file; stored with ``in_files = 0`` and no seen-years."""
    with conn:                                     # one transaction: all or nothing
        conn.execute("DELETE FROM circles")
        conn.execute("DELETE FROM districts")
        conn.execute("DELETE FROM states")
        conn.executemany("INSERT INTO states (state_code, name, aliases_json) VALUES (?, ?, ?)", states)
        conn.executemany("INSERT INTO districts (district_code, state_code, name, aliases_json, first_seen_year, "
                         "last_seen_year) VALUES (?, ?, ?, ?, ?, ?)", districts)
        conn.executemany("INSERT INTO districts (district_code, state_code, name, aliases_json, first_seen_year, "
                         "last_seen_year, in_files) VALUES (?, ?, ?, '[]', NULL, NULL, 0)", map_only)
        conn.executemany("INSERT INTO circles (district_code, circle_norm, circle_name, first_seen_year, "
                         "last_seen_year) VALUES (?, ?, ?, ?, ?)", circles)


def add_escalation(conn: sqlite3.Connection, *, kind: str, ref: dict, violations: list[dict]) -> bool:
    """Queue something for a human. Returns False when an identical item is
    already open, so re-running a build never duplicates the queue."""
    ref_json = json.dumps(ref, sort_keys=True)
    if conn.execute("SELECT 1 FROM escalations WHERE kind = ? AND ref_json = ? AND status = 'open'",
                    (kind, ref_json)).fetchone():
        return False
    conn.execute("INSERT INTO escalations (kind, ref_json, violations_json) VALUES (?, ?, ?)",
                 (kind, ref_json, json.dumps(violations)))
    conn.commit()
    return True


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
