"""The schema is the contract. These tests pin the constraints other modules rely on."""
import sqlite3

import pytest

from agents.schemas import CORE_CATEGORIES
from agents.store import DDL, TABLES, _districts_ddl, connect, get_cached_call, put_call, reset, table_counts

COUNTS = {c: 1 for c in CORE_CATEGORIES}


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "t.db")
    c.execute("INSERT INTO states (state_code, name) VALUES (28, 'andhra pradesh')")
    c.execute("INSERT INTO districts (district_code, state_code, name, first_seen_year, last_seen_year) "
              "VALUES (502, 28, 'ananthapuramu', 2016, 2022)")
    c.execute("INSERT INTO raw_rows (id, era, source_file, source_row, payload_json) VALUES (1, '2016', 'f.csv', 0, '{}')")
    c.commit()
    return c


def insert_record(conn, raw_row_id=1, district_code=502, **overrides):
    counts = {**COUNTS, **overrides}
    cols = ", ".join(counts)
    conn.execute(
        f"INSERT INTO records (raw_row_id, era, state, district, year, district_code, {cols}) "
        f"VALUES (?, '2016', 'andhra pradesh', 'ananthapuramu', 2016, ?, {', '.join('?' * len(counts))})",
        (raw_row_id, district_code, *counts.values()),
    )


def test_all_tables_exist_and_start_empty(tmp_path):
    counts = table_counts(connect(tmp_path / "t.db"))
    assert set(counts) == set(TABLES)
    assert counts["meta"] == 1 and all(v == 0 for t, v in counts.items() if t != "meta")


def test_total_is_generated_from_the_parts(conn):
    insert_record(conn, rape=3, dowry_deaths=4)
    assert conn.execute("SELECT total FROM records").fetchone()[0] == 3 + 4 + 5


def test_negative_count_is_rejected(conn):
    with pytest.raises(sqlite3.IntegrityError):
        insert_record(conn, rape=-1)


def test_unknown_district_code_is_rejected(conn):
    with pytest.raises(sqlite3.IntegrityError):
        insert_record(conn, district_code=999)


def test_unresolved_record_is_allowed(conn):
    insert_record(conn, district_code=None)
    assert conn.execute("SELECT district_code FROM records").fetchone()[0] is None


def test_record_must_point_at_an_existing_raw_row(conn):
    with pytest.raises(sqlite3.IntegrityError):
        insert_record(conn, raw_row_id=42)


def test_unit_map_stage_is_an_enum(conn):
    conn.execute("INSERT INTO unit_map (era, state, unit, district_code, stage, confidence) "
                 "VALUES ('2001_2014', 'andhra pradesh', 'anantapur', 502, 'fuzzy', 0.93)")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO unit_map (era, state, unit, stage) VALUES ('2001_2014', 'x', 'y', 'guess')")


def test_llm_call_cache_roundtrip(conn):
    assert get_cached_call(conn, "k1") is None
    put_call(conn, cache_key="k1", run_id="r", tag="t", provider="gemini", model="m", temperature=0.0,
             prompt_hash="p", schema_hash="s", prompt_tokens=10, completion_tokens=5, latency_ms=120,
             attempts=1, response_json='{"ok": true}')
    row = get_cached_call(conn, "k1")
    assert row["provider"] == "gemini" and row["response_json"] == '{"ok": true}' and row["created_at"]


def test_reset_gives_a_fresh_file(conn, tmp_path):
    insert_record(conn)
    conn.commit()
    conn.close()                       # reset requires no open connections (Windows enforces it)
    fresh = reset(tmp_path / "t.db")
    assert table_counts(fresh)["records"] == 0


def test_store_built_by_older_code_is_refused_with_the_fix_in_the_message(tmp_path):
    old = connect(tmp_path / "old.db")
    old.execute("UPDATE meta SET value = '1' WHERE key = 'schema_version'")
    old.commit()
    old.close()
    with pytest.raises(RuntimeError, match="reset --yes"):
        connect(tmp_path / "old.db")


def test_receipts_are_counted_per_provider_for_the_last_day(conn):
    from agents.store import add_attempt, usage_last_24h
    add_attempt(conn, run_id="r", tag="t", cache_key="k", provider="gemini", model="m", outcome="ok",
                prompt_tokens=100, completion_tokens=20)
    add_attempt(conn, run_id="r", tag="t", cache_key="k", provider="gemini", model="m", outcome="transient_error")
    add_attempt(conn, run_id="r", tag="t", cache_key="k", provider="groq", model="g", outcome="ok", prompt_tokens=7)
    assert usage_last_24h(conn, "gemini", "m") == (2, 120)
    assert usage_last_24h(conn, "groq", "g") == (1, 7)


def test_circles_belong_to_a_real_district_and_new_enum_values_are_accepted(conn):
    conn.execute("INSERT INTO circles (district_code, circle_norm, circle_name, first_seen_year, last_seen_year) "
                 "VALUES (502, 'guntakal railway', 'Guntakal Railway', 2016, 2022)")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO circles (district_code, circle_norm, circle_name, first_seen_year, last_seen_year) "
                     "VALUES (999, 'x', 'X', 2016, 2016)")
    conn.execute("INSERT INTO unit_map (era, state, unit, district_code, stage, confidence) "
                 "VALUES ('2001_2014', 'andhra pradesh', 'guntakal rly', 502, 'circle', 1.0)")
    conn.execute("INSERT INTO states (state_code, name) VALUES (8, 'rajasthan')")
    conn.execute("INSERT INTO districts (district_code, state_code, name, first_seen_year, last_seen_year) VALUES (102, 8, 'jaipur', 2016, 2022)")
    conn.execute("INSERT INTO district_lineage (parent_code, child_code, kind, source) VALUES (502, 102, 'recode', 'human')")


def test_an_identical_open_escalation_is_queued_only_once(conn):
    from agents.store import add_escalation
    first = add_escalation(conn, kind="unit", ref={"district_code": 783}, violations=[{"rule": "code_disappeared"}])
    second = add_escalation(conn, kind="unit", ref={"district_code": 783}, violations=[{"rule": "code_disappeared"}])
    assert (first, second) == (True, False)
    assert conn.execute("SELECT COUNT(*) FROM escalations").fetchone()[0] == 1


def test_crosswalk_core_is_a_rule_not_a_habit(conn):
    conn.execute("INSERT INTO crosswalk (era, source_column, category, core_category, reason, proposed_by) "
                 "VALUES ('2016', 'rape', 'rape', 'rape', 'r', 'human')")
    conn.execute("INSERT INTO crosswalk (era, source_column, category, core_category, reason, proposed_by) "
                 "VALUES ('2016', 'acid_attack', 'acid_attack', NULL, 'r', 'human')")
    with pytest.raises(sqlite3.IntegrityError):                       # core must equal category ...
        conn.execute("INSERT INTO crosswalk (era, source_column, category, core_category, reason, proposed_by) "
                     "VALUES ('2016', 'a', 'acid_attack', 'rape', 'r', 'human')")
    with pytest.raises(sqlite3.IntegrityError):                       # ... and be one of the seven
        conn.execute("INSERT INTO crosswalk (era, source_column, category, core_category, reason, proposed_by) "
                     "VALUES ('2016', 'b', 'acid_attack', 'acid_attack', 'r', 'human')")
    with pytest.raises(sqlite3.IntegrityError):                       # unknown category
        conn.execute("INSERT INTO crosswalk (era, source_column, category, core_category, reason, proposed_by) "
                     "VALUES ('2016', 'c', 'murder', NULL, 'r', 'human')")


V5_DISTRICTS = """
CREATE TABLE IF NOT EXISTS districts (
    district_code   INTEGER PRIMARY KEY,
    state_code      INTEGER NOT NULL REFERENCES states(state_code),
    name            TEXT NOT NULL,
    aliases_json    TEXT NOT NULL DEFAULT '[]',
    first_seen_year INTEGER NOT NULL,
    last_seen_year  INTEGER NOT NULL
);"""


def test_a_v5_store_is_migrated_in_place_with_its_cache_mappings_and_edges(tmp_path):
    path = tmp_path / "v5.db"
    old = sqlite3.connect(path)
    old.executescript(DDL.replace(_districts_ddl("districts"), V5_DISTRICTS))     # v5 shape of the table that changed
    old.execute("INSERT INTO meta (key, value) VALUES ('schema_version', '5')")
    old.execute("INSERT INTO states (state_code, name) VALUES (28, 'andhra pradesh')")
    old.execute("INSERT INTO districts (district_code, state_code, name, first_seen_year, last_seen_year) "
                "VALUES (502, 28, 'anantapur', 2016, 2022)")
    old.execute("INSERT INTO unit_map (era, state, unit, district_code, kind, stage, confidence, reason, verified) "
                "VALUES ('2001_2014', 'andhra pradesh', 'anantapur', 502, 'district', 'exact', 1.0, 'r', 1)")
    old.execute("INSERT INTO district_lineage (parent_code, child_code, kind, year, source, verified) "
                "VALUES (502, 502, 'rename', 2017, 'human', 1)")
    put_call(old, cache_key="k", run_id="r", tag="t", provider="gemini", model="m", temperature=0.0, prompt_hash="p",
             schema_hash="s", prompt_tokens=1, completion_tokens=1, latency_ms=1, attempts=1, response_json="{}")
    old.commit()
    old.close()

    conn = connect(path)                                                         # migrates 5 -> 6 on open
    assert conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0] == "6"
    assert "in_files" in {r[1] for r in conn.execute("PRAGMA table_info(districts)")}
    assert tuple(conn.execute("SELECT district_code, in_files FROM districts").fetchone()) == (502, 1)
    assert conn.execute("SELECT district_code FROM unit_map WHERE unit = 'anantapur'").fetchone()[0] == 502
    assert conn.execute("SELECT COUNT(*) FROM district_lineage").fetchone()[0] == 1
    assert get_cached_call(conn, "k")["provider"] == "gemini"                    # the cache survived
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    conn.execute("INSERT INTO districts (district_code, state_code, name, in_files) VALUES (526, 28, 'bengaluru rural', 0)")
    with pytest.raises(sqlite3.IntegrityError):                                  # the rebuilt table still binds
        conn.execute("INSERT INTO unit_map (era, state, unit, district_code, kind, stage, confidence, reason, verified) "
                     "VALUES ('2001_2014', 'x', 'y', 999, 'district', 'exact', 1.0, 'r', 1)")
    conn.close()
