"""The schema is the contract. These tests pin the constraints other modules rely on."""
import sqlite3

import pytest

from agents.schemas import CORE_CATEGORIES
from agents.store import TABLES, connect, get_cached_call, put_call, reset, table_counts

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
