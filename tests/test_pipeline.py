"""Loaders, validator, panel and orchestrator on fixture files: raw rows are copied once and never edited,
records come only from verified decisions, every rule leaves a visible mark, and a rebuild is a no-op."""
import csv
import json

import openpyxl
import pandas as pd
import pytest

from agents import lineage, loaders, orchestrator, panel, rebase, resolver, store, validator
from data_ingestion import crosswalk

ERA_A_HEADER = ["", "STATE/UT", "DISTRICT", "Year", *crosswalk.ERA_A_DEFINITIONS]
REGISTRY = {"states": [{"state_code": 28, "name": "Andhra Pradesh", "aliases": ["andhra pradesh"]}], "districts": []}


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


def unit_row(unit, code, kind, stage, verified, state="andhra pradesh", era="2001_2014"):
    return {"era": era, "state": state, "state_code": "28", "unit": unit, "first_year": "2001", "last_year": "2014",
            "n_years": "14", "district_code": code, "district_name": "", "kind": kind, "stage": stage, "confidence": "1",
            "reason": "fixture", "candidates": "", "geo_code": "", "geo_share": "", "proposed_by": "rule",
            "verified": verified, "reviewer_note": ""}


def xwalk_row(era, col, category, verified="1"):
    return {"era": era, "source_column": col, "category": category, "core_category": "", "reason": "fixture", "ambiguous": "0",
            "evidence": "", "proposed_by": "human", "verified": verified, "reviewer_note": ""}


@pytest.fixture
def world(tmp_path):
    """Three tiny raw files, the four decision CSVs and a store with the geography loaded."""
    raw = {
        "2001_2014": write_csv(tmp_path / "a.csv", ERA_A_HEADER, [
            [0, "ANDHRA PRADESH", "ANANTAPUR", 2001, 10, 5, 1, 20, 3, 30, 0],
            [1, "ANDHRA PRADESH", "ANANTAPUR", 2002, 11, 5, 1, 21, 3, 31, 0],
            [2, "ANDHRA PRADESH", "ANANTAPUR", 2003, 12, 5, 1, 22, 3, 32, 0],
            [3, "ANDHRA PRADESH", "ANANTAPUR", 2004, 500, 5, 1, 23, 3, 33, 0],       # a spike: 500 vs a median of 12
            [4, "ANDHRA PRADESH", "ANANTAPUR", 2005, 12, 5, 1, 22, 3, 32, 0],
            [5, "ANDHRA PRADESH", "TOTAL", 2001, 999, 9, 9, 9, 9, 9, 9],              # a state sum, not a unit
            [6, "ANDHRA PRADESH", "RAILWAYS", 2001, 1, 0, 0, 0, 0, 0, 0],
            [7, "ANDHRA PRADESH", "RAILWAYS", 2001, 0, 0, 0, 0, 0, 0, 0],             # duplicate unit-year
            [8, "ANDHRA PRADESH", "MYSTERY", 2001, 1, 1, 1, 1, 1, 1, 1],              # no verified mapping
        ]),
        "2016": write_csv(tmp_path / "b.csv", ["id", "year", "state_name", "state_code", "district_name", "district_code",
                                               "registration_circles", "rape", "acid_attack"],
                          [[0, 2016, "Andhra Pradesh", 28, "Anantapur", 502, "Anantapur", 7, 2]]),
        "2017_plus": write_csv(tmp_path / "c.csv", ["id", "year", "state_name", "state_code", "district_name", "district_code",
                                                    "registration_circles", "rape_women_above_18", "rape_girls_below_18",
                                                    "attempt_to_commit_rape_girls_below_18", "acid_attack", "kidnapping_for_ransom"], [
            [0, 2017, "Andhra Pradesh", 28, "Anantapur", 502, "Anantapur", 20, 5, 1, 2, 4],
            [1, 2018, "Andhra Pradesh", 28, "Anantapur", 502, "Anantapur", 30, 999, 6, "", ""],   # rotated column, one empty cell
            [2, 2018, "Andhra Pradesh", 28, "Y.S.R.", 504, "Kadapa", 10, 2, 0, 3, ""],            # ransom empty for all of 2018
        ]),
    }
    wb = openpyxl.Workbook()                                                                  # NCRB's 2015 workbook shape
    ws = wb.active
    ws.append(["District-wise Crimes committed against Women during 2015"])
    ws.append(["SL", "District", "YEAR", "Rape", "Rape_Gang Rape", "Attempt to commit Rape"])
    ws.append(["State: Andhra Pradesh"])
    ws.append([1, "Anantapur", 2015, 40, 2, 3])
    ws.append([2, "Guntakal Railway", 2015, 1, 0, 0])
    ws.append([3, "Total District(s)", None, 41, 2, 3])
    ws.append(["Source: Crime in India"])
    wb.save(tmp_path / "d.xlsx")
    raw["2015"] = tmp_path / "d.xlsx"
    csvs = {"crosswalk": tmp_path / "crosswalk.csv", "column_fixes": tmp_path / "fixes.csv",
            "unit_map": tmp_path / "unit_map.csv", "lineage": tmp_path / "lineage.csv"}
    xrows = [xwalk_row("2015", "Rape", "rape"), xwalk_row("2015", "Rape_Gang Rape", "excluded"),
             xwalk_row("2015", "Attempt to commit Rape", "attempt_to_rape"),
             xwalk_row("2016", "rape", "rape"), xwalk_row("2016", "acid_attack", "acid_attack"),
             xwalk_row("2017_plus", "rape_women_above_18", "rape"), xwalk_row("2017_plus", "rape_girls_below_18", "rape"),
             xwalk_row("2017_plus", "attempt_to_commit_rape_girls_below_18", "attempt_to_rape"),
             xwalk_row("2017_plus", "acid_attack", "acid_attack"), xwalk_row("2017_plus", "kidnapping_for_ransom", "kidnapping_abduction")]
    with open(csvs["crosswalk"], "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=crosswalk.CSV_COLUMNS)
        w.writeheader()
        w.writerows(xrows)
    write_csv(csvs["column_fixes"], ["era", "year_from", "year_to", "file_column", "actual_column", "evidence", "verified", "reviewer_note"],
              [["2017_plus", 2018, 2022, "attempt_to_commit_rape_girls_below_18", "rape_girls_below_18", "fixture", 1, ""]])
    resolver.write_rows(csvs["unit_map"], {
        ("2001_2014", "andhra pradesh", "anantapur"): unit_row("anantapur", "502", "district", "exact", "1"),
        ("2001_2014", "andhra pradesh", "railways"): unit_row("railways", "", "railway", "llm", "1"),
        ("2001_2014", "andhra pradesh", "mystery"): unit_row("mystery", "", "unknown", "unresolved", "0"),
        ("2015", "andhra pradesh", "anantapur"): unit_row("anantapur", "502", "district", "exact", "1", era="2015"),
        ("2015", "andhra pradesh", "guntakal railway"): unit_row("guntakal railway", "", "railway", "llm", "1", era="2015")})
    lineage.write_rows(csvs["lineage"], {
        ("502", "504", "split"): {"parent_code": "502", "parent_name": "Anantapur", "child_code": "504", "child_name": "Y.S.R.",
                                  "kind": "split", "year": "2017", "evidence": "", "confidence": "1.00", "reason": "fixture",
                                  "proposed_by": "human", "verified": "1", "reviewer_note": ""},
        ("504", "502", "merge"): {"parent_code": "504", "parent_name": "", "child_code": "502", "child_name": "", "kind": "merge",
                                  "year": "", "evidence": "", "confidence": "", "reason": "", "proposed_by": "llm:x", "verified": "0",
                                  "reviewer_note": ""}})
    conn = store.connect(tmp_path / "t.db")
    store.replace_registry(conn, states=[(28, "Andhra Pradesh", "[]")],
                           districts=[(502, 28, "Anantapur", "[]", 2016, 2022), (504, 28, "Y.S.R.", "[]", 2016, 2022)], circles=[])
    return {"conn": conn, "raw": raw, "csvs": csvs, "tmp": tmp_path}


def run_build(world):
    tmp = world["tmp"]
    return orchestrator.build(world["conn"], registry=False, raw_files=world["raw"], csv_paths=world["csvs"],
                              trace_dir=tmp / "traces", parquet_path=tmp / "panel.parquet", registry_dict=REGISTRY,
                              rebase_paths={"parquet_path": tmp / "rebased.parquet", "units_path": tmp / "stable_units.json",
                                            "reports_dir": tmp / "reports", "area_path": tmp / "no_area.csv"})


def record(conn, **where):
    cond = " AND ".join(f"{k} = ?" for k in where)
    return dict(conn.execute(f"SELECT * FROM records WHERE {cond}", tuple(where.values())).fetchone())


def test_raw_rows_are_copied_once_and_records_rebuilt_from_them(world):
    conn = world["conn"]
    first = loaders.load_raw_rows(conn, "2001_2014", world["raw"]["2001_2014"])
    again = loaders.load_raw_rows(conn, "2001_2014", world["raw"]["2001_2014"])
    assert (first["rows_in_file"], first["inserted"], again["inserted"], again["raw_rows"]) == (9, 9, 0, 9)
    payload = json.loads(conn.execute("SELECT payload_json FROM raw_rows WHERE source_row = 3").fetchone()[0])
    assert payload["DISTRICT"] == "ANANTAPUR" and payload["Rape"] == "500"                 # the file's own strings, untouched


def test_era_a_records_use_the_verified_unit_map_and_skip_state_totals(world):
    conn = world["conn"]
    resolver.apply(conn, csv_path=world["csvs"]["unit_map"])
    loaders.load_raw_rows(conn, "2001_2014", world["raw"]["2001_2014"])
    s = loaders.build_records(conn, "2001_2014", registry=REGISTRY)
    assert (s["records"], s["state_total_rows_skipped"], s["unallocated"], s["unmapped_units"]) == (8, 1, 2, ["andhra pradesh/mystery"])
    r = record(conn, district="ANANTAPUR", year=2004)
    assert (r["state_code"], r["district_code"], r["rape"], r["total"], r["validation_status"]) == (28, 502, 500, 565, "pending")
    assert record(conn, district="RAILWAYS", year=2001)["district_code"] is None               # the unallocated bucket
    assert record(conn, district="MYSTERY", year=2001)["district_code"] is None
    assert conn.execute("SELECT COUNT(*) FROM records WHERE district = 'TOTAL'").fetchone()[0] == 0


def test_model_era_records_apply_fixes_fold_categories_and_flag_an_empty_cell(world):
    conn = world["conn"]
    crosswalk.apply(conn, csv_path=world["csvs"]["crosswalk"])
    loaders.load_raw_rows(conn, "2017_plus", world["raw"]["2017_plus"])
    s = loaders.build_records(conn, "2017_plus", fixes_path=world["csvs"]["column_fixes"])
    assert (s["records"], s["not_reported_cells"], s["records_with_missing_counts"], s["escalations_queued"]) == (3, 2, 1, 1)
    r17 = record(conn, district="Anantapur", year=2017)
    assert (r17["rape"], r17["kidnapping_abduction"], r17["validation_status"]) == (25, 4, "pending")
    r18 = record(conn, district="Anantapur", year=2018)
    assert r18["rape"] == 36                                                                     # 30 + the fixed girls column (6), not 999
    assert r18["validation_status"] == "escalated"                                               # acid_attack was empty while Kadapa's was 3
    ext = {(row[0], row[1]): row[2] for row in conn.execute(
        "SELECT r.year, e.category, e.count FROM extended_counts e JOIN records r ON r.id = e.record_id WHERE r.district = 'Anantapur'")}
    assert ext == {(2017, "attempt_to_rape"): 1, (2017, "acid_attack"): 2, (2018, "attempt_to_rape"): 6}
    assert record(conn, district="Y.S.R.", year=2018)["kidnapping_abduction"] == 0                # not reported that year: nothing, no flag
    esc = json.loads(conn.execute("SELECT violations_json FROM escalations").fetchone()[0])
    assert esc == [{"rule": "missing_count", "field": "acid_attack", "message": "empty cell, stored as 0"}]
    assert loaders.build_records(conn, "2017_plus", fixes="none", fixes_path=world["csvs"]["column_fixes"])["records"] == 3
    assert record(conn, district="Anantapur", year=2018)["rape"] == 1029                         # without fixes the file is taken as it is


def test_a_verified_unit_map_row_for_a_model_era_overrides_the_file_code(world):
    conn = world["conn"]
    crosswalk.apply(conn, csv_path=world["csvs"]["crosswalk"])
    conn.execute("INSERT INTO unit_map (era, state, unit, district_code, kind, stage, confidence, reason, verified) "
                 "VALUES ('2017_plus', 'andhra pradesh', 'kadapa', 502, 'circle', 'human', 1, 'fixture', 1)")
    loaders.load_raw_rows(conn, "2017_plus", world["raw"]["2017_plus"])
    s = loaders.build_records(conn, "2017_plus", fixes_path=world["csvs"]["column_fixes"])
    assert s["overrides_applied"] == 1
    assert record(conn, district="Y.S.R.", year=2018)["district_code"] == 502


def test_the_2015_workbook_counts_totals_not_parts_and_resolves_units_through_the_unit_map(world):
    conn = world["conn"]
    crosswalk.apply(conn, csv_path=world["csvs"]["crosswalk"])
    resolver.apply(conn, csv_path=world["csvs"]["unit_map"])
    loaders.load_raw_rows(conn, "2015", world["raw"]["2015"])
    s = loaders.build_records(conn, "2015", registry=REGISTRY)
    assert (s["records"], s["unallocated"], s["unmapped_units"], s["extended_rows"]) == (2, 1, [], 2)   # a zero count is still a row
    r = record(conn, district="Anantapur", year=2015)
    assert (r["rape"], r["district_code"], r["state_code"], r["registration_circle"]) == (40, 502, 28, None)  # the gang-rape part is excluded: 40, not 42
    assert record(conn, district="Guntakal Railway", year=2015)["district_code"] is None
    raw = json.loads(conn.execute("SELECT payload_json FROM raw_rows WHERE era = '2015' AND source_row = 4").fetchone()[0])
    assert (raw["state_name"], raw["district"], raw["Rape"]) == ("Andhra Pradesh", "Anantapur", "40")
    ext = conn.execute("SELECT r.district, e.category, e.count FROM extended_counts e JOIN records r ON r.id = e.record_id "
                       "WHERE r.era = '2015' ORDER BY r.id").fetchall()
    assert [tuple(x) for x in ext] == [("Anantapur", "attempt_to_rape", 3), ("Guntakal Railway", "attempt_to_rape", 0)]


def test_validator_marks_every_rule_and_never_changes_a_count(world):
    conn = world["conn"]
    resolver.apply(conn, csv_path=world["csvs"]["unit_map"])
    loaders.load_raw_rows(conn, "2001_2014", world["raw"]["2001_2014"])
    loaders.build_records(conn, "2001_2014", registry=REGISTRY)
    s = validator.validate(conn)
    assert (s["records"], s["valid"], s["escalated"], s["pending"]) == (8, 4, 3, 1)
    assert s["violations_by_rule"] == {"unresolved": 1, "duplicate_key": 2, "spike": 1}
    assert record(conn, district="ANANTAPUR", year=2004)["rape"] == 500                           # flagged, not altered
    assert record(conn, district="ANANTAPUR", year=2004)["validation_status"] == "escalated"
    assert record(conn, district="ANANTAPUR", year=2003)["validation_status"] == "valid"
    assert record(conn, district="MYSTERY", year=2001)["validation_status"] == "pending"
    assert {r["validation_status"] for r in conn.execute("SELECT validation_status FROM records WHERE district = 'RAILWAYS'")} == {"escalated"}
    spike = [v for row in conn.execute("SELECT violations_json FROM escalations") for v in json.loads(row[0]) if v["rule"] == "spike"]
    assert spike == [{"rule": "spike", "field": "rape", "message": "500 in 2004 vs a median of 11.5 over the unit's other years"}]
    assert validator.validate(conn)["escalations_queued"] == 0                                    # a second pass queues nothing twice


def test_build_produces_the_panel_the_parquet_file_and_a_trace_and_is_a_no_op_when_rerun(world):
    conn = world["conn"]
    out = run_build(world)
    assert out["panel"] == {"rows": 12, "districts": 2, "unallocated_rows": 2, "years": [2001, 2018], "records_left_out_pending": 1,
                            "parquet": str(world["tmp"] / "panel.parquet")}
    table = {(r["unit_id"], r["year"]): dict(r) for r in conn.execute("SELECT * FROM panel_observed")}
    assert table[("d502", 2004)]["rape"] == 500 and table[("d502", 2004)]["n_records"] == 1
    assert (table[("s28u", 2001)]["rape"], table[("s28u", 2001)]["n_records"]) == (1, 2)          # the two railway rows, summed; MYSTERY left out
    assert table[("d502", 2015)]["rape"] == 40 and table[("s28u", 2015)]["era"] == "2015"           # the workbook year fills the gap
    assert table[("d502", 2018)]["rape"] == 36 and table[("d504", 2018)]["era"] == "2017_plus"
    assert len(pd.read_parquet(world["tmp"] / "panel.parquet")) == 12
    steps = [json.loads(l)["step"] for l in open(out["trace"], encoding="utf-8")]
    assert steps == ["inputs", "wipe_derived", "crosswalk_apply", "unit_map_apply", "lineage_apply", "raw_rows:2001_2014", "raw_rows:2015",
                     "raw_rows:2016", "raw_rows:2017_plus", "records:2001_2014", "records:2015", "records:2016", "records:2017_plus",
                     "validate", "panel", "rebase"]
    first = json.loads(open(out["trace"], encoding="utf-8").readline())
    assert set(first["result"]["files"]) == {"raw:2001_2014", "raw:2015", "raw:2016", "raw:2017_plus", "csv:crosswalk", "csv:column_fixes",
                                             "csv:unit_map", "csv:lineage", "csv:split_weights"}   # the repo's split weights: its rows name no fixture edge
    assert (world["tmp"] / "traces" / "latest.jsonl").read_text(encoding="utf-8") == open(out["trace"], encoding="utf-8").read()
    escalations = conn.execute("SELECT COUNT(*) FROM escalations").fetchone()[0]

    again = run_build(world)
    lines = [json.loads(l) for l in open(again["trace"], encoding="utf-8")]
    assert all(l["result"]["inserted"] == 0 for l in lines if l["step"].startswith("raw_rows"))  # nothing new to copy
    assert conn.execute("SELECT COUNT(*) FROM escalations").fetchone()[0] == escalations           # nothing queued twice
    assert conn.execute("SELECT COUNT(*) FROM panel_observed").fetchone()[0] == 12
    assert conn.execute("SELECT COUNT(*) FROM panel_rebased").fetchone()[0] == 19
    assert conn.execute("SELECT COUNT(*) FROM district_lineage").fetchone()[0] == 1


def test_build_rebases_the_panel_on_the_verified_lineage_and_marks_every_estimate(world):
    out = run_build(world)
    conn = world["conn"]
    # Y.S.R. (504) was carved out of Anantapur (502) in 2017: its 2001-05, 2015 and 2016 are inside Anantapur's rows.
    obs = {(r["unit_id"], r["year"]): dict(r) for r in conn.execute("SELECT * FROM panel_observed")}
    reb = {(r["unit_id"], r["year"]): dict(r) for r in conn.execute("SELECT * FROM panel_rebased")}
    share = obs[("d504", 2018)]["total"] / (obs[("d502", 2018)]["total"] + obs[("d504", 2018)]["total"])   # one separate year
    grouped = share >= rebase.GROUP_THRESHOLD
    assert out["rebased"]["rows"] == 19 and out["rebased"]["district_rows"] == 17                 # 12 observed + 7 estimated
    assert out["rebased"] == {**out["rebased"], "observed": 3, "adjusted": 7, "estimated": 7, "basis": {"crime:1y": 7},
                              "missing_district_years": 1, "edges": 1, "edges_no_evidence": 0, "map_only": 0,
                              "groups": 1 if grouped else 2, "largest_group": 2 if grouped else 1, "grouping_edges": int(grouped)}
    assert reb[("d504", 2001)]["rape"] == pytest.approx(share * obs[("d502", 2001)]["rape"])
    assert reb[("d502", 2001)]["rape"] == pytest.approx((1 - share) * obs[("d502", 2001)]["rape"])
    assert (reb[("d504", 2001)]["is_estimated"], reb[("d504", 2001)]["share_basis"], reb[("d504", 2001)]["holder_code"]) == (1, "crime:1y", 502)
    assert (reb[("d502", 2001)]["is_estimated"], reb[("d502", 2001)]["share_basis"]) == (1, "remainder")
    assert reb[("d502", 2018)]["share_basis"] == "observed" and ("d504", 2017) not in reb               # 2017: not filed, not pooled
    assert reb[("s28u", 2001)] == {**reb[("s28u", 2001)], "total": obs[("s28u", 2001)]["total"], "is_estimated": 0}
    assert len(pd.read_parquet(world["tmp"] / "rebased.parquet")) == 19
    units = json.loads((world["tmp"] / "stable_units.json").read_text(encoding="utf-8"))
    assert units["meta"]["edges"][0]["share"] == pytest.approx(share, abs=1e-4) and units["meta"]["edges"][0]["grouped"] == grouped
    assert {p.name.split("_", 1)[1] for p in (world["tmp"] / "reports").iterdir()} == {"rebasing.md", "stable_units.md"}
