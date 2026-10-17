"""The eval harness on hand-built inputs: precision by stage, the reliability bins and their intervals, the
notebook's coverage rule, the threshold sweep, the fingerprint, and the report renderer."""
import json

import pandas as pd
import pytest

from evals import metrics, run_evals


def gold(rows):
    return pd.DataFrame(rows, columns=["proposed_stage", "proposed_confidence", "corrected"])


def test_resolution_precision_scores_only_the_automatic_stages():
    g = gold([("llm", "0.9", 0), ("llm", "0.9", 1), ("fuzzy", "0.92", 0), ("human", "", 0), ("unresolved", "", 0)])
    r = metrics.resolution(g)
    assert r["llm"]["answered"] == 2 and r["llm"]["precision"] == 0.5 and r["fuzzy"]["precision"] == 1.0
    assert r["all_automatic"]["answered"] == 3 and r["unresolved"] == 1 and r["human_only"] == 1 and r["gold_rows"] == 5


def test_calibration_bins_carry_wilson_intervals():
    g = gold([("llm", "0.95", 0)] * 9 + [("llm", "0.95", 1)] + [("llm", "0.7", 1), ("llm", "0.7", 0)])
    rows = {b["bin"]: b for b in metrics.calibration(g)}
    hi = rows["[0.95, 0.99)"]
    assert hi["answered"] == 10 and hi["precision"] == 0.9 and hi["ci95"][0] < 0.9 < hi["ci95"][1]
    assert rows["[0.00, 0.80)"]["answered"] == 2 and rows["[0.99, 1.00]"]["answered"] == 0 and rows["[0.99, 1.00]"]["precision"] is None
    assert metrics.wilson(0, 0) == (0.0, 0.0) and metrics.wilson(5, 5)[1] == 1.0 and metrics.wilson(5, 5)[0] < 1.0


def test_the_notebook_rule_is_name_equality(tmp_path):
    from agents import resolver
    rows = {("2001_2014", "s", "anantapur"): {"era": "2001_2014", "state": "s", "unit": "anantapur", "district_code": "502", "n_years": "14"},
            ("2001_2014", "s", "kadapa"): {"era": "2001_2014", "state": "s", "unit": "kadapa", "district_code": "", "n_years": "6"},
            ("2015", "s", "ysr"): {"era": "2015", "state": "s", "unit": "ysr"}}
    path = tmp_path / "um.csv"
    resolver.write_rows(path, {k: {c: v.get(c, "") for c in resolver.CSV_COLUMNS} for k, v in rows.items()})
    registry = {"districts": [{"name": "Anantapur", "aliases": ["anantapur"]}, {"name": "Y.S.R.", "aliases": ["ysr", "cuddapah"]}]}
    out = metrics.notebook_coverage(path, registry)
    assert (out["units"], out["matched"], out["share"]) == (2, 1, 0.5)
    assert (out["pipeline_share"], out["no_district_units"], out["rows"], out["rows_share"], out["pipeline_rows_share"]) == (0.5, 1, 20, 0.7, 0.7)


def test_threshold_sweep_regroups_from_the_recorded_shares(tmp_path):
    su = {"meta": {"edges": [{"parent_code": 1, "child_code": 2, "kind": "split", "year": 2005, "share": 0.15, "grouped": False},
                             {"parent_code": 2, "child_code": 3, "kind": "split", "year": 2010, "share": 0.25, "grouped": True},
                             {"parent_code": 4, "child_code": 1, "kind": "reported_together", "year": 2014, "share": None, "grouped": True}]},
          "units": [{"members": [1, 4]}, {"members": [2, 3]}]}
    path = tmp_path / "su.json"
    path.write_text(json.dumps(su), encoding="utf-8")
    out = metrics.rebasing_sensitivity(path)
    assert out["0.1"] == {"grouping_edges": 3, "groups": 1, "largest": 4}
    assert out["0.2"] == {"grouping_edges": 2, "groups": 2, "largest": 2}
    assert out["0.3"] == {"grouping_edges": 1, "groups": 3, "largest": 2}


def test_prompt_fingerprint_and_versions_name_every_prompt_file():
    from agents import prompts
    a, b = metrics.prompt_fingerprint(), metrics.prompt_fingerprint()
    assert a == b and set(a) == set(prompts.VERSIONS) and all(len(v) == 12 for v in a.values())
    assert metrics.prompt_versions() == prompts.VERSIONS and metrics.prompt_versions() is not prompts.VERSIONS


def test_the_report_renders_and_the_history_row_has_every_column(tmp_path):
    m = {"date": "2026-10-10", "gold_districts": "evals/gold/districts.csv",
         "coverage": {"all": {"records": 10, "with_district": 9, "unallocated": 1, "share": 0.9}},
         "notebook_coverage": {"units": 4, "matched": 3, "share": 0.75, "pipeline_units": 4, "pipeline_share": 1.0, "no_district_units": 0,
                               "rows": 40, "rows_share": 0.8, "pipeline_rows_share": 1.0, "spec_figure": 0.727},
         "resolution": {"llm": {"answered": 2, "correct": 1, "precision": 0.5, "ci95": [0.1, 0.9]},
                        "all_automatic": {"answered": 2, "correct": 1, "precision": 0.5, "ci95": [0.1, 0.9]},
                        "unresolved": 0, "human_only": 1, "gold_rows": 3},
         "calibration_llm": [{"bin": "[0.95, 0.99)", "answered": 1, "correct": 1, "precision": 1.0, "ci95": [0.2, 1.0], "mean_confidence": 0.95}],
         "calibration_fuzzy": [], "hallucination": {"records": 10, "records_without_source_row": 0, "resolver_answers": 2,
                                                    "resolver_codes_outside_candidates": 0, "resolver_rate": 0.0, "lineage_calls": 1,
                                                    "lineage_proposals": 3, "lineage_codes_outside_state": 1, "lineage_rate": 1 / 3},
         "escalation": {"2001_2014": {"units": 4, "needed_review": 1, "rate": 0.25}, "records": {"records": 10, "flagged": 1, "rate": 0.1},
                        "open_escalations": {"record": 1}},
         "operations": {"resolve": {"calls": 2, "models": ["m"], "latency_ms_p50": 100.0, "latency_ms_p95": 200.0, "prompt_tokens_mean": 50.0,
                                    "completion_tokens_mean": 5.0, "answers": 4, "usd_total": 0.001, "inr_total": 0.088, "inr_per_1000_answers": 22.0},
                        "pricing": {"rates_usd_per_m": {}, "as_of": "2026-10-10", "usd_inr": 88.0}},
         "rebasing_sensitivity": {"0.2": {"grouping_edges": 1, "groups": 2, "largest": 2}}, "prompts": {"resolver": "abc"}}
    text = run_evals.render(m)
    assert "| 2001_2014 | 4 | 1 | 25.0% |" in text and "**0** of 10" in text
    md, js = run_evals.write(m, tmp_path / "reports", tmp_path / "history.csv")
    assert md.exists() and json.loads(js.read_text(encoding="utf-8"))["date"] == "2026-10-10"
    hist = pd.read_csv(tmp_path / "history.csv")
    assert list(hist.columns) == run_evals.HISTORY_COLUMNS and float(hist.loc[0, "coverage_all"]) == pytest.approx(0.9)
    run_evals.write(m, tmp_path / "reports", tmp_path / "history.csv")                 # same date again: replaced, not appended
    assert len(pd.read_csv(tmp_path / "history.csv")) == 1


def test_extraction_compares_the_stored_records_with_the_verified_gold(tmp_path):
    from agents import store
    from agents.schemas import CORE_CATEGORIES
    conn = store.connect(tmp_path / "t.db")
    conn.execute("INSERT INTO states (state_code, name) VALUES (28, 'ap')")
    conn.execute("INSERT INTO districts (district_code, state_code, name, first_seen_year, last_seen_year, in_files) "
                 "VALUES (502, 28, 'a', 2016, 2022, 1)")
    counts = ", ".join(CORE_CATEGORIES)
    for raw_id, (row_no, first) in enumerate(((0, 5), (1, 7)), start=1):
        conn.execute("INSERT INTO raw_rows (id, era, source_file, source_row, payload_json) VALUES (?, '2016', 'f.csv', ?, '{}')",
                     (raw_id, row_no))
        conn.execute(f"INSERT INTO records (raw_row_id, era, state, district, year, district_code, {counts}) "
                     f"VALUES (?, '2016', 'ap', 'a', 2016, 502, {first}, {', '.join('1' for _ in CORE_CATEGORIES[1:])})", (raw_id,))
    conn.commit()
    header = ["era", "source_file", "source_row", *CORE_CATEGORIES, "verified", "checked_by"]
    rows = [["2016", "f.csv", 0, 5, *[1] * 6, 1, "script"],                # all seven equal
            ["2016", "f.csv", 1, 6, *[1] * 6, 1, "script"],                # gold 6, record 7
            ["2016", "f.csv", 2, 1, *[1] * 6, 1, "script, hand"],          # its record no longer exists
            ["2016", "f.csv", 3, 1, *[1] * 6, 0, "script"]]                # not verified: ignored
    path = tmp_path / "records.csv"
    pd.DataFrame(rows, columns=header).to_csv(path, index=False)
    out = metrics.extraction(conn, path)
    assert out["all"]["gold_rows"] == 3 and out["all"]["missing"] == 1 and out["all"]["counts_compared"] == 21
    assert out["all"]["field_exact_match"] == pytest.approx(13 / 21) and out["all"]["record_exact_match"] == pytest.approx(1 / 3)
    assert out["checked_by"] == {"script": 3, "script, hand": 1} and list(out["by_era"]) == ["2016"]


def scored(rows):
    """(stage, score, kind, right, unit) -> a gold frame with the columns the acceptance metric reads."""
    return pd.DataFrame([{"era": "2001_2014", "state": "s", "unit": unit, "proposed_stage": stage, "proposed_confidence": str(score),
                          "proposed_kind": kind, "proposed_code": 1, "gold_code": 1 if right else 2, "gold_name": "G",
                          "corrected": 0 if right else 1} for stage, score, kind, right, unit in rows])


def test_acceptance_picks_the_lowest_cut_that_passes_the_rule_and_checks_itself_by_dropping_one_answer():
    rows = ([("llm", 0.95, "district", True, f"r{i}") for i in range(25)]
            + [("llm", 0.9, "district", True, "ok"), ("llm", 0.9, "district", False, "w1"), ("llm", 0.9, "rename", False, "w2"),
               ("llm", 0.99, "unknown", False, "dunno")])                     # 'I cannot tell' is never taken, whatever its score
    out = metrics.acceptance(scored(rows), "llm", 0.95)
    assert [(p["cut"], p["accepted"], p["wrong"], p["to_human"]) for p in out["points"]] == [(0.99, 0, 0, 29), (0.95, 25, 0, 4), (0.9, 28, 2, 1)]
    assert out["pick"] == 0.95 and out["same_calls"] and out["accepted"] == 25 and out["wrong_accepted"] == []   # 0.90: 26 of 28
    assert out["drop_one_runs"] == 29 and out["drop_one_same_calls"] == 27                 # without w1 or w2, 0.90 passes (26 of 27)
    assert [(mv["unit"], mv["cut"]) for mv in out["drop_one_moves"]] == [("w1", 0.9), ("w2", 0.9)]
    assert (out["held_out_accepted"], out["held_out_right"]) == (27, 25)                  # w1 and w2 each accepted by a cut chosen without them
    looser = metrics.acceptance(scored(rows), "llm", 0.9)
    assert not looser["same_calls"] and [w["unit"] for w in looser["wrong_accepted"]] == ["w1", "w2"]
    few = metrics.acceptance(scored([("geometry", 1.0, "district", True, f"g{i}") for i in range(7)]), "geometry", None)
    assert few["pick"] is None and few["same_calls"] and few["accepted"] == 0             # 7 answers: too few for any cut
    text = "\n".join(run_evals._acceptance_lines({"llm": out, "geometry": few}))
    assert "| 0.95 (in code) | 25 | 0 | 4 | 100.0% |" in text and "| geometry | 7 | 7 | none | none | yes | 0 | 0 |" in text
    assert "without w1 (0.90, wrong) the cut is 0.90" in text and "25 of 27 accepted answers right" in text


def test_escalation_counts_what_went_to_a_human_and_what_the_cuts_would_take(tmp_path):
    from agents import resolver, store
    base = {"era": "2001_2014", "state": "s", "proposed_by": "x", "verified": "1", "reviewer_note": ""}
    rows = {("2001_2014", "s", "a"): {**base, "unit": "a", "stage": "exact"},
            ("2001_2014", "s", "b"): {**base, "unit": "b", "stage": "llm", "proposed_by": "llm:m"},
            ("2001_2014", "s", "c"): {**base, "unit": "c", "stage": "human", "proposed_by": "fuzzy"},
            ("2001_2014", "s", "d"): {**base, "unit": "d", "stage": "llm", "reviewer_note": "Accepted without review: llm score 0.95 >= cut 0.90"},
            ("2001_2014", "s", "cyber cell"): {**base, "unit": "cyber cell", "stage": "circle", "proposed_by": "rule",
                                               "reason": "a circle of X; held for review: the name looks like a police unit"}}
    path = tmp_path / "um.csv"
    resolver.write_rows(path, {k: {c: v.get(c, "") for c in resolver.CSV_COLUMNS} for k, v in rows.items()})
    g = scored([("llm", 0.95, "rename", False, "b"), ("fuzzy", 0.92, "district", True, "c")])   # the scored answers a human reviewed
    out = metrics.escalation(store.connect(tmp_path / "t.db"), path, gold=g)["2001_2014"]
    assert (out["units"], out["needed_review"], out["rate"]) == (5, 3, 0.6)                # b, c and the held lookup; d never reached a human
    assert (out["accepted_by_cut"], out["wrong_accepted"], out["with_cut"], out["rate_with_cut"]) == (2, 1, 1, 0.2)   # the held lookup has no cut
