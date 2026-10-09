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
    rows = {("2001_2014", "s", "anantapur"): {"era": "2001_2014", "state": "s", "unit": "anantapur"},
            ("2001_2014", "s", "kadapa"): {"era": "2001_2014", "state": "s", "unit": "kadapa"},
            ("2015", "s", "ysr"): {"era": "2015", "state": "s", "unit": "ysr"}}
    path = tmp_path / "um.csv"
    resolver.write_rows(path, {k: {c: v.get(c, "") for c in resolver.CSV_COLUMNS} for k, v in rows.items()})
    registry = {"districts": [{"name": "Anantapur", "aliases": ["anantapur"]}, {"name": "Y.S.R.", "aliases": ["ysr", "cuddapah"]}]}
    out = metrics.notebook_coverage(path, registry)
    assert (out["units"], out["matched"], out["share"]) == (2, 1, 0.5)


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


def test_prompt_fingerprint_is_stable_and_names_every_prompt():
    a, b = metrics.prompt_fingerprint(), metrics.prompt_fingerprint()
    assert a == b and set(a) == {"project_context", "resolver", "lineage", "crosswalk"} and all(len(v) == 12 for v in a.values())


def test_the_report_renders_and_the_history_row_has_every_column(tmp_path):
    m = {"date": "2026-10-10", "gold_districts": "evals/gold/districts.csv",
         "coverage": {"all": {"records": 10, "with_district": 9, "unallocated": 1, "share": 0.9}},
         "notebook_coverage": {"units": 4, "matched": 3, "share": 0.75, "spec_figure": 0.727},
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
