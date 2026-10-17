"""The gate's checks on hand-built numbers: every floor, the cut agreement, the prompt-version rule and the panel
comparison. The rebuild itself is the orchestrator's, covered by test_pipeline."""
import json

import pandas as pd

from evals import gate


def metrics(**over):
    m = {"coverage": {"all": {"share": 0.9956}}, "extraction": {"all": {"field_exact_match": 1.0}},
         "hallucination": {"records_without_source_row": 0}, "resolution": {"all_automatic": {"precision": 0.927}},
         "acceptance": {"llm": {"same_calls": True, "pick": 0.9, "in_code": 0.9}, "geometry": {"same_calls": True, "pick": None, "in_code": None}}}
    m.update(over)
    return m


VERSIONS = {"resolver": 1, "lineage": 1}
REPORT = ("2026-10-18", {"prompt_versions": {"resolver": 1, "lineage": 1}})
PANELS = {"panel_rebased": (True, "10 rows equal")}


def failed(results):
    return [r["check"] for r in results if not r["ok"]]


def test_every_check_passes_on_todays_numbers():
    results = gate.checks(metrics(), VERSIONS, REPORT, PANELS)
    assert failed(results) == [] and len(results) == 8
    assert "gate: 8 checks, 0 failed" in gate.render(results)


def test_each_floor_fails_on_its_own():
    assert failed(gate.checks(metrics(coverage={"all": {"share": 0.98}}), VERSIONS, REPORT, PANELS)) == ["coverage, all eras >= 99%"]
    assert failed(gate.checks(metrics(extraction={"all": {"field_exact_match": 0.99}}), VERSIONS, REPORT, PANELS)) == ["extraction field exact-match = 100%"]
    assert failed(gate.checks(metrics(hallucination={"records_without_source_row": 3}), VERSIONS, REPORT, PANELS)) == ["records without a source row = 0"]
    assert failed(gate.checks(metrics(resolution={"all_automatic": {"precision": 0.89}}), VERSIONS, REPORT, PANELS)) == \
        ["resolution precision, automatic stages >= 90%"]
    low = metrics(acceptance={"llm": {"same_calls": False, "pick": 0.9, "in_code": 0.8}})
    r = [x for x in gate.checks(low, VERSIONS, REPORT, PANELS) if not x["ok"]]
    assert [x["check"] for x in r] == ["cut rule agrees with the cut in code: llm"] and r[0]["detail"] == "rule 0.90, code 0.80"


def test_a_prompt_version_must_have_been_measured_by_the_latest_report():
    bumped = {"resolver": 2, "lineage": 1}
    r = [x for x in gate.checks(metrics(), bumped, REPORT, PANELS) if not x["ok"]]
    assert [x["check"] for x in r] == ["the latest report measured the prompt versions in code"] and "report 2026-10-18" in r[0]["detail"]
    old = ("2026-10-17", {})                                                   # a report from before versions were recorded
    assert "records no prompt versions" in [x for x in gate.checks(metrics(), VERSIONS, old, PANELS) if not x["ok"]][0]["detail"]
    assert failed(gate.checks(metrics(), VERSIONS, None, PANELS)) == ["the latest report measured the prompt versions in code"]


def test_latest_report_is_the_newest_json_by_date(tmp_path):
    for d in ("2026-10-10", "2026-10-18", "2026-10-12"):
        (tmp_path / f"{d}.json").write_text(json.dumps({"date": d}), encoding="utf-8")
    assert gate.latest_report(tmp_path) == ("2026-10-18", {"date": "2026-10-18"})
    assert gate.latest_report(tmp_path / "none") is None


def test_frames_equal_tolerates_float_noise_but_not_a_changed_value_or_row():
    a = pd.DataFrame({"district_code": [2, 1], "year": [2001, 2001], "total": [10.0, 20.0], "share_basis": ["observed", None]})
    b = pd.DataFrame({"district_code": [1, 2], "year": [2001, 2001], "total": [20.0 + 1e-9, 10.0], "share_basis": [None, "observed"]})
    assert gate.frames_equal(a, b) == (True, "2 rows equal")                      # order and 1e-9 do not matter
    assert gate.frames_equal(a, b.assign(total=[21.0, 10.0])) == (False, "column total differs")
    assert gate.frames_equal(a, b.head(1)) == (False, "2 rows committed, 1 rebuilt")
    assert gate.frames_equal(a, b.rename(columns={"total": "t"}))[0] is False
    panels = {"panel_rebased": gate.frames_equal(a, b.assign(total=[21.0, 10.0]))}
    assert failed(gate.checks(metrics(), VERSIONS, REPORT, panels)) == ["committed panel_rebased equals the one rebuilt from the committed inputs"]
