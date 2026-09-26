"""The crosswalk file is the human's decision record. These pin how it is written, checked and read."""
import pandas as pd
import pytest

from agents import store
from agents.llm_client import LLMResult
from data_ingestion.crosswalk import (ERA_A_DEFINITIONS, ColumnMapping, CrosswalkProposal, apply, continuity, propose,
                                      read_rows, verified_mapping, write_rows)


class FakeClient:
    """Plays back scripted proposals; records what it was asked."""

    def __init__(self, script):
        self.script, self.calls = list(script), []

    def complete(self, prompt, schema, *, tag, system=None):
        self.calls.append((tag, prompt))
        return LLMResult(parsed=self.script.pop(0), provider="fake", model="fake-model", cached=False,
                         prompt_tokens=1, completion_tokens=1, latency_ms=1, tries=1, cache_key="k")


def proposal(**columns):
    return CrosswalkProposal(mappings=[ColumnMapping(source_column=c, category=cat, reason="IPC x", ambiguous=False)
                                       for c, cat in columns.items()])


def row(era, col, category, verified="1"):
    return {"era": era, "source_column": col, "category": category, "core_category": "", "reason": "r",
            "ambiguous": "0", "evidence": "", "proposed_by": "human", "verified": verified, "reviewer_note": ""}


@pytest.fixture
def conn(tmp_path):
    c = store.connect(tmp_path / "t.db")
    yield c
    c.close()


def test_propose_writes_definitions_as_verified_and_model_rows_as_unverified(conn, tmp_path):
    path = tmp_path / "cw.csv"
    client = FakeClient([proposal(rape="rape", acid_attack="acid_attack"), proposal(rape_women_above_18="rape")])
    propose(client, conn, csv_path=path, columns_by_era={"2016": ["rape", "acid_attack"], "2017_plus": ["rape_women_above_18"]})
    rows = read_rows(path)
    assert len(rows) == 7 + 3
    assert rows[("2001_2014", "Rape")]["verified"] == "1" and rows[("2001_2014", "Rape")]["proposed_by"] == "definition"
    r = rows[("2016", "rape")]
    assert (r["category"], r["core_category"], r["verified"], r["proposed_by"]) == ("rape", "rape", "0", "llm:fake-model")
    acid = rows[("2016", "acid_attack")]
    assert (acid["category"], acid["core_category"]) == ("acid_attack", "")        # kept, but not in the 2001+ series


def test_a_forgotten_column_gets_one_repair_try_then_a_placeholder_and_an_escalation(conn, tmp_path):
    path = tmp_path / "cw.csv"
    client = FakeClient([proposal(rape="rape"), proposal(rape="rape")])               # forgets acid_attack twice
    propose(client, conn, csv_path=path, columns_by_era={"2016": ["rape", "acid_attack"]})
    assert [tag for tag, _ in client.calls] == ["crosswalk:2016", "crosswalk:2016:repair"]
    assert "acid_attack" in client.calls[1][1]                                        # the repair prompt names it
    assert read_rows(path)[("2016", "acid_attack")]["reason"].startswith("NOT PROPOSED")
    assert conn.execute("SELECT COUNT(*) FROM escalations WHERE kind = 'crosswalk'").fetchone()[0] == 1


def test_human_verified_rows_survive_a_re_propose_and_apply_loads_only_them(conn, tmp_path):
    path = tmp_path / "cw.csv"
    cols = {"2016": ["rape", "attempt_to_commit_rape"]}
    propose(FakeClient([proposal(rape="rape", attempt_to_commit_rape="rape")]), conn, csv_path=path, columns_by_era=cols)

    rows = read_rows(path)                                                            # the human overrules one row
    rows[("2016", "attempt_to_commit_rape")].update(category="attempt_to_rape", verified="1",
                                                    reviewer_note="separate NCRB head since 2014")
    write_rows(path, rows, {})

    propose(FakeClient([proposal(rape="dowry_deaths", attempt_to_commit_rape="rape")]), conn, csv_path=path, columns_by_era=cols)
    rows = read_rows(path)
    assert (rows[("2016", "attempt_to_commit_rape")]["category"], rows[("2016", "attempt_to_commit_rape")]["verified"]) == ("attempt_to_rape", "1")
    assert (rows[("2016", "rape")]["category"], rows[("2016", "rape")]["verified"]) == ("dowry_deaths", "0")

    assert apply(conn, csv_path=path) == {"loaded": {"2001_2014": 7, "2016": 1}, "still_unverified": {"2016": 1}}
    assert verified_mapping(conn, "2016") == {"attempt_to_commit_rape": "attempt_to_rape"}


def test_apply_rejects_an_unknown_category_and_a_stale_core_is_recomputed(conn, tmp_path):
    path = tmp_path / "cw.csv"
    write_rows(path, {("2016", "x"): row("2016", "x", "murder")}, {})
    with pytest.raises(ValueError, match="murder"):
        apply(conn, csv_path=path)
    stale = row("2016", "acid_attack", "acid_attack")
    stale["core_category"] = "rape"                      # a reviewer changed category and left the old core behind
    write_rows(path, {("2016", "acid_attack"): stale}, {})
    assert read_rows(path)[("2016", "acid_attack")]["core_category"] == ""          # fixed on write
    apply(conn, csv_path=path)
    assert conn.execute("SELECT core_category FROM crosswalk WHERE source_column = 'acid_attack'").fetchone()[0] is None


def test_continuity_reports_the_boundary_jump_with_and_without_a_column():
    zeros = {col: [0, 0] for col in ERA_A_DEFINITIONS}
    era_a = pd.DataFrame({"Year": [2013, 2014], "DISTRICT": ["x", "x"], **zeros})
    era_a["Rape"] = [70, 80]
    era_b = pd.DataFrame({"year": [2016], "rape": [100], "attempt_to_commit_rape": [20]})
    era_c = pd.DataFrame({"year": [2017], "rape_women_above_18": [90]})
    frames = {"2001_2014": era_a, "2016": era_b, "2017_plus": era_c}
    rows = {("2016", "rape"): row("2016", "rape", "rape"),
            ("2016", "attempt_to_commit_rape"): row("2016", "attempt_to_commit_rape", "rape"),   # the questionable one
            ("2017_plus", "rape_women_above_18"): row("2017_plus", "rape_women_above_18", "rape")}
    table, evidence = continuity(rows, frames)
    assert table.loc["rape", [2013, 2014, 2016, 2017]].tolist() == [70, 80, 120, 90]
    note = evidence[("2016", "attempt_to_commit_rape")]
    assert "+50% with this column" in note and "+25% without" in note and "20 in 2016" in note


def test_column_fixes_move_values_all_at_once_and_only_inside_the_year_range():
    from data_ingestion.crosswalk import apply_column_fixes
    df = pd.DataFrame({"year": [2017, 2018], "a": [1, 10], "b": [2, 20], "c": [3, 30]})
    rot = [{"era": "2017_plus", "year_from": "2018", "year_to": "2022", "file_column": "a", "actual_column": "b"},
           {"era": "2017_plus", "year_from": "2018", "year_to": "2022", "file_column": "b", "actual_column": "c"},
           {"era": "2017_plus", "year_from": "2018", "year_to": "2022", "file_column": "c", "actual_column": "a"}]
    out = apply_column_fixes(df, rot, "2017_plus")
    assert out.loc[0, ["a", "b", "c"]].tolist() == [1, 2, 3]            # 2017 untouched
    assert out.loc[1, ["a", "b", "c"]].tolist() == [30, 10, 20]         # b <- a, c <- b, a <- c, simultaneously
    assert apply_column_fixes(df, rot, "2016").equals(df)               # other eras untouched


def test_fidelity_compares_the_file_with_published_figures_with_and_without_fixes():
    from data_ingestion.crosswalk import fidelity
    zeros = {col: [0] for col in ERA_A_DEFINITIONS}
    era_a = pd.DataFrame({"Year": [2014], "DISTRICT": ["x"], **zeros})
    raw_c = pd.DataFrame({"year": [2018], "rape_women_above_18": [90], "rape_girls_below_18": [50], "attempt_x": [10]})
    fixed_c = raw_c.assign(rape_girls_below_18=[10])                    # the fix moves 10 into the right column
    rows = {("2017_plus", "rape_women_above_18"): row("2017_plus", "rape_women_above_18", "rape"),
            ("2017_plus", "rape_girls_below_18"): row("2017_plus", "rape_girls_below_18", "rape")}
    reference = pd.DataFrame([{"head": "Rape", "category": "rape", "year": 2018, "cases": 100}])
    frames = lambda c: {"2001_2014": era_a, "2016": pd.DataFrame({"year": []}), "2017_plus": c}
    out = fidelity(rows, reference, {"none": frames(raw_c), "proposed": frames(fixed_c)})
    r = out.iloc[0]
    assert (r["published"], r["file_as_is"], r["with_fixes"]) == (100, 140, 100) and r["diff"] == 0
