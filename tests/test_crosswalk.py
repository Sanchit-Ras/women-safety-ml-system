"""The crosswalk file is the human's decision record. These pin how it is written and read."""
import pytest

from agents import store
from agents.llm_client import LLMResult
from data_ingestion.crosswalk import ColumnMapping, CrosswalkProposal, apply, propose, read_rows, verified_mapping, write_rows


class FakeClient:
    """Plays back scripted proposals; records what it was asked."""

    def __init__(self, script):
        self.script, self.calls = list(script), []

    def complete(self, prompt, schema, *, tag, system=None):
        self.calls.append((tag, prompt))
        return LLMResult(parsed=self.script.pop(0), provider="fake", model="fake-model", cached=False,
                         prompt_tokens=1, completion_tokens=1, latency_ms=1, tries=1, cache_key="k")


def proposal(**columns):
    return CrosswalkProposal(mappings=[ColumnMapping(source_column=c, core_category=cat, reason="IPC x", ambiguous=False)
                                       for c, cat in columns.items()])


@pytest.fixture
def conn(tmp_path):
    c = store.connect(tmp_path / "t.db")
    yield c
    c.close()


def test_propose_writes_definitions_as_verified_and_model_rows_as_unverified(conn, tmp_path):
    path = tmp_path / "cw.csv"
    client = FakeClient([proposal(rape="rape", acid_attack=None), proposal(rape_women_above_18="rape")])
    propose(client, conn, csv_path=path, columns_by_era={"2016": ["rape", "acid_attack"], "2017_plus": ["rape_women_above_18"]})
    rows = read_rows(path)
    assert len(rows) == 7 + 3
    assert rows[("2001_2014", "Rape")]["verified"] == "1" and rows[("2001_2014", "Rape")]["proposed_by"] == "definition"
    r = rows[("2016", "rape")]
    assert (r["core_category"], r["verified"], r["proposed_by"]) == ("rape", "0", "llm:fake-model")
    assert rows[("2016", "acid_attack")]["core_category"] == ""                       # null = not one of the seven


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
    rows[("2016", "attempt_to_commit_rape")].update(core_category="", verified="1", reviewer_note="Era A Rape excluded attempts")
    write_rows(path, rows, {})

    propose(FakeClient([proposal(rape="dowry_deaths", attempt_to_commit_rape="rape")]), conn, csv_path=path, columns_by_era=cols)
    rows = read_rows(path)
    assert (rows[("2016", "attempt_to_commit_rape")]["core_category"], rows[("2016", "attempt_to_commit_rape")]["verified"]) == ("", "1")
    assert (rows[("2016", "rape")]["core_category"], rows[("2016", "rape")]["verified"]) == ("dowry_deaths", "0")

    assert apply(conn, csv_path=path) == {"loaded": {"2001_2014": 7, "2016": 1}, "still_unverified": {"2016": 1}}
    assert verified_mapping(conn, "2016") == {"attempt_to_commit_rape": None}


def test_apply_rejects_a_category_that_is_not_one_of_the_seven(conn, tmp_path):
    path = tmp_path / "cw.csv"
    write_rows(path, {("2016", "x"): {"era": "2016", "source_column": "x", "core_category": "murder", "reason": "r",
                                      "ambiguous": "0", "proposed_by": "human", "verified": "1", "reviewer_note": ""}}, {})
    with pytest.raises(ValueError, match="murder"):
        apply(conn, csv_path=path)
