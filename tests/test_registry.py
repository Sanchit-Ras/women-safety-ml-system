"""The registry is read from the files, never guessed. These pin how it reads."""
import pandas as pd
import pytest

from agents import store
from data_ingestion.registry import KEY, build_registry, write_registry

ROWS = [
    # year, state_name, state_code, district_name, district_code, registration_circles
    (2016, "Andhra Pradesh", 28, "Anantapur", 502, "Anantapur"),
    (2016, "Andhra Pradesh", 28, "Anantapur", 502, "Guntakal Rly."),
    (2017, "Andhra Pradesh", 28, "Ananthapuramu", 502, "Anantapur"),
    (2017, "Andhra Pradesh", 28, "Ananthapuramu", 502, "Guntakal Railway"),
    (2016, "Jammu and Kashmir", 1, "Leh", 300, "Leh"),
    (2017, "Ladakh", 37, "Leh", 300, "Leh"),
    (2016, "Rajasthan", 8, "Jaipur (Gramin)", 783, "Jaipur Rural"),
    (2017, "Rajasthan", 8, "Jaipur", 102, "Jaipur Rural"),
]


@pytest.fixture
def reg():
    return build_registry(pd.DataFrame(ROWS, columns=KEY))


def district(reg, code):
    return next(d for d in reg["districts"] if d["district_code"] == code)


def test_latest_spelling_is_the_name_and_every_spelling_is_an_alias(reg):
    d = district(reg, 502)
    assert d["name"] == "Ananthapuramu"
    assert d["aliases"] == ["anantapur", "ananthapuramu"]


def test_circles_are_collected_with_their_years(reg):
    circles = {c["norm"]: c for c in district(reg, 502)["circles"]}
    assert set(circles) == {"anantapur", "guntakal rly", "guntakal railway"}
    assert (circles["guntakal rly"]["first_seen_year"], circles["guntakal rly"]["last_seen_year"]) == (2016, 2016)
    assert circles["anantapur"]["last_seen_year"] == 2017


def test_a_district_that_changed_state_keeps_the_latest_state_and_is_listed(reg):
    leh = district(reg, 300)
    assert leh["state_code"] == 37 and leh["state_codes_seen"] == [1, 37]
    assert reg["checks"]["codes_that_changed_state"] == [300]


def test_a_code_that_disappears_is_flagged_and_queued_exactly_once(reg, tmp_path):
    flagged = [d["district_code"] for d in reg["checks"]["codes_that_disappear_before_last_year"]]
    assert flagged == [783]
    conn = store.connect(tmp_path / "t.db")
    first, second = write_registry(conn, reg), write_registry(conn, reg)     # a rebuild must not duplicate the queue
    assert first["escalations_queued"] == 1 and second["escalations_queued"] == 0
    counts = store.table_counts(conn)
    assert (counts["states"], counts["districts"], counts["circles"], counts["escalations"]) == (4, 4, 6, 1)
