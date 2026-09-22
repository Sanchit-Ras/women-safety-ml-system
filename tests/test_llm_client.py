"""The door is tested with a scripted fake provider and a fake clock.

No network, no real sleeping, fully repeatable. Each test pins one promise the
rest of the pipeline relies on.
"""
import pytest
from pydantic import BaseModel

from agents import store
from agents.llm_client import (BudgetExceeded, Limits, LLMClient, PermanentError, ProviderError, QuotaExhausted,
                               RateLimiter, RawResponse, TransientError, make_cache_key)


class Reply(BaseModel):
    answer: str


GOOD = '{"answer": "ok"}'
BIG = Limits(rpm=1000, tpm=10**9, rpd=10**6)


class FakeClock:
    def __init__(self):
        self.now, self.sleeps = 0.0, []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class FakeProvider:
    """Plays back a script: a string is a reply, an exception is raised."""

    def __init__(self, name, script, *, limits=BIG, max_output_tokens=100):
        self.name, self.model, self.limits, self.max_output_tokens = name, f"{name}-model", limits, max_output_tokens
        self.script, self.prompts, self.systems = list(script), [], []

    def generate(self, prompt, schema, temperature, system=None):
        self.prompts.append(prompt)
        self.systems.append(system)
        item = self.script.pop(0)              # an unexpected extra call fails loudly here
        if isinstance(item, Exception):
            raise item
        return RawResponse(item, 10, 5)


@pytest.fixture
def conn(tmp_path):
    c = store.connect(tmp_path / "t.db")
    yield c
    c.close()


def make(conn, providers, max_calls=50):
    fc = FakeClock()
    client = LLMClient(conn, providers, max_calls_per_run=max_calls, run_id="test",
                       clock=fc.clock, sleep=fc.sleep, jitter=lambda: 0.0)
    return client, fc


def outcomes(conn):
    return [r["outcome"] for r in conn.execute("SELECT outcome FROM llm_attempts ORDER BY id")]


# ---- saved answers ----------------------------------------------------------------------

def test_same_question_twice_makes_one_call(conn):
    p = FakeProvider("gemini", [GOOD])
    client, _ = make(conn, [p])
    first, second = client.complete("q", Reply, tag="t"), client.complete("q", Reply, tag="t")
    assert (first.cached, second.cached) == (False, True)
    assert second.parsed.answer == "ok" and len(p.prompts) == 1 and client.calls_made == 1


def test_fingerprint_changes_with_model_prompt_schema_and_temperature():
    base = make_cache_key("m", 0.0, "p", "s")
    assert base == make_cache_key("m", 0.0, "p", "s")
    others = {make_cache_key("m2", 0.0, "p", "s"), make_cache_key("m", 0.0, "p2", "s"),
              make_cache_key("m", 0.0, "p", "s2"), make_cache_key("m", 0.5, "p", "s")}
    assert len(others | {base}) == 5


def test_system_text_reaches_the_provider_and_is_part_of_the_fingerprint(conn):
    p = FakeProvider("gemini", [GOOD, GOOD])
    client, _ = make(conn, [p])
    a = client.complete("q", Reply, tag="t", system="context A")
    b = client.complete("q", Reply, tag="t", system="context B")
    assert p.systems == ["context A", "context B"]
    assert a.cache_key != b.cache_key and not b.cached


# ---- retry, then switch -------------------------------------------------------------------

def test_busy_provider_is_retried_with_growing_waits(conn):
    p = FakeProvider("gemini", [TransientError("503"), TransientError("503"), GOOD])
    client, fc = make(conn, [p])
    result = client.complete("q", Reply, tag="t")
    assert result.tries == 3 and fc.sleeps == [1.0, 2.0]
    assert outcomes(conn) == ["transient_error", "transient_error", "ok"]


def test_three_failures_switch_provider_and_the_answer_is_saved_under_the_intended_model(conn):
    gemini, groq = FakeProvider("gemini", [TransientError("503")] * 3), FakeProvider("groq", [GOOD])
    client, _ = make(conn, [gemini, groq])
    result = client.complete("q", Reply, tag="t")
    assert (result.provider, result.tries) == ("groq", 4)

    # A fresh client with empty scripts: any network call would crash. It must find the saved answer.
    again, _ = make(conn, [FakeProvider("gemini", []), FakeProvider("groq", [])])
    replay = again.complete("q", Reply, tag="t")
    assert replay.cached and replay.provider == "groq"          # honest about who answered


def test_permanent_error_switches_at_once_without_waiting(conn):
    client, fc = make(conn, [FakeProvider("gemini", [PermanentError("401 bad key")]), FakeProvider("groq", [GOOD])])
    result = client.complete("q", Reply, tag="t")
    assert (result.provider, result.tries, fc.sleeps) == ("groq", 2, [])


def test_provider_wait_hint_is_obeyed_when_short_and_ignored_when_long(conn):
    client, fc = make(conn, [FakeProvider("gemini", [TransientError("429", retry_after=7), GOOD])])
    client.complete("short", Reply, tag="t")
    assert fc.sleeps == [7]

    client2, fc2 = make(conn, [FakeProvider("gemini", [TransientError("429", retry_after=600)]),
                               FakeProvider("groq", [GOOD])])
    assert client2.complete("long", Reply, tag="t").provider == "groq" and fc2.sleeps == []


# ---- structure ---------------------------------------------------------------------------------

def test_invalid_reply_gets_one_repair_try_that_shows_the_error(conn):
    p = FakeProvider("gemini", ['{"wrong": 1}', GOOD])
    client, _ = make(conn, [p])
    assert client.complete("q", Reply, tag="t").parsed.answer == "ok"
    assert "rejected by the validator" in p.prompts[1] and "answer" in p.prompts[1]
    assert outcomes(conn) == ["invalid_output", "ok"]


def test_when_nobody_gives_a_valid_answer_it_fails_clearly(conn):
    client, _ = make(conn, [FakeProvider("gemini", ["nope", "nope"]), FakeProvider("groq", ["nope", "nope"])])
    with pytest.raises(ProviderError, match="still invalid"):
        client.complete("q", Reply, tag="t")
    assert outcomes(conn) == ["invalid_output"] * 4


# ---- the wall -------------------------------------------------------------------------------------

def test_wall_stops_the_run_and_saved_answers_are_free(conn):
    client, _ = make(conn, [FakeProvider("gemini", [GOOD, GOOD, GOOD])], max_calls=2)
    client.complete("a", Reply, tag="t")
    client.complete("a", Reply, tag="t")                       # saved answer: does not count
    client.complete("b", Reply, tag="t")
    with pytest.raises(BudgetExceeded):
        client.complete("c", Reply, tag="t")


# ---- the gatekeeper -------------------------------------------------------------------------------

def test_gatekeeper_waits_for_the_oldest_request_to_leave_the_minute():
    fc = FakeClock()
    gate = RateLimiter(Limits(rpm=5, tpm=10**9, rpd=1), clock=fc.clock, sleep=fc.sleep)   # 80% of 5 = 4 a minute
    assert [gate.acquire(1) for _ in range(4)] == [0, 0, 0, 0]
    assert 60 <= gate.acquire(1) < 61


def test_gatekeeper_counts_tokens_and_uses_the_real_count_once_known():
    fc = FakeClock()
    gate = RateLimiter(Limits(rpm=1000, tpm=1000, rpd=1), clock=fc.clock, sleep=fc.sleep)  # 800 tokens a minute
    gate.acquire(500)
    gate.correct_last(100)                                     # provider said it was really 100
    assert gate.acquire(500) == 0                              # 100 + 500 fits
    assert gate.acquire(500) >= 60                             # 100 + 500 + 500 does not


def test_request_too_big_for_a_provider_skips_it_without_calling(conn):
    tiny = FakeProvider("gemini", [], limits=Limits(rpm=1000, tpm=100, rpd=10**6))
    client, _ = make(conn, [tiny, FakeProvider("groq", [GOOD])])
    assert client.complete("q", Reply, tag="t").provider == "groq" and tiny.prompts == []


def test_daily_quota_skips_a_provider_and_stops_clearly_when_all_are_used_up(conn):
    ten_a_day = Limits(rpm=1000, tpm=10**9, rpd=10)            # 90% of 10 = 9
    for _ in range(9):
        store.add_attempt(conn, run_id="earlier", tag="t", cache_key="k", provider="gemini", model="gemini-model", outcome="ok")
    gemini = FakeProvider("gemini", [], limits=ten_a_day)
    client, _ = make(conn, [gemini, FakeProvider("groq", [GOOD], limits=ten_a_day)])
    assert client.complete("q1", Reply, tag="t").provider == "groq" and gemini.prompts == []

    for _ in range(8):                                         # groq now has 1 + 8 = 9
        store.add_attempt(conn, run_id="earlier", tag="t", cache_key="k", provider="groq", model="groq-model", outcome="ok")
    with pytest.raises(QuotaExhausted, match="continues where it stopped"):
        client.complete("q2", Reply, tag="t")


def test_daily_token_quota_is_respected_too(conn):
    store.add_attempt(conn, run_id="earlier", tag="t", cache_key="k", provider="groq", model="groq-model",
                      outcome="ok", prompt_tokens=800, completion_tokens=150)
    groq = FakeProvider("groq", [], limits=Limits(rpm=1000, tpm=10**9, rpd=10**6, tpd=1000))   # 90% = 900
    client, _ = make(conn, [groq])
    with pytest.raises(QuotaExhausted):
        client.complete("q", Reply, tag="t")
