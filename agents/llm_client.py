"""The one door every model call in this project goes through.

    result = client.complete(prompt, schema=SomePydanticModel, tag="crosswalk:2017_plus")
    result.parsed      -> a validated SomePydanticModel
    result.provider    -> who actually answered ("gemini" / "groq")
    result.cached      -> True when no network call happened

What the door does, in order, on every call:

1. Saved answers. The question is fingerprinted (sha256 of intended model,
   temperature, prompt and output schema). If that fingerprint is already in
   ``llm_calls`` the saved answer is returned: no network, no cost, and the
   same run gives the same result tomorrow.
2. The wall. ``MAX_CALLS_PER_RUN`` network tries per run; one more raises
   ``BudgetExceeded``. A loop bug hits this wall, not the daily quota.
3. The gatekeeper. Per provider, a sliding 60-second window over requests and
   tokens (kept at 80% of the published limit), plus a 24-hour count read
   from the receipts table (90%). A provider that is out of room is skipped.
4. Retry, then switch. Up to three tries per provider with backoff on
   "busy / rate limited / timeout"; then the next provider in the chain.
5. Structure. The model must answer in JSON matching the pydantic schema. The
   answer is validated again here, because "JSON mode" promises valid JSON,
   not the right fields. One repair try shows the model its own error.
6. Receipts. Every network try writes one row to ``llm_attempts``: outcome,
   tokens, latency. Cost and quota are computed from these rows.

Nothing here knows about districts or crime data. Providers are passed in, so
tests drive the whole door with a scripted fake and a fake clock: no network,
no sleeping, fully deterministic.

Run ``python -m agents.llm_client ping`` to check keys, model names and
structured output end to end; ``usage`` shows the last 24 hours per provider.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sqlite3
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Callable, Generic, Literal, Optional, Protocol, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from agents import store

T = TypeVar("T", bound=BaseModel)

TEMPERATURE = 0.0                 # evals need the same input to give the same output
SEED = 7                          # passed to providers that accept one
MAX_TRIES_PER_PROVIDER = 3
BACKOFF_BASE_SECONDS = 1.0        # waits 1s, 2s (+ jitter) between tries
MAX_WORTHWHILE_WAIT_SECONDS = 65  # a provider asking for longer than this is treated as exhausted
MINUTE_HEADROOM = 0.8             # use at most 80% of a per-minute limit
DAY_HEADROOM = 0.9                # use at most 90% of a per-day limit
CHARS_PER_TOKEN = 3               # deliberately pessimistic estimate for JSON-heavy text

REPAIR_SUFFIX = (
    "\n\nYour previous reply was rejected by the validator with this error:\n{error}\n"
    "Reply again with ONLY a JSON object that fixes it and matches the required schema exactly."
)


# ---- errors ---------------------------------------------------------------------------

class LLMError(Exception):
    """Base class for everything this module raises on purpose."""


class BudgetExceeded(LLMError):
    """The per-run wall was hit. Callers must let this stop the run."""


class QuotaExhausted(LLMError):
    """Every provider is out of daily quota. Saved answers are kept; rerun later."""


class ProviderError(LLMError):
    """Every provider was tried and none produced a valid answer."""


class TransientError(Exception):
    """Raised by a provider for 'try again later': HTTP 429, 5xx, timeouts, dropped connections."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class PermanentError(Exception):
    """Raised by a provider when retrying cannot help: bad key, bad request, blocked content."""


# ---- small value types ------------------------------------------------------------------

@dataclass(frozen=True)
class Limits:
    rpm: int                      # requests per minute
    tpm: int                      # tokens per minute
    rpd: int                      # requests per day
    tpd: Optional[int] = None     # tokens per day (None = provider has no such limit)


@dataclass
class RawResponse:
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None


@dataclass
class LLMResult(Generic[T]):
    parsed: T
    provider: str
    model: str
    cached: bool
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: int | None
    tries: int
    cache_key: str


class Provider(Protocol):
    name: str
    model: str
    limits: Limits
    max_output_tokens: int

    def generate(self, prompt: str, schema: type[BaseModel], temperature: float) -> RawResponse: ...


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN + 1


def make_cache_key(intended_model: str, temperature: float, prompt: str, schema_json: str) -> str:
    """Fingerprint of the *question*. The model in it is the one we meant to ask
    (first in the chain), so a fallback answer is found again on the next run."""
    blob = json.dumps({"model": intended_model, "temperature": temperature, "prompt": prompt, "schema": schema_json},
                      sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---- the gatekeeper ------------------------------------------------------------------------

class RateLimiter:
    """Sliding 60-second window over requests and tokens for one provider."""

    WINDOW = 60.0

    def __init__(self, limits: Limits, *, clock: Callable[[], float], sleep: Callable[[float], None]):
        self.max_requests = max(1, int(limits.rpm * MINUTE_HEADROOM))
        self.max_tokens = max(1, int(limits.tpm * MINUTE_HEADROOM))
        self._clock, self._sleep = clock, sleep
        self._events: deque[list[float]] = deque()          # [time, tokens]

    def can_ever_fit(self, tokens: int) -> bool:
        return tokens <= self.max_tokens

    def acquire(self, tokens: int) -> float:
        """Block until one more request of ``tokens`` fits in the window. Returns seconds waited."""
        waited = 0.0
        while True:
            now = self._clock()
            while self._events and now - self._events[0][0] >= self.WINDOW:
                self._events.popleft()
            used = sum(e[1] for e in self._events)
            if len(self._events) < self.max_requests and used + tokens <= self.max_tokens:
                self._events.append([now, float(tokens)])
                return waited
            pause = self._events[0][0] + self.WINDOW - now + 0.05
            self._sleep(pause)
            waited += pause

    def correct_last(self, actual_tokens: int) -> None:
        """Replace the estimate for the latest request with what the provider really counted."""
        if self._events:
            self._events[-1][1] = float(actual_tokens)


# ---- real providers ---------------------------------------------------------------------------

def _retry_after_seconds(text: str) -> float | None:
    """Best-effort: pull a wait time out of an error ('retryDelay': '23s', 'try again in 7.5s')."""
    m = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", text) or \
        re.search(r"try again in (\d+(?:\.\d+)?)\s*s", text, flags=re.I)
    return float(m.group(1)) if m else None


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, model: str, limits: Limits, max_output_tokens: int = 8192, timeout_s: int = 90):
        from google import genai
        from google.genai import types
        self.model, self.limits, self.max_output_tokens = model, limits, max_output_tokens
        self._types = types
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout_s * 1000))

    def generate(self, prompt: str, schema: type[BaseModel], temperature: float) -> RawResponse:
        import httpx
        from google.genai import errors
        try:
            resp = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=self._types.GenerateContentConfig(
                    temperature=temperature,
                    seed=SEED,
                    max_output_tokens=self.max_output_tokens,
                    response_mime_type="application/json",
                    response_json_schema=schema.model_json_schema(),
                    # We never let the model call tools; switching this off also silences an SDK warning.
                    automatic_function_calling=self._types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
        except errors.APIError as e:
            code = getattr(e, "code", None)
            text = f"{code}: {getattr(e, 'message', '') or e}"[:300]
            if code == 429 or (isinstance(code, int) and code >= 500):
                raise TransientError(text, retry_after=_retry_after_seconds(str(e))) from e
            raise PermanentError(text) from e
        except httpx.HTTPError as e:                       # timeouts, dropped connections
            raise TransientError(f"{type(e).__name__}: {e}"[:300]) from e

        if not resp.text:                                  # blocked or cut off: retrying the same model cannot help
            block = getattr(getattr(resp, "prompt_feedback", None), "block_reason", None)
            finish = resp.candidates[0].finish_reason if resp.candidates else None
            raise PermanentError(f"empty reply (block_reason={block}, finish_reason={finish})")
        um = resp.usage_metadata
        completion = (um.candidates_token_count or 0) + (um.thoughts_token_count or 0) if um else None
        return RawResponse(resp.text, um.prompt_token_count if um else None, completion)


class GroqProvider:
    name = "groq"

    def __init__(self, api_key: str, model: str, limits: Limits, max_output_tokens: int = 4096, timeout_s: int = 90):
        from groq import Groq
        self.model, self.limits, self.max_output_tokens = model, limits, max_output_tokens
        self._client = Groq(api_key=api_key, max_retries=0, timeout=timeout_s)   # retries are ours, so they are counted

    def generate(self, prompt: str, schema: type[BaseModel], temperature: float) -> RawResponse:
        import groq
        system = ("You are a careful data assistant. Reply with a single JSON object and nothing else. "
                  "The JSON object must conform to this JSON Schema:\n" + json.dumps(schema.model_json_schema()))
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                seed=SEED,
                max_completion_tokens=self.max_output_tokens,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            )
        except groq.RateLimitError as e:
            header = e.response.headers.get("retry-after") if getattr(e, "response", None) is not None else None
            wait = float(header) if header and header.replace(".", "", 1).isdigit() else _retry_after_seconds(str(e))
            raise TransientError(f"429: {e}"[:300], retry_after=wait) from e
        except (groq.APITimeoutError, groq.APIConnectionError, groq.InternalServerError) as e:
            raise TransientError(f"{type(e).__name__}: {e}"[:300]) from e
        except groq.APIStatusError as e:
            raise PermanentError(f"{e.status_code}: {e}"[:300]) from e

        text = resp.choices[0].message.content or ""
        if not text:
            raise PermanentError(f"empty reply (finish_reason={resp.choices[0].finish_reason})")
        u = resp.usage
        return RawResponse(text, u.prompt_tokens if u else None, u.completion_tokens if u else None)


# ---- the door ---------------------------------------------------------------------------------------

class LLMClient:
    def __init__(self, conn: sqlite3.Connection, providers: list[Provider], *, max_calls_per_run: int,
                 run_id: str | None = None, temperature: float = TEMPERATURE,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
                 jitter: Callable[[], float] = lambda: random.uniform(0.0, 0.5)):
        if not providers:
            raise ValueError("LLMClient needs at least one provider")
        self.conn, self.providers = conn, providers
        self.max_calls_per_run, self.temperature = max_calls_per_run, temperature
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.calls_made = 0
        self._clock, self._sleep, self._jitter = clock, sleep, jitter
        self._limiters = {id(p): RateLimiter(p.limits, clock=clock, sleep=sleep) for p in providers}

    # -- helpers ------------------------------------------------------------------------------------

    def _spend_one_call(self) -> None:
        if self.calls_made >= self.max_calls_per_run:
            raise BudgetExceeded(f"run {self.run_id} reached MAX_CALLS_PER_RUN={self.max_calls_per_run}; "
                                 "stopping so a loop cannot eat the daily quota")
        self.calls_made += 1

    def _daily_block(self, p: Provider) -> str | None:
        tries, tokens = store.usage_last_24h(self.conn, p.name, p.model)
        if tries >= int(p.limits.rpd * DAY_HEADROOM):
            return f"daily request quota used ({tries} of {p.limits.rpd} in the last 24h)"
        if p.limits.tpd is not None and tokens >= int(p.limits.tpd * DAY_HEADROOM):
            return f"daily token quota used ({tokens} of {p.limits.tpd} in the last 24h)"
        return None

    def _receipt(self, p: Provider, tag: str, key: str, outcome: str, *, detail: str | None = None,
                 raw: RawResponse | None = None, latency_ms: int | None = None) -> None:
        store.add_attempt(self.conn, run_id=self.run_id, tag=tag, cache_key=key, provider=p.name, model=p.model,
                          outcome=outcome, detail=detail, latency_ms=latency_ms,
                          prompt_tokens=raw.prompt_tokens if raw else None,
                          completion_tokens=raw.completion_tokens if raw else None)

    # -- the one public method ---------------------------------------------------------------------------

    def complete(self, prompt: str, schema: type[T], *, tag: str) -> LLMResult[T]:
        schema_json = json.dumps(schema.model_json_schema(), sort_keys=True)
        key = make_cache_key(self.providers[0].model, self.temperature, prompt, schema_json)

        saved = store.get_cached_call(self.conn, key)                                   # 1. saved answers
        if saved is not None:
            return LLMResult(parsed=schema.model_validate_json(saved["response_json"]), provider=saved["provider"],
                             model=saved["model"], cached=True, prompt_tokens=saved["prompt_tokens"],
                             completion_tokens=saved["completion_tokens"], latency_ms=saved["latency_ms"],
                             tries=0, cache_key=key)

        tries, problems, quota_blocks = 0, [], 0
        for p in self.providers:
            limiter = self._limiters[id(p)]
            estimate = estimate_tokens(prompt) + estimate_tokens(schema_json) + p.max_output_tokens
            if not limiter.can_ever_fit(estimate):                                      # 3. gatekeeper (size)
                problems.append(f"{p.name}: a ~{estimate}-token request can never fit its per-minute token limit")
                continue
            blocked = self._daily_block(p)                                              # 3. gatekeeper (day)
            if blocked:
                quota_blocks += 1
                problems.append(f"{p.name}: {blocked}")
                continue

            current_prompt, repaired = prompt, False
            for n in range(1, MAX_TRIES_PER_PROVIDER + 1):                               # 4. retry, then switch
                self._spend_one_call()                                                  # 2. the wall
                tries += 1
                limiter.acquire(estimate)                                               # 3. gatekeeper (minute)
                started = self._clock()
                try:
                    raw = p.generate(current_prompt, schema, self.temperature)
                except TransientError as e:
                    self._receipt(p, tag, key, "transient_error", detail=str(e),
                                  latency_ms=int((self._clock() - started) * 1000))
                    too_long = e.retry_after is not None and e.retry_after > MAX_WORTHWHILE_WAIT_SECONDS
                    if too_long or n == MAX_TRIES_PER_PROVIDER:
                        problems.append(f"{p.name}: {e}")
                        break
                    self._sleep(e.retry_after if e.retry_after is not None
                                else BACKOFF_BASE_SECONDS * 2 ** (n - 1) + self._jitter())
                    continue
                except PermanentError as e:
                    self._receipt(p, tag, key, "permanent_error", detail=str(e),
                                  latency_ms=int((self._clock() - started) * 1000))
                    problems.append(f"{p.name}: {e}")
                    break

                latency_ms = int((self._clock() - started) * 1000)
                if raw.prompt_tokens is not None:
                    limiter.correct_last(raw.prompt_tokens + (raw.completion_tokens or 0))
                try:
                    parsed = schema.model_validate_json(raw.text)                       # 5. structure
                except ValidationError as e:
                    short = "; ".join(f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()[:5])
                    self._receipt(p, tag, key, "invalid_output", detail=short[:300], raw=raw, latency_ms=latency_ms)
                    if repaired:
                        problems.append(f"{p.name}: reply still invalid after one repair try ({short[:120]})")
                        break
                    repaired, current_prompt = True, prompt + REPAIR_SUFFIX.format(error=short)
                    continue

                self._receipt(p, tag, key, "ok", raw=raw, latency_ms=latency_ms)        # 6. receipts
                store.put_call(self.conn, cache_key=key, run_id=self.run_id, tag=tag, provider=p.name, model=p.model,
                               temperature=self.temperature, prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
                               schema_hash=hashlib.sha256(schema_json.encode()).hexdigest(),
                               prompt_tokens=raw.prompt_tokens, completion_tokens=raw.completion_tokens,
                               latency_ms=latency_ms, attempts=tries, response_json=parsed.model_dump_json())
                return LLMResult(parsed=parsed, provider=p.name, model=p.model, cached=False,
                                 prompt_tokens=raw.prompt_tokens, completion_tokens=raw.completion_tokens,
                                 latency_ms=latency_ms, tries=tries, cache_key=key)

        summary = " | ".join(problems)
        if quota_blocks == len(self.providers):
            raise QuotaExhausted(f"every provider is out of daily quota ({summary}). Saved answers are kept: "
                                 "rerun later and the work continues where it stopped.")
        raise ProviderError(f"no provider produced a valid answer for '{tag}': {summary}")


# ---- building the real client from .env ------------------------------------------------------------------

def _limits_from_env(prefix: str) -> Limits:
    tpd = os.environ.get(f"{prefix}_TPD")
    return Limits(rpm=int(os.environ[f"{prefix}_RPM"]), tpm=int(os.environ[f"{prefix}_TPM"]),
                  rpd=int(os.environ[f"{prefix}_RPD"]), tpd=int(tpd) if tpd else None)


def build_client(conn: sqlite3.Connection | None = None, *, only: str | None = None) -> LLMClient:
    """Gemini first, Groq second, from the settings in .env. ``only`` keeps a single provider (for checks)."""
    load_dotenv(store.ROOT / ".env")
    providers: list[Provider] = []
    if os.environ.get("GEMINI_API_KEY") and only in (None, "gemini"):
        providers.append(GeminiProvider(os.environ["GEMINI_API_KEY"], os.environ["GEMINI_MODEL"], _limits_from_env("GEMINI")))
    if os.environ.get("GROQ_API_KEY") and only in (None, "groq"):
        providers.append(GroqProvider(os.environ["GROQ_API_KEY"], os.environ["GROQ_MODEL"], _limits_from_env("GROQ")))
    if not providers:
        raise LLMError("no provider configured: put GEMINI_API_KEY and/or GROQ_API_KEY in .env")
    return LLMClient(conn or store.connect(), providers, max_calls_per_run=int(os.environ.get("MAX_CALLS_PER_RUN", "150")))


# ---- command line: ping / usage -------------------------------------------------------------------------------

class _PingItem(BaseModel):
    district: str
    state: str
    kind: Optional[Literal["state_capital", "other"]] = None


class _PingReply(BaseModel):
    items: list[_PingItem]


PING_PROMPT = ("For each of these Indian districts give the state it belongs to, and set kind to 'state_capital' "
               "if the district contains its state's capital city, otherwise 'other': Anantapur, Hyderabad, Jammu.")


def _print_usage(client: LLMClient) -> None:
    print("\nlast 24 hours, from the receipts table:")
    for p in client.providers:
        tries, tokens = store.usage_last_24h(client.conn, p.name, p.model)
        tpd = f" of {p.limits.tpd}" if p.limits.tpd else ""
        print(f"  {p.name:<7} {p.model:<28} tries {tries:>4} of {p.limits.rpd:<5} tokens {tokens:>7}{tpd}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Check the LLM setup.")
    ap.add_argument("command", choices=["ping", "usage"])
    ap.add_argument("--provider", choices=["gemini", "groq"], help="use only this provider")
    args = ap.parse_args(argv)
    client = build_client(only=args.provider)
    if args.command == "ping":
        r = client.complete(PING_PROMPT, _PingReply, tag="ping")
        print(f"answered by {r.provider} ({r.model})  saved-answer={r.cached}  tries={r.tries}  "
              f"tokens in/out={r.prompt_tokens}/{r.completion_tokens}  latency={r.latency_ms}ms")
        for item in r.parsed.items:
            print(f"  {item.district:<12} {item.state:<20} {item.kind}")
    _print_usage(client)


if __name__ == "__main__":
    main()
