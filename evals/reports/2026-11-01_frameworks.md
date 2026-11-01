# Hand-built loop vs LangGraph (2026-11-01)

The resolver's model step, built three ways, run live on the same 66 answer-key names (24 state batches) with the same prompt text, candidate lists and model (`gemini-3.5-flash-lite`, temperature 0), 2 rounds each, every round from an empty store. The builds took turns state by state; one pacer spaced every call 5.0 s apart, and pacing waits are not counted as response time.

- **A hand-built**: `resolver.resolve_llm` with our model client.
- **B LangGraph, our client**: `agents/resolver_graph.py` calling the same client. Only the control flow changes.
- **C LangGraph + LangChain connector**: the same graph calling `ChatGoogleGenerativeAI` with structured output (JSON schema), given our client's temperature, seed, output cap, timeout and safety filters.

What the connector warned during the run (settings it changed or dropped on its own):

- Model 'gemini-3.5-flash-lite' uses fixed sampling defaults; the sampling parameter(s) temperature will be ignored.

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	prompt(prompt)
	call(call)
	check(check)
	repair(repair)
	finish(finish)
	__end__([<p>__end__</p>]):::last
	__start__ --> prompt;
	call --> check;
	check -.-> finish;
	check -.-> repair;
	prompt --> call;
	repair --> call;
	finish --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

## Results, 2 rounds pooled

| | A hand-built | B LangGraph, our client | C LangGraph + LangChain connector |
|---|---|---|---|
| answers matching the key, round by round | 61 of 66, 61 of 66 | 61 of 66, 61 of 66 | 61 of 66, 61 of 66 |
| share right, 95% interval over the names | 92.4% (83.5% to 96.7%) | 92.4% (83.5% to 96.7%) | 92.4% (83.5% to 96.7%) |
| 'cannot tell' answers | 2 | 2 | 2 |
| invented codes, rejected by the guard | 0 | 0 | 0 |
| names never returned | 0 | 0 | 0 |
| states that failed outright | 0 | 0 | 0 |
| model calls (of which repairs) | 48 (0) | 48 (0) | 48 (0) |
| replies that did not parse | 0 | 0 | 0 |
| tokens per call, in / out | 1,668 / 192 | 1,668 / 192 | 1,667 / 192 |
| cost of all 2 rounds at paid rates | ₹4.54 | ₹4.54 | ₹4.54 |
| response time per call, typical / slowest 5% | 1,868 / 2,691 ms | 1,906 / 2,773 ms | 1,914 / 2,624 ms |
| time spent outside the model call, per state | 14 ms | 15 ms | 3 ms |

The interval treats the names as the unit: a second round asks the same names again, so it is not new evidence. Time outside the model call includes, for A and B, our client's bookkeeping (the cache lookup and the cost records written to the store); LangGraph's own share is the difference between A and B.

## Consistency

How often a build gave the same answer (stage and code) to a name in both rounds: A hand-built 66 of 66 (100.0%); B LangGraph, our client 66 of 66 (100.0%); C LangGraph + LangChain connector 66 of 66 (100.0%).

How often two builds gave the same answer in round 1: A vs B 66 of 66 (100.0%); A vs C 66 of 66 (100.0%); B vs C 66 of 66 (100.0%).

## Code and dependencies

| | A hand-built | B LangGraph, our client | C LangGraph + LangChain connector |
|---|---|---|---|
| lines of control code (no comments or docstrings) | 14 | 35 | 51 |
| packages added to the project | 0 | 21 | 21 |
| disk space they take | 0 | 24.0 MB | 24.0 MB |

Packages added by requirements-frameworks.txt: filetype, httpcore2, httpx2, jsonpatch, jsonpointer, langchain-core, langchain-google-genai, langchain-protocol, langgraph, langgraph-checkpoint, langgraph-prebuilt, langgraph-sdk, langsmith, orjson, ormsgpack, pyyaml, requests-toolbelt, truststore, uuid-utils, xxhash, zstandard. Installing them also raised `idna` from 3.11 to 3.20 (pip's dry run, 2026-11-01).

## What each build gives without extra code

| | A | B | C |
|---|---|---|---|
| answer cache keyed by the exact question | yes | yes | no |
| token, cost and failure records in the store | yes | yes | no |
| rate limits from the account's quotas | yes | yes | no (an `InMemoryRateLimiter` can be added) |
| fallback to a second provider (Groq) | yes | yes | no (`with_fallbacks` can be added) |
| resume from a saved step (checkpointer) | no | yes | yes |
| retry rule per step (`RetryPolicy`) | no (retries live in the client) | yes | yes |
| step-by-step streaming and tracing hooks | own trace log | yes | yes |
| a drawn diagram of the flow | no | yes | yes |

## Reading

- **Answers:** all three builds gave the same answer to every name, and each gave the same answers in both rounds. The framework changed nothing about what the model was asked or how it answered.
- **Hidden settings:** LangChain's connector changed a setting on its own (listed at the top); here it made no difference to the answers, but it is the kind of choice a framework makes without asking.
- **Speed:** typical response times were A 1,868 ms, B 1,906 ms, C 1,914 ms: the model's own time dominates, and the control flow adds milliseconds.
- **Cost of the framework:** 14 lines of control code by hand against 35 with LangGraph (51 with LangChain's connector), and 21 extra packages (24.0 MB). What it buys is in the table above: resume from a saved step, per-step retry rules, streaming, a drawn diagram.
- **When it pays:** for this short loop (one call, one check, at most one repair) the hand-built version is smaller and keeps the cache, cost records and fallback; a longer flow with branches, waiting for a human, or resuming after a failure is where a graph framework starts to earn its packages.
