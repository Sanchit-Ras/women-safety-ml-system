"""The text the pipeline sends to the model, one file per prompt in ``prompts/``, named ``<name>.v<N>.txt``.

Prompts are code. A change is a new file (``resolver.v2.txt``) plus a bump in VERSIONS, never an edit in
place: the old version stays, so every eval report, which names the versions it measured, can be reproduced,
and ``evals.gate`` refuses a version no report has measured. The files hold every fixed sentence; the builders
in resolver, lineage and crosswalk pour the data lines (candidates, names, columns) into the ``{slots}``.
A file ends with one newline that is not part of the prompt. A literal brace in a prompt is written ``{{``.

``PROJECT_CONTEXT`` is sent as the system instruction on every call so the model knows it is reading
statistical metadata from official crime tables. Column headers such as ``rape_girls_below_18`` are legal
offence categories from government statistics; saying so up front is both accurate and the main defence
against a model refusing to discuss them.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = ROOT / "prompts"
VERSIONS: dict[str, int] = {"project_context": 1, "resolver": 1, "lineage": 1, "lineage_neighbour": 1, "crosswalk": 1,
                            "repair": 1}


def path(name: str, version: int | None = None) -> Path:
    return PROMPTS_DIR / f"{name}.v{VERSIONS[name] if version is None else version}.txt"


@lru_cache(maxsize=None)
def load(name: str) -> str:
    """The prompt text of the version in use, without the file's final newline."""
    text = path(name).read_text(encoding="utf-8")
    return text[:-1] if text.endswith("\n") else text


def render(name: str, **slots) -> str:
    return load(name).format_map(slots)


def repair_lines(what: str, n: int, missing=(), unknown=()) -> list[str]:
    """The follow-up when a batch answer did not cover the set asked: a blank line, then the lines of repair.v<N>.txt,
    the middle two only when needed. Empty when nothing is missing or unknown."""
    if not missing and not unknown:
        return []
    first, miss, unk, last = load("repair").split("\n")
    out = ["", first]
    if missing:
        out.append(miss.format(what=what, missing=sorted(missing)))
    if unknown:
        out.append(unk.format(what=what, unknown=sorted(unknown)))
    out.append(last.format(what=what, n=n))
    return out


def fingerprint() -> dict[str, str]:
    """sha256 of each prompt file in use, 12 hex characters: a report says exactly which text it measured."""
    return {name: hashlib.sha256(load(name).encode("utf-8")).hexdigest()[:12] for name in VERSIONS}


PROJECT_CONTEXT = load("project_context")
