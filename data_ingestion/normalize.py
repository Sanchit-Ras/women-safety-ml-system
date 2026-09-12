"""Text normalisation shared by the profiler, the ML features and the resolver.

One function, one place. Every module that compares a state or district name
must go through ``norm`` so that "A & N Islands", "a&n islands" and "A&N ISLANDS"
cannot be treated as three different places by three different modules.
"""
from __future__ import annotations

import re

_NON_ALNUM = re.compile(r"[^a-z0-9 ]")
_SPACES = re.compile(r"\s+")

# Spellings that differ between years of the *same* source file. Applied after
# ``norm`` so the keys are already lower-case and stripped of punctuation.
STATE_ALIASES: dict[str, str] = {
    "an islands": "a n islands",
    "dn haveli": "d n haveli",
    "delhi ut": "delhi",
}


def norm(value: str) -> str:
    """Lower-case, drop punctuation, collapse whitespace."""
    s = str(value).lower().strip()
    s = _NON_ALNUM.sub("", s)
    return _SPACES.sub(" ", s).strip()


def norm_state(value: str) -> str:
    s = norm(value)
    return STATE_ALIASES.get(s, s)
