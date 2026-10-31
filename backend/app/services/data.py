"""The app's data: one JSON file written by ``python -m ml.export_app`` (batch scoring) and read once per worker.

The API runs no model. The data changes once a year, when NCRB publishes, so every answer is computed ahead of
time and committed with the code that serves it.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ARTIFACT = Path(__file__).resolve().parents[2] / "artifacts" / "app_data.json"


@lru_cache(maxsize=1)
def get() -> dict:
    """The whole file, loaded on first use. A missing or broken file raises here, at startup, not on a user's request."""
    data = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    for key in ("meta", "states", "districts"):
        if key not in data:
            raise ValueError(f"{ARTIFACT.name} has no '{key}' section")
    return data


def states() -> list[dict]:
    """States with their districts, both sorted by name."""
    data = get()
    by_state: dict[int, list[dict]] = {}
    for d in data["districts"].values():
        by_state.setdefault(d["state_code"], []).append({"code": d["code"], "name": d["name"], "status": d["status"]})
    out = [{"code": int(c), "name": name, "districts": sorted(by_state.get(int(c), []), key=lambda x: x["name"])}
           for c, name in data["states"].items() if by_state.get(int(c))]
    return sorted(out, key=lambda s: s["name"])


def district(code: int) -> dict | None:
    d = get()["districts"].get(str(code))
    if d is None:
        return None
    names = {int(k): v["name"] for k, v in get()["districts"].items()}
    return {**d, "unit_members": [{"code": m, "name": names.get(m, str(m))} for m in d["unit_members"] if m != code]}


def watchlist() -> list[dict]:
    """Districts on the watch list, best-ranked first; districts of one unit share its rank."""
    rows = [d for d in get()["districts"].values() if d["status"] == "watch"]
    return [{"code": d["code"], "name": d["name"], "state": d["state"], "rank": d["rank"], "score": d["score"],
             "latest_year": d["latest_year"], "latest_total": d["latest_total"]}
            for d in sorted(rows, key=lambda d: (d["rank"], d["name"]))]


def meta() -> dict:
    return get()["meta"]
