"""District lineage: which district came from which, and the stable units built from it.

Districts split, merge, get a new code, or have their police data pooled under a
neighbour (a commissionerate). A time series added up over one modern code is
therefore not the same land in every year. This module does three things:

1. ``detect_candidates``  finds, from data alone, every district that needs a
   'came from' fact: born after 2016, vanished before 2022, first seen late in
   the 2001-2014 file, or holding a commissionerate named after somewhere else.
2. ``propose``            asks the model, one call per state, to name the relationship
   for each candidate, with the evidence the tools found. Answers land in the
   decision record ``data/lineage.csv`` with ``verified = 0``. A human edits that
   file and sets ``verified = 1``. As in ``data/crosswalk.csv`` a re-run overwrites
   only its own earlier row with the same (parent, child, kind) key, never a row a
   human verified, wrote (``proposed_by = human``) or annotated (``reviewer_note``),
   and deletes nothing else: an earlier guess the human has not judged stays next
   to the new one. A candidate the model said nothing about gets a ``NOT PROPOSED``
   placeholder row, removed once a later run proposes an edge that answers it.
3. ``build_units``        joins districts connected by verified facts into groups
   (union-find) and writes ``data/stable_units.json`` plus a short report. A group
   is the smallest set of districts whose combined counts mean the same land in
   every year. Groups are never shown to users; they exist so that time series
   can be added up over the same land, and for tests.

A district's question is answered by a row where it is the child (where it came
from) or, for a district that vanished, where it is the parent of a merge, recode
or reported_together (where it went). A split or rename where it is only the
parent answers nothing about it, so it is still asked about.

Nodes are LGD district codes: ``data/lgd_districts.csv`` (all 785 codes of the
current boundaries, ``in_files`` = 1 when the code appears in the crime files)
when it exists, else the 754 codes of ``data/registry.json``. A verified edge that
names a code outside the node list cannot be used; ``build`` lists such edges in
the JSON meta and on screen instead of dropping them in silence.

    python -m agents.lineage candidates   # detected candidates by kind, no model
    python -m agents.lineage propose      # model proposals -> data/lineage.csv (verified rows untouched)
    python -m agents.lineage apply        # verified edges -> district_lineage (the districts table must know every code)
    python -m agents.lineage build        # verified edges -> data/stable_units.json + data/reports/<date>_stable_units.md
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Iterable, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field

from agents import store
from agents.prompts import PROJECT_CONTEXT
from data_ingestion.normalize import norm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REGISTRY_PATH = ROOT / "data" / "registry.json"
LGD_CSV = ROOT / "data" / "lgd_districts.csv"
OVERLAP_CSV = ROOT / "data" / "area_overlap.csv"
UNIT_MAP_CSV = ROOT / "data" / "unit_map.csv"
CSV_PATH = ROOT / "data" / "lineage.csv"
UNITS_PATH = ROOT / "data" / "stable_units.json"
REPORTS_DIR = ROOT / "data" / "reports"
RAW_2017 = RAW / "districtwise-crime-against-women-2017-onwards.csv"

CSV_COLUMNS = ["parent_code", "parent_name", "child_code", "child_name", "kind", "year", "evidence", "confidence",
               "reason", "proposed_by", "verified", "reviewer_note"]
KINDS = ("split", "merge", "recode", "rename", "reported_together")
PARENT_ANSWERS = ("merge", "recode", "reported_together")     # kinds in which the parent is the district that disappeared
KIND_HINTS = ("born_after_2014", "vanished", "born_in_era_a", "reported_together")
ERA_A_LATE_BIRTH = 2003         # a unit first seen in 2003 or later did not exist in 2001: it needs a parent
TOP_OVERLAPS = 3
NOT_PROPOSED = "NOT PROPOSED"
NON_CATEGORY_COLS = {"id", "year", "state_code", "district_code"}      # numeric columns of the 2017+ file that are not counts

# States whose districts moved between codes: a parent may sit in the other state.
SUCCESSOR_STATES: dict[int, int] = {36: 28, 28: 36, 37: 1, 1: 37}     # Telangana <-> Andhra Pradesh, Ladakh <-> J&K

Kind = Literal["split", "merge", "recode", "rename", "reported_together"]


class LineageEdge(BaseModel):
    parent_code: int
    child_code: int
    kind: Kind
    year: Optional[int] = None
    confidence: float = Field(ge=0, le=1)
    reason: str


class LineageBatch(BaseModel):
    items: list[LineageEdge]


# ---- inputs ------------------------------------------------------------------------------------

def load_registry(path: Path = REGISTRY_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_nodes(registry: dict, lgd_path: Path = LGD_CSV) -> list[dict]:
    """The node list: every LGD code from lgd_districts.csv when it exists, else the registry's districts."""
    if lgd_path.exists():
        with open(lgd_path, newline="", encoding="utf-8") as fh:
            return [{"district_code": int(r["district_code"]), "name": r["name"], "state_code": int(r["state_code"]),
                     "state_name": r["state_name"], "in_files": int(r["in_files"])} for r in csv.DictReader(fh)]
    state_names = {s["state_code"]: s["name"] for s in registry.get("states", [])}
    return [{"district_code": d["district_code"], "name": d["name"], "state_code": d["state_code"],
             "state_name": state_names.get(d["state_code"], str(d["state_code"])), "in_files": 1}
            for d in registry["districts"]]


def load_optional_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def totals_by_district(df: pd.DataFrame, year: int = 2022) -> dict[int, int]:
    """Sum of every category column for one year, per district code. Used only to name groups."""
    rows = df[df["year"] == year]
    cols = [c for c in rows.select_dtypes("number").columns if c not in NON_CATEGORY_COLS]
    summed = rows.groupby("district_code")[cols].sum().sum(axis=1)
    return {int(code): int(v) for code, v in summed.items()}


def load_totals_2022(path: Path = RAW_2017) -> dict[int, int]:
    return totals_by_district(pd.read_csv(path), 2022)


# ---- the decision record ----------------------------------------------------------------------------

def read_rows(path: Path) -> dict[tuple[str, str, str], dict]:
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        return {(r["parent_code"], r["child_code"], r["kind"]): r for r in csv.DictReader(fh)}


def _int_or(value, default: int | None = -1) -> int | None:
    """int('') and int(None) raise; the CSV stores a missing code or year as an empty cell."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def write_rows(path: Path, rows: dict[tuple[str, str, str], dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for key in sorted(rows, key=lambda k: (_int_or(k[1]), _int_or(k[0]), k[2])):
            writer.writerow({c: rows[key].get(c, "") for c in CSV_COLUMNS})


def _protected(row: dict | None) -> bool:
    """A row the pipeline never overwrites or removes: verified, written by a human, or carrying a reviewer note."""
    return bool(row) and (row.get("verified") == "1" or row.get("proposed_by") == "human" or bool(row.get("reviewer_note")))


def answered_codes(triples: Iterable[tuple[int, int, str]]) -> tuple[set[int], set[int]]:
    """From (parent_code, child_code, kind) triples: the codes answered as a child of any kind, and the codes
    answered as the parent of a merge, recode or reported_together. A missing code is negative and skipped."""
    as_child: set[int] = set()
    as_parent: set[int] = set()
    for parent, child, kind in triples:
        if child >= 0:
            as_child.add(child)
        if parent >= 0 and kind in PARENT_ANSWERS:
            as_parent.add(parent)
    return as_child, as_parent


def verified_triples(rows: dict) -> list[tuple[int, int, str]]:
    return [(_int_or(r["parent_code"]), _int_or(r["child_code"]), r.get("kind", ""))
            for r in rows.values() if r.get("verified") == "1"]


def is_answered(code: int, kind_hint: str, as_child: set[int], as_parent: set[int]) -> bool:
    """'vanished' asks where the district went, which a merge/recode/reported_together with it as parent answers.
    Every other hint asks where it came from: only a row with it as child answers that."""
    return code in as_child or (kind_hint == "vanished" and code in as_parent)


# ---- candidate detection (deterministic) -----------------------------------------------------------------

def _last_year(registry: dict) -> int:
    years = registry.get("meta", {}).get("years")
    return int(years[1]) if years else max(d["last_seen_year"] for d in registry["districts"])


def _overlap_evidence(overlap: pd.DataFrame | None, code: int) -> str:
    """The old polygons covering a modern code, largest share first: 'old_name (old_state): 0.62'."""
    if overlap is None or overlap.empty:
        return ""
    hits = overlap[overlap["new_code"] == code].sort_values("new_share", ascending=False).head(TOP_OVERLAPS)
    return "; ".join(f"{r.old_name} ({r.old_state}): {float(r.new_share):.2f}" for r in hits.itertuples())


def _contains_word(haystack: str, needle: str) -> bool:
    return f" {needle} " in f" {haystack} "


def _circle_matches(registry: dict, code: int, circle_norm: str) -> list[str]:
    """Other districts holding a circle with the same normalised name, or one that contains it as whole words."""
    out = []
    for d in registry["districts"]:
        if d["district_code"] == code:
            continue
        for c in d["circles"]:
            if c["norm"] == circle_norm or _contains_word(c["norm"], circle_norm) or _contains_word(circle_norm, c["norm"]):
                out.append(f"circle '{circle_norm}' also under {d['district_code']} {d['name']} as '{c['name']}' "
                           f"({c['first_seen_year']}-{c['last_seen_year']})")
    return out


def _era_a_first_years(unit_map: pd.DataFrame | None) -> dict[int, int]:
    """Earliest 2001-2014 year per resolved modern code (rows with a district_code)."""
    if unit_map is None or unit_map.empty:
        return {}
    um = unit_map.copy()
    for col in ("district_code", "first_year"):
        um[col] = pd.to_numeric(um[col], errors="coerce")
    um = um.dropna(subset=["district_code", "first_year"])
    first = um.groupby(um["district_code"].astype(int))["first_year"].min()
    return {int(code): int(y) for code, y in first.items()}


def _candidate(d: dict, kind_hint: str, year_hint: int | None, evidence: str) -> dict:
    return {"child_code": d["district_code"], "child_name": d["name"], "state_code": d["state_code"],
            "kind_hint": kind_hint, "year_hint": year_hint, "evidence": evidence}


def detect_candidates(registry: dict, lgd_list: list[dict], overlap: pd.DataFrame | None = None,
                      unit_map: pd.DataFrame | None = None, *, existing: dict | None = None) -> list[dict]:
    """Every district that needs a lineage fact, with the evidence the data offers.
    ``existing`` is the current lineage.csv (read_rows): a candidate whose question a verified row already
    answers (see ``is_answered``) is skipped."""
    last_year = _last_year(registry)
    era_a_first = _era_a_first_years(unit_map)
    out: list[dict] = []
    for d in registry["districts"]:
        code = d["district_code"]
        # Born after the 2001-14 era: first seen after 2016, or first seen in the 2016 file with no 2001-14
        # unit mapped to it (Hojai, Namsai: created 2015-16). The second test needs a unit map to be meaningful.
        born_late = d["first_seen_year"] > 2016 or (bool(era_a_first) and d["first_seen_year"] == 2016
                                                     and code not in era_a_first)
        if born_late:
            out.append(_candidate(d, "born_after_2014", d["first_seen_year"], _overlap_evidence(overlap, code)))
        if d["last_seen_year"] < last_year:
            lines = [f"last seen {d['last_seen_year']}, files run to {last_year}"]
            lines += [f"had circle '{c['name']}' ({c['first_seen_year']}-{c['last_seen_year']})" for c in d["circles"]]
            for c in d["circles"]:
                lines += _circle_matches(registry, code, c["norm"])
            out.append(_candidate(d, "vanished", d["last_seen_year"] + 1, "; ".join(lines)))
        if not born_late and era_a_first.get(code, 0) >= ERA_A_LATE_BIRTH:
            evidence = f"first seen in the 2001-2014 file in {era_a_first[code]}"
            geo = _overlap_evidence(overlap, code)
            out.append(_candidate(d, "born_in_era_a", era_a_first[code], f"{evidence}; {geo}" if geo else evidence))
        own_names = [norm(d["name"])] + list(d.get("aliases", []))
        for c in d["circles"]:
            if "commissionerate" in c["norm"] and not any(_contains_word(c["norm"], n) for n in own_names if n):
                out.append(_candidate(d, "reported_together", c["first_seen_year"],
                                      f"circle '{c['name']}' ({c['first_seen_year']}-{c['last_seen_year']}) is filed "
                                      f"under {code} {d['name']} but is not named after it"))
    as_child, as_parent = answered_codes(verified_triples(existing or {}))
    return [c for c in out if not is_answered(c["child_code"], c["kind_hint"], as_child, as_parent)]


# ---- model proposal ---------------------------------------------------------------------------------------

def _district_line(n: dict, registry_by_code: dict[int, dict]) -> str:
    d = registry_by_code.get(n["district_code"])
    seen = f"seen {d['first_seen_year']}-{d['last_seen_year']}" if d else "seen never"
    return f"  {n['district_code']} | {n['name']} | {seen} | in data files: {'yes' if n.get('in_files', 1) else 'no'}"


def build_prompt(state_name: str, state_code: int, districts: list[dict], candidates: list[dict],
                 registry_by_code: dict[int, dict], neighbour: tuple[str, list[dict]] | None = None) -> str:
    lines = [
        "Task: for each candidate district below, name the boundary or reporting change that links it to another "
        "district: which district it was carved out of, absorbed, replaced under a new code, or reports police data for.",
        "",
        f"State: {state_name} (code {state_code})",
        "",
        f"Districts of {state_name} (LGD code | name | years seen in the 2016-2022 files | in data files):",
        *[_district_line(n, registry_by_code) for n in districts],
    ]
    if neighbour:
        lines += ["", f"Districts of {neighbour[0]} (same territory before/after a state reorganisation; these codes are "
                      "also allowed):", *[_district_line(n, registry_by_code) for n in neighbour[1]]]
    lines += ["", "Candidates (code | name | hint | year hint | evidence from the data):"]
    for i, c in enumerate(candidates, 1):
        lines.append(f"{i}. {c['child_code']} | {c['child_name']} | hint: {c['kind_hint']} | "
                     f"year: {c['year_hint'] if c['year_hint'] is not None else 'unknown'} | "
                     f"evidence: {c['evidence'] or 'none'}")
    lines += [
        "",
        "Hints: 'born_after_2014' = the code first appears in the 2016+ files and no 2001-2014 unit maps to it, so it "
        "needs the parent(s) it was carved out of; "
        "'vanished' = the code stops appearing before 2022, so it was re-coded, merged or renamed; "
        "'born_in_era_a' = the unit first appears in the 2001-2014 file in the year given, so it needs a parent for the "
        "years before; 'reported_together' = the district files a police commissionerate named after somewhere else, "
        "which may pool other districts' data under it.",
        "",
        "Rules:",
        "1. For each candidate return one or more relationships: parent_code, child_code, kind, year, confidence (0-1), "
        "reason. The candidate may be the parent or the child, whichever the kind requires.",
        "2. kind is one of: 'split' = child was carved out of parent (several parents allowed, one row each); "
        "'merge' = child absorbed parent; 'recode' = the same district under a new code (parent = old code, child = new "
        "code); 'rename' = same code, new name (parent_code = child_code); 'reported_together' = the police data of "
        "parent is filed under child in the 2017+ files.",
        "3. parent_code and child_code must both come from the lists above. Never invent a code.",
        "4. For a 'reported_together' candidate say which districts of the list (including those not in the data "
        "files) are reported under it, one row per district.",
        "5. year = the year the change took effect (the first year the child appears, or the first year the parent is "
        "missing); null when unknown.",
        "6. If unsure, still answer with confidence below 0.5 and say why in the reason, rather than inventing. If "
        "there is truly no relationship, return no row for that candidate: it is then flagged for human review.",
        "7. reason: one line naming the evidence (a known reorganisation, a notification year, the evidence lines above).",
    ]
    return "\n".join(lines)


def _allowed_codes(state_code: int, lgd_list: list[dict]) -> set[int]:
    """Exactly the codes the prompt listed: the state's own and its successor/predecessor state's."""
    states = {state_code, SUCCESSOR_STATES.get(state_code, state_code)}
    return {n["district_code"] for n in lgd_list if n["state_code"] in states}


def _escalate(conn, edge: LineageEdge, state_code: int) -> None:
    store.add_escalation(conn, kind="unit", ref={"child_code": edge.child_code, "parent_code": edge.parent_code},
                         violations=[{"rule": "lineage_code_not_in_state",
                                      "message": f"the model proposed {edge.kind} {edge.parent_code} -> {edge.child_code} "
                                                 f"for state {state_code} but a code is not in that state's list"}])


def _write_placeholders(rows: dict, cands: list[dict], as_child: set[int], as_parent: set[int]) -> int:
    """One NOT PROPOSED row per candidate code with a hint no kept edge answers, naming every open hint. A
    placeholder left by an earlier run goes away once the code is answered. Returns the number written."""
    by_code: dict[int, list[dict]] = defaultdict(list)
    for c in cands:
        by_code[c["child_code"]].append(c)
    written = 0
    for code, hints in by_code.items():
        key = ("", str(code), "")
        if _protected(rows.get(key)):
            continue
        open_hints = [c for c in hints if not is_answered(code, c["kind_hint"], as_child, as_parent)]
        if not open_hints:
            rows.pop(key, None)
            continue
        years = [c["year_hint"] for c in open_hints if c["year_hint"] is not None]
        rows[key] = {"parent_code": "", "parent_name": "", "child_code": str(code), "child_name": hints[0]["child_name"],
                     "kind": "", "year": str(min(years)) if years else "",
                     "evidence": " | ".join(dict.fromkeys(c["evidence"] for c in open_hints if c["evidence"])),
                     "confidence": "",
                     "reason": f"{NOT_PROPOSED}: hints {', '.join(c['kind_hint'] for c in open_hints)}; "
                               "the model returned no relationship",
                     "proposed_by": "rule", "verified": "0", "reviewer_note": ""}
        written += 1
    return written


def propose(client, candidates: list[dict], registry: dict, lgd_list: list[dict], *, csv_path: Path = CSV_PATH,
            conn=None) -> dict:
    """One model call per state that has candidates. Edges land in the CSV with verified = 0, overwriting only an
    unprotected row with the same key (see ``_protected``). A candidate no kept edge answers gets a NOT PROPOSED
    placeholder so the human sees it. An edge whose codes are not in the state's lists is dropped and escalated
    (counted only when ``conn`` is None)."""
    rows = read_rows(csv_path)
    registry_by_code = {d["district_code"]: d for d in registry["districts"]}
    names = {d["district_code"]: d["name"] for d in registry["districts"]} | {n["district_code"]: n["name"] for n in lgd_list}
    state_names = {s["state_code"]: s["name"] for s in registry.get("states", [])}
    state_names |= {n["state_code"]: n["state_name"] for n in lgd_list if n.get("state_name")}
    by_state: dict[int, list[dict]] = defaultdict(list)
    evidence_by_code: dict[int, list[str]] = defaultdict(list)
    for c in candidates:
        by_state[c["state_code"]].append(c)
        if c["evidence"]:
            evidence_by_code[c["child_code"]].append(c["evidence"])

    summary: dict = {"by_state": {}, "escalations": 0}
    kept_all: list[LineageEdge] = []
    for state_code in sorted(by_state):
        cands = by_state[state_code]
        districts = [n for n in lgd_list if n["state_code"] == state_code]
        neighbour = None
        if state_code in SUCCESSOR_STATES:
            other = SUCCESSOR_STATES[state_code]
            neighbour = (state_names.get(other, str(other)), [n for n in lgd_list if n["state_code"] == other])
        prompt = build_prompt(state_names.get(state_code, str(state_code)), state_code, districts, cands,
                              registry_by_code, neighbour)
        result = client.complete(prompt, LineageBatch, tag=f"lineage:{state_code}", system=PROJECT_CONTEXT)

        allowed = _allowed_codes(state_code, lgd_list)
        kept, dropped = [], 0
        for e in result.parsed.items:
            if e.parent_code in allowed and e.child_code in allowed:
                kept.append(e)
            else:
                dropped += 1
                if conn is not None:
                    _escalate(conn, e, state_code)
        summary["escalations"] += dropped

        written = 0
        for e in kept:
            key = (str(e.parent_code), str(e.child_code), e.kind)
            if _protected(rows.get(key)):
                continue
            evidence = " | ".join(dict.fromkeys(evidence_by_code[e.parent_code] + evidence_by_code[e.child_code]))
            rows[key] = {"parent_code": str(e.parent_code), "parent_name": names.get(e.parent_code, ""),
                         "child_code": str(e.child_code), "child_name": names.get(e.child_code, ""),
                         "kind": e.kind, "year": "" if e.year is None else str(e.year), "evidence": evidence,
                         "confidence": f"{e.confidence:.2f}", "reason": e.reason, "proposed_by": f"llm:{result.model}",
                         "verified": "0", "reviewer_note": ""}
            written += 1
        kept_all += kept
        summary["by_state"][state_code] = {"candidates": len(cands), "edges": written, "dropped": dropped,
                                           "placeholders": 0, "provider": result.provider, "model": result.model,
                                           "cached": result.cached, "tries": result.tries,
                                           "prompt_tokens": result.prompt_tokens,
                                           "completion_tokens": result.completion_tokens}

    # Placeholders wait until every state has answered: one state's batch may answer a candidate of its
    # successor state (Telangana <-> Andhra Pradesh, Ladakh <-> J&K), which the guard allows.
    as_child, as_parent = answered_codes((e.parent_code, e.child_code, e.kind) for e in kept_all)
    for state_code, cands in by_state.items():
        summary["by_state"][state_code]["placeholders"] = _write_placeholders(rows, cands, as_child, as_parent)
    write_rows(csv_path, rows)
    return summary


# ---- into the store ---------------------------------------------------------------------------------------

def apply(conn, *, csv_path: Path = CSV_PATH) -> dict:
    """Load verified edges into district_lineage. Unverified rows and placeholders stay out. A code the
    districts table does not know fails with a message naming the row, not a bare constraint error."""
    known = {row[0] for row in conn.execute("SELECT district_code FROM districts")}
    loaded = pending = 0
    for (p, c, kind), r in read_rows(csv_path).items():
        if r.get("verified") != "1":
            pending += 1
            continue
        if kind not in KINDS:
            raise ValueError(f"{p} -> {c}: kind '{kind}' is not one of {KINDS}")
        parent, child = int(p), int(c)
        for code in (parent, child):
            if code not in known:
                raise ValueError(f"{parent} -> {child} ({kind}): district_code {code} is not in the districts table; "
                                 "run `python -m data_ingestion.registry` (it loads the map-only codes from data/lgd_districts.csv)")
        year = int(r["year"]) if r.get("year", "").strip() else None
        confidence = float(r["confidence"]) if r.get("confidence", "").strip() else None
        conn.execute("INSERT OR REPLACE INTO district_lineage (parent_code, child_code, kind, year, confidence, reason, "
                     "source, verified) VALUES (?, ?, ?, ?, ?, ?, ?, 1)",
                     (parent, child, kind, year, confidence, r.get("reason", ""), r.get("proposed_by", "")))
        loaded += 1
    conn.commit()
    return {"loaded": loaded, "still_unverified": pending}


# ---- stable units (union-find over verified edges) -------------------------------------------------------

def _usable(e: dict) -> bool:
    return e.get("verified") == "1" and e.get("kind") in KINDS


def _edge(e: dict) -> dict:
    return {"parent_code": _int_or(e.get("parent_code")), "child_code": _int_or(e.get("child_code")),
            "kind": e["kind"], "year": _int_or(e.get("year"), None)}


def unusable_edges(edges: list[dict], nodes: list[dict]) -> list[dict]:
    """Verified edges of a known kind naming a code outside the node list: build_units cannot use them. In the
    registry fallback (no lgd_districts.csv) that is every edge to a GeoJSON-only code, so ``build`` shows them."""
    codes = {int(n["district_code"]) for n in nodes}
    return [u for u in map(_edge, filter(_usable, edges)) if not {u["parent_code"], u["child_code"]} <= codes]


def build_units(edges: list[dict], nodes: list[dict], totals_2022: dict[int, int]) -> list[dict]:
    """Groups of districts connected by verified edges of a known kind. Every node lands in exactly one
    group; a lone district is a group of one. The group is named after the member with the largest 2022
    total (ties and all-zero: the smallest code). Edges naming a code outside ``nodes`` are left out;
    ``unusable_edges`` lists them."""
    codes = sorted({int(n["district_code"]) for n in nodes})
    names = {int(n["district_code"]): n["name"] for n in nodes}
    parent = {c: c for c in codes}

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:                     # path compression: every visited node now points at the root
            parent[x], x = root, parent[x]
        return root

    used = []
    for u in map(_edge, filter(_usable, edges)):
        if u["parent_code"] not in parent or u["child_code"] not in parent:
            continue
        parent[find(u["parent_code"])] = find(u["child_code"])
        used.append(u)

    members: dict[int, list[int]] = defaultdict(list)
    for c in codes:
        members[find(c)].append(c)
    units = []
    for group in members.values():
        group.sort()
        largest = min(group, key=lambda c: (-totals_2022.get(c, 0), c))
        in_group = set(group)
        units.append({"unit_id": f"su_{group[0]}", "name": names[largest], "largest_member_code": largest,
                      "members": group, "edges": [u for u in used if u["parent_code"] in in_group]})
    return sorted(units, key=lambda u: u["members"][0])


def write_stable_units(units: list[dict], path: Path = UNITS_PATH, skipped: list[dict] = ()) -> dict:
    meta = {"generated": date.today().isoformat(), "n_nodes": sum(len(u["members"]) for u in units),
            "n_groups": len(units), "n_edges_used": sum(len(u["edges"]) for u in units),
            "largest_group_size": max((len(u["members"]) for u in units), default=0),
            "n_edges_skipped": len(skipped), "skipped_edges": list(skipped)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta, "units": units}, indent=1, ensure_ascii=False), encoding="utf-8")
    return meta


SIZE_BANDS = (("1", 1, 1), ("2", 2, 2), ("3-5", 3, 5), ("6-10", 6, 10), ("11+", 11, 10**9))


def size_distribution(units: list[dict]) -> dict[str, int]:
    sizes = [len(u["members"]) for u in units]
    return {label: sum(lo <= s <= hi for s in sizes) for label, lo, hi in SIZE_BANDS}


def write_report(units: list[dict], path: Path, nodes: list[dict] = ()) -> str:
    """Markdown summary of the groups: size distribution, the ten largest, GeoJSON-only members."""
    names = {int(n["district_code"]): n["name"] for n in nodes}
    geo_only = {int(n["district_code"]) for n in nodes if not int(n.get("in_files", 1))}
    n_edges = sum(len(u["edges"]) for u in units)
    lines = [f"# Stable units ({date.today().isoformat()})", "",
             f"{len(units)} groups over {sum(len(u['members']) for u in units)} districts, "
             f"built from {n_edges} verified edges.", "",
             "## Size distribution", "", "| size | groups |", "|---|---|"]
    lines += [f"| {label} | {n} |" for label, n in size_distribution(units).items()]
    lines += ["", "## Ten largest groups", ""]
    for u in sorted(units, key=lambda u: (-len(u["members"]), u["members"][0]))[:10]:
        listed = ", ".join(f"{c} {names.get(c, '')}".strip() for c in u["members"])
        lines.append(f"- {u['unit_id']} {u['name']} ({len(u['members'])} members): {listed}")
    in_groups = [c for u in units for c in u["members"] if c in geo_only]
    lines += ["", "## GeoJSON-only members", "",
              f"{len(in_groups)} member(s) never appear in the crime files (in_files = 0)"
              + (": " + ", ".join(f"{c} {names[c]}" for c in in_groups) if in_groups else ".")]
    text = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


# ---- CLI ---------------------------------------------------------------------------------------------------

def _load_inputs() -> tuple[dict, list[dict], pd.DataFrame | None, pd.DataFrame | None]:
    registry = load_registry()
    return registry, load_nodes(registry), load_optional_csv(OVERLAP_CSV), load_optional_csv(UNIT_MAP_CSV)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="District lineage: detect candidates, propose with the model, build stable units.")
    ap.add_argument("command", choices=["candidates", "propose", "apply", "build"])
    args = ap.parse_args(argv)

    if args.command == "candidates":
        registry, nodes, overlap, unit_map = _load_inputs()
        cands = detect_candidates(registry, nodes, overlap, unit_map, existing=read_rows(CSV_PATH))
        print(f"{len(cands)} candidates (nodes: {len(nodes)}, overlap: {'yes' if overlap is not None else 'no'}, "
              f"unit_map: {'yes' if unit_map is not None else 'no'})")
        for hint in KIND_HINTS:
            group = [c for c in cands if c["kind_hint"] == hint]
            print(f"\n{hint}: {len(group)}")
            for c in group[:5]:
                print(f"   {c['child_code']:>5} {c['child_name']:<28} state {c['state_code']:<3} year {c['year_hint']}  "
                      f"{c['evidence'][:100]}")
    elif args.command == "propose":
        from agents.llm_client import build_client
        registry, nodes, overlap, unit_map = _load_inputs()
        conn = store.connect()
        cands = detect_candidates(registry, nodes, overlap, unit_map, existing=read_rows(CSV_PATH))
        summary = propose(build_client(conn), cands, registry, nodes, csv_path=CSV_PATH, conn=conn)
        for state_code, s in summary["by_state"].items():
            print(f"state {state_code}: {s}")
        print(f"\nescalations (edges with a code outside the state): {summary['escalations']}")
        print(f"review file -> {CSV_PATH}")
    elif args.command == "apply":
        print(apply(store.connect()))
    else:
        registry = load_registry()
        nodes = load_nodes(registry)
        edges = list(read_rows(CSV_PATH).values())
        units = build_units(edges, nodes, load_totals_2022())
        skipped = unusable_edges(edges, nodes)
        meta = write_stable_units(units, UNITS_PATH, skipped)
        report = REPORTS_DIR / f"{date.today().isoformat()}_stable_units.md"
        write_report(units, report, nodes)
        print(f"stable units: {meta}")
        for label, n in size_distribution(units).items():
            print(f"  size {label:<5} {n:>5} groups")
        if skipped:
            print(f"\nWARNING: {len(skipped)} verified edge(s) name a code outside the {len(nodes)} nodes and were not used"
                  + (" (no data/lgd_districts.csv: GeoJSON-only codes are unknown here):" if not LGD_CSV.exists() else ":"))
            for u in skipped:
                print(f"   {u['parent_code']} -> {u['child_code']} {u['kind']} {u['year'] or ''}")
        print(f"written -> {UNITS_PATH}\nreport  -> {report}")


if __name__ == "__main__":
    main()
