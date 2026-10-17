"""Resolve every 2001-2014 police unit name to an LGD district code, or say why it cannot have one.

The 2001-2014 file carries no codes. Its DISTRICT column is a police reporting
unit spelled however that year's clerk spelled it: "Cuddapah" for Y.S.R.,
"Guntakal Rly." for the railway unit inside Anantapur, "CAW" for a cell with no
geography at all. Everything downstream needs a district code on each row, so
this module decides each distinct (state, unit) pair once and writes the
decision to ``data/unit_map.csv``, the human's decision record.

Stages are tried in order; the first hit wins and every row records which stage
decided it and how sure it is. Cheap deterministic stages go first so the model
is only asked about what they cannot do:

    carried      a human already verified this exact (state, unit) spelling in another
                 era (the 2015 workbook reuses 2001-2014 names): the same decision is
                 written, stage 'human', proposed_by 'carried:<era>', verified = 1
    exact        the unit is a known spelling of a district in the registry
    circle       the unit is a police circle the 2016+ files list under a district
    rule         strip 'rly', 'commr', 'city', 'urban', ... and look again (patna rly -> Patna)
    successor    any of the above, but the district now belongs to a state carved out
                 later (Telangana from Andhra Pradesh, Ladakh from Jammu and Kashmir)
    (not a district)  a police unit with no geography: CAW, STF, CB-CID, crime branch...
    fuzzy        rapidfuzz WRatio >= 90 with a clear margin over the next-best district
    geometry     the old boundary map says which modern district the unit's polygon is in
    llm          one model call per state for what is left, with that state's candidate
                 list and the fuzzy hints in the prompt; a code outside the candidate
                 list is rejected, never trusted

The not-a-district check runs before fuzzy on purpose: 'crime jammu' scores 90
against 'jammu', and a crime branch is not a district however close the spelling.
An ambiguous fuzzy result (two districts within 5 points) is not a hit: the chain
continues, and if nothing else decides, the model sees the unit with its hints.

A spelling the registry lists under two districts of one state ('grp' is a circle of
both Lucknow and Dehradun in Uttar Pradesh's pool) is never an identity match: the
exact, circle and rule stages skip it, fuzzy gives every district that claims it the
same score (so the margin fails), and the model gets all of them as hints.

Auto-verify policy: exact, circle, rule and successor are identity matches on
spellings the registry already knows, so they are written with verified = 1. The one
exception is an identity match whose name also looks like a geography-less police
unit ('cyber cell' listed as a circle under one district): it keeps that district as
the proposal but is held for a human (verified = 0). A fuzzy or model answer whose
score reaches its stage's cut in ACCEPT_WITHOUT_REVIEW is accepted too (verified = 1),
with a reviewer_note that says so, so it is never mistaken for a human's check; the
cuts come from the calibration in the eval reports. Everything else is written with
verified = 0 for a human. A verified row is never overwritten and stays in the file
even when the input no longer produces its (state, unit) key; the pipeline writes
reviewer_note only on a row it accepts without review, and only when it is empty.

    python -m agents.resolver run [--llm]   # build/refresh data/unit_map.csv (model stage only with --llm)
    python -m agents.resolver report        # stage/kind tables + unresolved list -> data/reports/<date>_resolver.md
    python -m agents.resolver apply         # verified rows -> unit_map table
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, get_args

import pandas as pd
from pydantic import BaseModel, Field
from rapidfuzz import fuzz, process

from agents import store
from agents.llm_client import build_client
from agents.prompts import PROJECT_CONTEXT
from data_ingestion import ncrb_xlsx
from data_ingestion.normalize import norm, norm_state

ROOT = Path(__file__).resolve().parents[1]
ERA_A_PATH = ROOT / "data" / "raw" / "crimes_against_women_2001-2014.csv"
REGISTRY_PATH = ROOT / "data" / "registry.json"
OVERLAP_PATH = ROOT / "data" / "area_overlap.csv"
CSV_PATH = ROOT / "data" / "unit_map.csv"
REPORTS_DIR = ROOT / "data" / "reports"
ERA = "2001_2014"
NAME_KEYED_ERAS = ("2001_2014", "2015")                # the files without codes: what this module resolves
ERA_2015_PATH = ROOT / "data" / "raw" / "ncrb_district_wise_crimes_against_women_2015.xlsx"

CSV_COLUMNS = ["era", "state", "state_code", "unit", "first_year", "last_year", "n_years", "district_code",
               "district_name", "kind", "stage", "confidence", "reason", "candidates", "geo_code", "geo_share",
               "proposed_by", "verified", "reviewer_note"]

Kind = Literal["district", "circle", "railway", "commissionerate", "headquarters_town", "rename", "not_a_district",
               "unknown"]
KINDS: tuple[str, ...] = get_args(Kind)
STAGES = ("exact", "circle", "rule", "successor", "fuzzy", "geometry", "llm", "human", "unresolved")
AUTO_VERIFIED_STAGES = ("exact", "circle", "rule", "successor")

# Scored answers accepted without a human: at or above its stage's cut, a fuzzy or model answer is written verified
# with ACCEPTED_NOTE in reviewer_note; below it, it waits for review. Set from the calibration table of
# evals/reports/2026-10-17.md by the rule in evals.metrics.pick_cut (the lowest cut with at least 20 reviewed answers
# at or above it, at least 95% of them right): the model was right 49 of 51 times at 0.90 and above, 9 of 13 in
# [0.80, 0.90); fuzzy 35 of 35, the lowest at 0.90. Geometry has 7 reviewed answers, too few for a cut, so it stays
# with a human. Every eval report recomputes the rule's cut and says whether these still make the same calls.
# The model's cut was kept at 0.90 on 2026-10-18 after the leave-one-out checks in evals/reports/2026-10-18.md: in 3
# of 66 runs the rule picks 0.85, and answers graded by a cut chosen without them are right 49 of 52 times (94.2%).
LLM_ACCEPT_CONFIDENCE = 0.90
FUZZY_ACCEPT_SCORE = 0.90          # WRatio / 100, the scale the decision record stores
ACCEPT_WITHOUT_REVIEW = {"llm": LLM_ACCEPT_CONFIDENCE, "fuzzy": FUZZY_ACCEPT_SCORE}
ACCEPTED_NOTE = "Accepted without review"
HELD_FOR_REVIEW = "held for review"   # in the reason of an identity match the pipeline left to a human

# Era A spellings the registry's aliases do not carry, plus the old boundary map's
# pre-rename state names (the overlap file may spell states either way).
STATE_ALIASES: dict[str, str] = {
    "a n islands": "andaman and nicobar islands", "d n haveli": "dadra and nagar haveli",
    "daman diu": "daman and diu", "jammu kashmir": "jammu and kashmir",
    "orissa": "odisha", "uttaranchal": "uttarakhand", "pondicherry": "puducherry",
    "andaman and nicobar": "andaman and nicobar islands",
}
STATE_FUZZY_ACCEPT = 88

# States carved out after 2014: 2001-2013 rows list their districts under the old state.
SUCCESSOR_STATES: dict[int, tuple[int, ...]] = {28: (36,), 1: (37,)}     # AP -> Telangana, J&K -> Ladakh

RAILWAY_WORDS = ("rly", "railway", "railways", "grp", "g r p")
COMMISSIONERATE_WORDS = ("commr", "commissionerate", "city")
AREA_WORDS = ("urban", "rural", "north", "south", "east", "west", "central")
EDGE_WORDS = frozenset(RAILWAY_WORDS + COMMISSIONERATE_WORDS + AREA_WORDS)
RAILWAY_RE = re.compile(r"\b(rly|railway|railways|grp|g r p)\b")
NOT_A_DISTRICT_RE = re.compile(r"\b(caw|stf|cid|cbcid|crime branch|crime|cyber|eow|vigilance|women cell|mahila)\b")
NOT_A_DISTRICT_REASON = "police unit with no district geography"

FUZZY_ACCEPT = 90      # WRatio a best candidate must reach ...
FUZZY_MARGIN = 5       # ... and lead the next-best *district* by
GEO_ACCEPT = 0.5       # share of the old polygon inside one modern district


class UnmappedState(ValueError):
    """An Era A state name that no registry state matches."""


# ---- inputs -------------------------------------------------------------------------------

def load_units(era: str = ERA, path: Path | None = None) -> pd.DataFrame:
    """Distinct (state, unit) pairs of one code-less file with the years each appears in: the 2001-2014
    CSV (state-total rows dropped: they are sums, not units) or NCRB's 2015 workbook."""
    if era == "2015":
        rows = ncrb_xlsx.read_district_rows(path or ERA_2015_PATH)
        df = pd.DataFrame({"state": [norm_state(r["state_name"]) for r in rows], "unit": [norm(r["district"]) for r in rows],
                           "Year": [int(r["YEAR"]) for r in rows]})
    else:
        df = pd.read_csv(path or ERA_A_PATH, usecols=["STATE/UT", "DISTRICT", "Year"])
        df = df[~df["DISTRICT"].str.upper().str.contains("TOTAL", na=False)]
        df = df.assign(state=df["STATE/UT"].map(norm_state), unit=df["DISTRICT"].map(norm))
    g = df.groupby(["state", "unit"])["Year"].agg(first_year="min", last_year="max", n_years="nunique")
    return g.reset_index().sort_values(["state", "unit"]).reset_index(drop=True)


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_state_name(name: str, registry: dict) -> int:
    """Era A state name -> state_code: manual aliases, then the registry's aliases, then fuzzy as a last resort."""
    key = norm_state(name)
    key = STATE_ALIASES.get(key, key)
    by_alias = {a: s["state_code"] for s in registry["states"] for a in [*s["aliases"], norm_state(s["name"])]}
    if key in by_alias:
        return by_alias[key]
    best = process.extractOne(key, list(by_alias), scorer=fuzz.WRatio, score_cutoff=STATE_FUZZY_ACCEPT)
    if best:
        return by_alias[best[0]]
    raise UnmappedState(f"state '{name}' matches no registry state")


def map_states(names, registry: dict) -> dict[str, int]:
    """Every Era A state name -> code, or one error naming all the names that failed."""
    codes, failed = {}, []
    for n in names:
        try:
            codes[n] = resolve_state_name(n, registry)
        except UnmappedState:
            failed.append(n)
    if failed:
        raise UnmappedState(f"Era A state(s) not in the registry: {failed}. Add an alias to STATE_ALIASES.")
    return codes


def load_overlap(path: Path, registry: dict, state_code: dict[str, int]) -> pd.DataFrame | None:
    """The old-map overlap table (optional). Its old_state_norm is rewritten to the Era A
    spelling of the same state so the geometry stage can match (state, unit) exactly."""
    path = Path(path)
    if not path.exists():
        return None
    ov = pd.read_csv(path)
    era_a_name = {code: name for name, code in state_code.items()}

    def to_era_a(s: str) -> str:
        try:
            return era_a_name.get(resolve_state_name(s, registry), s)
        except UnmappedState:
            return s

    spelling = {s: to_era_a(s) for s in ov["old_state_norm"].dropna().unique()}
    ov["old_state_norm"] = ov["old_state_norm"].map(spelling)
    return ov


# ---- candidate pools ------------------------------------------------------------------------

SOURCE_RANK = ("alias", "circle")      # a district's own name beats another district's circle spelled the same


class Pool(dict):
    """One state's candidate names: name_norm -> (district_code, source), source 'alias' or 'circle'.

    Only a name that identifies exactly one district is in the dict. A spelling that two
    districts claim at the same rank ('grp' is a circle of Dehradun and of Lucknow) goes
    to ``ambiguous`` (name -> codes) instead, so no stage can treat it as an identity
    match; the fuzzy hints and the model still see every district that claims it.

    Also what the stages need to label a hit: ``state_code`` (the Era A state),
    ``district_name`` (code -> current name, for hints and reasons) and
    ``district_state`` (code -> the state the district belongs to today, which differs
    from ``state_code`` for successor-state districts)."""

    def __init__(self, state_code: int):
        super().__init__()
        self.state_code = state_code
        self.district_name: dict[int, str] = {}
        self.district_state: dict[int, int] = {}
        self.claims: dict[str, dict[str, set[int]]] = {}       # name -> source -> codes that claim it
        self.ambiguous: dict[str, set[int]] = {}

    def add(self, district: dict, names: list[str], source: str) -> None:
        code = district["district_code"]
        self.district_name[code] = district["name"]
        self.district_state[code] = district["state_code"]
        for n in names:
            self._claim(n, source, {code})

    def absorb(self, other: "Pool") -> None:
        """Take a successor state's names too, under the same ranking: a district's own name
        beats a circle, and two districts' own names for one spelling make it ambiguous."""
        self.district_name.update(other.district_name)
        self.district_state.update(other.district_state)
        for n, by_source in other.claims.items():
            for source, codes in by_source.items():
                self._claim(n, source, codes)

    def _claim(self, name: str, source: str, codes: set[int]) -> None:
        by_source = self.claims.setdefault(name, {})
        by_source.setdefault(source, set()).update(codes)
        rank = next(r for r in SOURCE_RANK if by_source.get(r))     # the best-ranked source decides
        winners = by_source[rank]
        if len(winners) == 1:
            self[name] = (next(iter(winners)), rank)
            self.ambiguous.pop(name, None)
        else:
            self.pop(name, None)
            self.ambiguous[name] = set(winners)

    def codes_for(self, name: str) -> set[int]:
        return self.ambiguous.get(name) or {self[name][0]}

    @property
    def names(self) -> list[str]:
        """Every spelling the pool knows, identifying or ambiguous (what fuzzy scores against)."""
        return [*self, *self.ambiguous]

    @property
    def codes(self) -> set[int]:
        return set(self.district_name)


def build_pools(registry: dict) -> dict[int, Pool]:
    """One pool per state from every district seen under it, plus successor states' districts."""
    pools = {s["state_code"]: Pool(s["state_code"]) for s in registry["states"]}
    for source in SOURCE_RANK:
        for d in registry["districts"]:
            names = d["aliases"] if source == "alias" else [c["norm"] for c in d["circles"]]
            for sc in d["state_codes_seen"]:
                pools.setdefault(sc, Pool(sc)).add(d, names, source)
    for old, successors in SUCCESSOR_STATES.items():
        for succ in successors:
            if old in pools and succ in pools:
                pools[old].absorb(pools[succ])
    return pools


# ---- the deterministic stages ---------------------------------------------------------------------

Hint = tuple[int, str, float]      # (district_code, current name, WRatio score)


@dataclass
class Resolution:
    district_code: Optional[int]
    kind: str
    stage: str
    confidence: float
    reason: str
    candidates: list[Hint] = field(default_factory=list)
    geo_code: Optional[int] = None
    geo_share: Optional[float] = None
    proposed_by: str = "none"
    needs_review: bool = False      # an identity match a human must still confirm (never auto-verified)


def spellings(unit: str) -> list[tuple[str, list[str]]]:
    """The unit, then what is left after each edge word comes off, trailing words first:
    'east godavari rly' -> [('east godavari rly', []), ('east godavari', ['rly']), ('godavari', ['rly', 'east'])].
    One word at a time, so a direction word that is part of the name survives when the
    lookup already hits without it. 'g r p patna' -> ..., ('patna', ['grp'])."""
    tokens = re.sub(r"\bg r p\b", "grp", unit).split()
    out, removed = [(unit, [])], []
    while tokens and tokens[-1] in EDGE_WORDS:
        removed.append(tokens.pop())
        out.append((" ".join(tokens), list(removed)))
    while tokens and tokens[0] in EDGE_WORDS:
        removed.append(tokens.pop(0))
        out.append((" ".join(tokens), list(removed)))
    return out


def _known_spelling(unit: str, pool: Pool) -> Resolution | None:
    """Stages exact, circle and rule: the unit, or what is left of it after edge words, is a
    spelling the pool knows. A spelling several districts claim stops the search: it is not
    an identity match, whatever else could still be stripped off. A hit on a district that
    now sits in another state is 'successor'."""
    for spelling, removed in spellings(unit):
        if spelling in pool.ambiguous:
            return None
        if spelling in pool:
            break
    else:
        return None
    code, source = pool[spelling]
    name = pool.district_name[code]
    if removed:
        kind = ("railway" if any(w in RAILWAY_WORDS for w in removed)
                else "commissionerate" if any(w in COMMISSIONERATE_WORDS for w in removed) else "district")
        res = Resolution(code, kind, "rule", 1.0,
                         f"'{unit}' without '{' '.join(removed)}' is a {'spelling' if source == 'alias' else 'circle'} of {name}")
    elif source == "alias":
        res = Resolution(code, "district", "exact", 1.0, f"'{unit}' is a spelling of {name}")
    else:
        res = Resolution(code, "railway" if RAILWAY_RE.search(unit) else "circle", "circle", 1.0,
                         f"'{unit}' is a police circle listed under {name} in 2016+")
    if pool.district_state[code] != pool.state_code:
        res.stage = "successor"
        res.reason += f" (now in state {pool.district_state[code]})"
    if NOT_A_DISTRICT_RE.search(unit):      # the spelling is known, but a cyber cell is not a district's crime
        res.needs_review = True
        res.reason += f"; {HELD_FOR_REVIEW}: the name looks like a police unit with no district geography"
    res.proposed_by = "rule"
    return res


def _not_a_district(unit: str) -> Resolution | None:
    if NOT_A_DISTRICT_RE.search(unit):
        return Resolution(None, "not_a_district", "unresolved", 1.0, NOT_A_DISTRICT_REASON, proposed_by="rule")
    return None


def fuzzy_hints(unit: str, pool: Pool, n: int = 3) -> list[Hint]:
    """The n closest districts by WRatio, one entry per district code (two spellings of the
    same district are one candidate, not two competing ones)."""
    best: dict[int, float] = {}
    for name, score, _ in process.extract(unit, pool.names, scorer=fuzz.WRatio, limit=None):
        for code in pool.codes_for(name):       # an ambiguous spelling scores for every district that claims it
            if score > best.get(code, -1.0):
                best[code] = score
    top = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
    return [(code, pool.district_name[code], round(score, 1)) for code, score in top]


def _fuzzy(unit: str, hints: list[Hint]) -> Resolution | None:
    if not hints or hints[0][2] < FUZZY_ACCEPT:
        return None
    code, name, best = hints[0]
    second = hints[1][2] if len(hints) > 1 else 0.0
    if best - second < FUZZY_MARGIN:
        return None                                   # ambiguous: not a hit, the model sees the hints
    kind = "railway" if RAILWAY_RE.search(unit) else "district"
    return Resolution(code, kind, "fuzzy", best / 100, f"fuzzy {best:g} vs next district {second:g}: '{unit}' ~ {name}",
                      proposed_by="fuzzy")


def geo_match(state: str, unit: str, overlap: pd.DataFrame | None) -> tuple[int, float] | None:
    """The modern district the unit's old-map polygon lies in: the one named like the unit
    if there is one, else the one holding the largest share of the old polygon."""
    if overlap is None or overlap.empty:
        return None
    rows = overlap[(overlap["old_state_norm"] == state) & (overlap["old_norm"] == unit)]
    if rows.empty:
        return None
    named = rows[rows["new_name"].map(norm) == unit]
    pick = (named if not named.empty else rows).sort_values("share", ascending=False).iloc[0]
    return int(pick["new_code"]), min(float(pick["share"]), 1.0)


def _geometry(unit: str, geo: tuple[int, float] | None, pool: Pool) -> Resolution | None:
    if geo is None:
        return None
    code, share = geo
    if share < GEO_ACCEPT or code not in pool.district_name:     # a code outside the registry cannot be used
        return None
    return Resolution(code, "district", "geometry", share,
                      f"old-map polygon '{unit}' lies {share:.0%} inside {pool.district_name[code]}", proposed_by="geometry")


def _pending(unit: str, hints: list[Hint], pool: Pool) -> Resolution:
    listed = next((pool.ambiguous[s] for s, _ in spellings(unit) if s in pool.ambiguous), None)
    if listed:
        reason = (f"'{unit}' is listed under {len(listed)} districts: "
                  + ", ".join(f"{c} {pool.district_name[c]}" for c in sorted(listed)))
    elif len(hints) > 1 and hints[0][2] >= FUZZY_ACCEPT:
        reason = f"ambiguous fuzzy match: {hints[0][1]} {hints[0][2]:g} vs {hints[1][1]} {hints[1][2]:g}"
    else:
        reason = "no deterministic match"
    return Resolution(None, "unknown", "unresolved", 0.0, reason)


def resolve_free(state: str, unit: str, pool: Pool, overlap: pd.DataFrame | None = None) -> Resolution:
    """Every stage that needs no model. 'unresolved' with kind 'unknown' means the model
    should look at it. Fuzzy hints and the geometry columns are filled on every result
    so a reviewer can compare them with whatever stage decided."""
    hints = fuzzy_hints(unit, pool)
    geo = geo_match(state, unit, overlap)
    res = (_known_spelling(unit, pool) or _not_a_district(unit) or _fuzzy(unit, hints)
           or _geometry(unit, geo, pool) or _pending(unit, hints, pool))
    res.candidates = hints
    if geo is not None:
        res.geo_code, res.geo_share = geo
    return res


# ---- the model stage ---------------------------------------------------------------------------------

class ResolveItem(BaseModel):
    unit: str
    district_code: Optional[int] = Field(default=None, description="a code from the candidate list, or null")
    kind: Kind
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(description="one line: how you know")


class ResolveBatch(BaseModel):
    items: list[ResolveItem]


@dataclass
class PendingUnit:
    unit: str
    first_year: int
    last_year: int
    hints: list[Hint]


def hint_text(hints: list[Hint]) -> str:
    return " | ".join(f"{code}:{name}:{score:g}" for code, name, score in hints)


def candidate_lines(pool: Pool, registry: dict) -> list[str]:
    """One line per candidate district of a state, with everything the model may match on."""
    by_code = {d["district_code"]: d for d in registry["districts"]}
    lines = []
    for code in sorted(pool.codes):
        d = by_code[code]
        circles = ", ".join(c["norm"] for c in d["circles"]) or "-"
        lines.append(f"{code} | {d['name']} | spellings: {', '.join(d['aliases'])} | circles: {circles} | "
                     f"seen {d['first_seen_year']}-{d['last_seen_year']}")
    return lines


def build_prompt(state_name: str, state_code: int, candidates: list[str], pending: list[PendingUnit],
                 missing=(), unknown=()) -> str:
    lines = [
        f"Task: for each police reporting unit name from the 2001-2014 NCRB 'crimes against women' table for "
        f"{state_name} (state code {state_code}), say which current district it belongs to, or that it has none.",
        "",
        f"Candidate districts of {state_name} (LGD code | current name | spellings seen 2016-2022 | police circles "
        "| years seen):",
        *candidates,
        "",
        "Names to resolve (years seen in the 2001-2014 file; hints = closest candidates by spelling, as "
        "code:name:score):",
        *[f"- {p.unit} (seen {p.first_year}-{p.last_year}) hints: {hint_text(p.hints) or 'none'}" for p in pending],
        "",
        "Common cases: renames (Cuddapah -> Y.S.R.); a district named after its headquarters town (Himatnagar is "
        "the HQ of Sabarkantha); abbreviations (kkumey = Kurung Kumey, nchills = North Cachar Hills = Dima Hasao); "
        "police units without geography (CAW cell, STF, CB-CID, crime branch); railway police units (GRP, Rly.) "
        "that sit inside a district.",
        "",
        "Rules:",
        "1. Every name appears exactly once, spelled exactly as given.",
        "2. district_code must be a code from the candidate list above, or null when the unit has no district "
        "(a police unit without geography) or you cannot tell.",
        f"3. kind is one of: {', '.join(KINDS)}. 'district' = the district itself under any spelling; 'rename' = "
        "the district was renamed; 'headquarters_town' = named after the district's HQ town; 'circle' = a police "
        "area inside the district; 'railway' = a railway police unit inside the district; 'commissionerate' = a "
        "city police unit inside the district; 'not_a_district' = no geography; 'unknown' = cannot tell.",
        "4. confidence between 0 and 1.",
        "5. reason: one line saying how you know.",
    ]
    if missing or unknown:
        lines += ["", "Your previous answer was incomplete."]
        if missing:
            lines.append(f"Missing names that MUST be included: {sorted(missing)}")
        if unknown:
            lines.append(f"Unknown names that must NOT appear: {sorted(unknown)}")
        lines.append(f"Return all {len(pending)} names exactly once.")
    return "\n".join(lines)


def resolve_llm(client, state_code: int, state_name: str, pending: list[PendingUnit], pool: Pool,
                registry: dict) -> dict[str, Resolution]:
    """One model call for one state, plus one repair call if the set of names does not
    match. Two guards the schema cannot express live here: every asked name must come
    back, and a proposed code must be one of this state's candidates."""
    want = {p.unit for p in pending}
    candidates = candidate_lines(pool, registry)
    tag = f"resolve:{state_code}"
    result = client.complete(build_prompt(state_name, state_code, candidates, pending), ResolveBatch, tag=tag,
                             system=PROJECT_CONTEXT)
    by_unit = {norm(i.unit): i for i in result.parsed.items}
    if set(by_unit) != want:
        result = client.complete(build_prompt(state_name, state_code, candidates, pending,
                                              missing=want - set(by_unit), unknown=set(by_unit) - want),
                                 ResolveBatch, tag=f"{tag}:repair", system=PROJECT_CONTEXT)
        by_unit.update({norm(i.unit): i for i in result.parsed.items})
    model = f"llm:{result.model}"

    out = {}
    for p in pending:
        item = by_unit.get(p.unit)
        if item is None:
            res = Resolution(None, "unknown", "unresolved", 0.0, "not answered by the model")
        elif item.district_code is not None and item.district_code not in pool.codes:
            res = Resolution(None, "unknown", "unresolved", 0.0,
                             f"model proposed code {item.district_code} which is not a candidate for this state")
        else:
            res = Resolution(item.district_code, item.kind, "llm", item.confidence, item.reason)
        res.proposed_by, res.candidates = model, p.hints
        out[p.unit] = res
    return out


# ---- the decision record ---------------------------------------------------------------------------

def read_rows(path: Path) -> dict[tuple[str, str, str], dict]:
    """The decision record keyed by (era, state, unit): one unit name is decided once per source file."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        return {(r["era"], r["state"], r["unit"]): r for r in csv.DictReader(fh)}


def write_rows(path: Path, rows: dict[tuple[str, str, str], dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow({c: rows[key].get(c, "") for c in CSV_COLUMNS})


def _fmt(x: float | None) -> str:
    return "" if x is None else f"{x:.3g}"


def accepts_without_review(stage: str, kind: str, score: float | None,
                           cuts: dict[str, float] = ACCEPT_WITHOUT_REVIEW) -> bool:
    """A scored answer the pipeline takes without a human: its stage has a cut, it names an answer (kind 'unknown'
    is the model saying it cannot tell) and its score reaches the cut."""
    cut = cuts.get(stage)
    return cut is not None and score is not None and kind != "unknown" and score >= cut


def accepted_without_review(row: dict) -> bool:
    """A decision-record row the cut accepted and no human has written on since."""
    return row.get("verified") == "1" and row.get("reviewer_note", "").startswith(ACCEPTED_NOTE)


def sent_to_a_human(row: dict) -> bool:
    """Whether the pipeline left this row's decision to a person, decided yet or not: every stage but a plain
    identity lookup (an identity match held for review because its name looks like a police unit is a person's
    call too), less the answers the cut accepted."""
    if accepted_without_review(row):
        return False
    return row.get("stage") not in AUTO_VERIFIED_STAGES or HELD_FOR_REVIEW in row.get("reason", "")


def make_row(u, state_code: int, res: Resolution, names: dict[int, str], reviewer_note: str = "", era: str = ERA) -> dict:
    code = res.district_code
    accepted = not reviewer_note and accepts_without_review(res.stage, res.kind, res.confidence)   # a human's note keeps the row theirs
    if accepted:
        reviewer_note = (f"{ACCEPTED_NOTE}: {res.stage} score {_fmt(res.confidence)} >= cut "
                         f"{ACCEPT_WITHOUT_REVIEW[res.stage]:.2f}")
    return {"era": era, "state": u.state, "state_code": state_code, "unit": u.unit,
            "first_year": int(u.first_year), "last_year": int(u.last_year), "n_years": int(u.n_years),
            "district_code": "" if code is None else code, "district_name": "" if code is None else names.get(code, ""),
            "kind": res.kind, "stage": res.stage, "confidence": _fmt(res.confidence), "reason": res.reason,
            "candidates": hint_text(res.candidates),
            "geo_code": "" if res.geo_code is None else res.geo_code, "geo_share": _fmt(res.geo_share),
            "proposed_by": res.proposed_by,
            "verified": "1" if (res.stage in AUTO_VERIFIED_STAGES and not res.needs_review) or accepted else "0",
            "reviewer_note": reviewer_note}


def run(client=None, *, era: str = ERA, csv_path: Path = CSV_PATH, registry_path: Path = REGISTRY_PATH,
        overlap_path: Path = OVERLAP_PATH, units: pd.DataFrame | None = None) -> dict:
    """Resolve every unit of one era and write the decision record. ``client=None`` skips the model
    stage. Rows a human has verified are kept exactly as they are, including any whose
    (state, unit) key the input no longer produces (counted as verified_not_in_input,
    so a changed normaliser or raw file shows up instead of silently dropping a decision).
    Rows of other eras are never touched."""
    registry = load_registry(registry_path)
    units = load_units(era) if units is None else units
    state_code = map_states(units["state"].unique(), registry)
    pools = build_pools(registry)
    overlap = load_overlap(overlap_path, registry, state_code)
    names = {d["district_code"]: d["name"] for d in registry["districts"]}
    state_names = {s["state_code"]: s["name"] for s in registry["states"]}

    existing = read_rows(csv_path)
    rows = {k: r for k, r in existing.items() if r.get("verified") == "1" or k[0] != era}   # a human decision outlives its input row
    carried: dict[tuple[str, str], dict] = {}                 # (state, unit) -> a verified decision from another era
    for (e, s, u), r in existing.items():
        if e != era and r.get("verified") == "1" and sent_to_a_human(r) and (s, u) not in carried:
            carried[(s, u)] = r              # a human's decision only: a plain lookup or a cut-accepted answer goes through the stages
    in_input = {(era, s, u) for s, u in zip(units["state"], units["unit"])}
    stale = sum(k[0] == era and k not in in_input for k in rows)
    kept = carried_n = 0
    pending: dict[int, list[tuple]] = {}                      # state_code -> [(unit row, free Resolution)]
    for u in units.itertuples(index=False):
        key = (era, u.state, u.unit)
        old = existing.get(key, {})
        if key in rows:
            kept += 1
            continue
        code = state_code[u.state]
        src = carried.get((u.state, u.unit))
        if src is not None:                                    # a human decided this exact name in another era: same answer, no model
            res = Resolution(int(src["district_code"]) if src["district_code"].strip() else None, src["kind"], "human", 1.0,
                             f"carried from {src['era']}: {src['reason']}", proposed_by=f"carried:{src['era']}")
            note = f"carried from {src['era']}" + (f": {src['reviewer_note']}" if src.get("reviewer_note") else "")
            rows[key] = {**make_row(u, code, res, names, note, era=era), "verified": "1"}
            carried_n += 1
            continue
        res = resolve_free(u.state, u.unit, pools[code], overlap)
        rows[key] = make_row(u, code, res, names, old.get("reviewer_note", ""), era=era)
        if res.stage == "unresolved" and res.kind == "unknown":
            pending.setdefault(code, []).append((u, res))

    for code, items in pending.items():
        if client is None:
            for u, res in items:
                rows[(era, u.state, u.unit)]["reason"] = f"model stage not run; {res.reason}"
            continue
        asked = [PendingUnit(u.unit, int(u.first_year), int(u.last_year), res.candidates) for u, res in items]
        answers = resolve_llm(client, code, state_names[code], asked, pools[code], registry)
        for u, free in items:
            res = answers[u.unit]
            res.geo_code, res.geo_share = free.geo_code, free.geo_share
            rows[(era, u.state, u.unit)] = make_row(u, code, res, names, existing.get((era, u.state, u.unit), {}).get("reviewer_note", ""),
                                                    era=era)

    write_rows(csv_path, rows)
    mine = [r for k, r in rows.items() if k[0] == era]
    return {"era": era, "total": len(mine), "all_rows": len(rows), "kept_verified": kept, "carried": carried_n,
            "verified_not_in_input": stale,
            "by_stage": dict(Counter(r["stage"] for r in mine)),
            "pending_states": len(pending), "model_stage": "run" if client is not None else "skipped"}


# ---- report ------------------------------------------------------------------------------------------

def _table(title: str, header: tuple[str, ...], body: list[tuple]) -> list[str]:
    return [f"### {title}", "", "| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|",
            *["| " + " | ".join(str(c) for c in r) + " |" for r in body], ""]


def report(csv_path: Path = CSV_PATH, out_dir: Path = REPORTS_DIR, today: datetime.date | None = None) -> str:
    """Per era: stage and kind tables, the resolution rate, model-vs-geometry agreement and the
    unresolved list. Written to data/reports/<date>_resolver.md and returned."""
    all_rows = list(read_rows(csv_path).values())
    today = today or datetime.date.today()
    lines = [f"# Resolver report {today.isoformat()}", "", f"{len(all_rows)} (era, state, unit) rows in the decision record.", ""]
    for era in sorted({r["era"] for r in all_rows}):
        rows = [r for r in all_rows if r["era"] == era]
        n = max(len(rows), 1)
        can_have = [r for r in rows if r["kind"] != "not_a_district"]
        resolved = [r for r in can_have if r["district_code"]]
        both = [r for r in rows if r["stage"] == "llm" and r["district_code"] and r["geo_code"]]
        agree = sum(r["district_code"] == r["geo_code"] for r in both)
        unresolved = [r for r in can_have if not r["district_code"]]
        lines += [f"## Era {era}: {len(rows)} (state, unit) pairs", ""]
        lines += _table("Stages", ("stage", "rows", "share"),
                        [(s, c, f"{c / n:.1%}") for s, c in Counter(r["stage"] for r in rows).most_common()])
        lines += _table("Kinds", ("kind", "rows"), Counter(r["kind"] for r in rows).most_common())
        lines += ["### Resolution rate", "",
                  f"{len(resolved)} of {len(can_have)} rows that can have a district have one "
                  f"({len(resolved) / max(len(can_have), 1):.1%}); {len(rows) - len(can_have)} rows are police units "
                  "with no district geography.", ""]
        lines += ["### Model vs geometry", "",
                  (f"{agree} of {len(both)} model answers agree with the old-map overlap ({agree / len(both):.1%})."
                   if both else "no rows have both a model answer and a geometry match."), ""]
        lines += _table(f"Unresolved ({len(unresolved)})", ("state", "unit", "years", "stage", "reason", "candidates"),
                        [(r["state"], r["unit"], f"{r['first_year']}-{r['last_year']}", r["stage"], r["reason"],
                          r["candidates"]) for r in unresolved])
    text = "\n".join(lines)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{today.isoformat()}_resolver.md").write_text(text, encoding="utf-8")
    return text


# ---- loading the human's decisions ------------------------------------------------------------------------

def apply(conn, *, csv_path: Path = CSV_PATH) -> dict:
    """Load verified rows into the unit_map table. Unverified rows stay out. Values are
    checked against the table's allowed stages and kinds so a typo fails with a message,
    not a constraint error."""
    loaded, pending = 0, 0
    known = {row[0] for row in conn.execute("SELECT district_code FROM districts")}
    for (era, state, unit), r in read_rows(csv_path).items():
        if r.get("verified") != "1":
            pending += 1
            continue
        stage, kind = r["stage"].strip(), r["kind"].strip()
        if stage not in STAGES:
            raise ValueError(f"{state}/{unit}: stage '{stage}' is not one of {STAGES}")
        if kind and kind not in KINDS:
            raise ValueError(f"{state}/{unit}: kind '{kind}' is not one of {KINDS}")
        code = int(r["district_code"]) if r["district_code"].strip() else None
        if code is not None and code not in known:      # say which row, not 'FOREIGN KEY constraint failed'
            raise ValueError(f"{state}/{unit}: district_code {code} is not in the districts table; "
                             "run `python -m data_ingestion.registry` (it loads the map-only codes from data/lgd_districts.csv)")
        confidence = float(r["confidence"]) if r["confidence"].strip() else None
        conn.execute("INSERT OR REPLACE INTO unit_map (era, state, unit, district_code, kind, stage, confidence, reason, "
                     "verified) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)",
                     (era, state, unit, code, kind or None, stage, confidence, r["reason"]))
        loaded += 1
    conn.commit()
    return {"loaded": loaded, "still_unverified": pending}


# ---- CLI -----------------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Resolve code-less unit names to district codes; report; apply after review.")
    ap.add_argument("command", choices=["run", "report", "apply"])
    ap.add_argument("--llm", action="store_true", help="run the model stage (needs provider keys in .env)")
    ap.add_argument("--era", choices=NAME_KEYED_ERAS, default=ERA, help="which code-less file to resolve")
    args = ap.parse_args(argv)
    if args.command == "run":
        client = build_client(store.connect()) if args.llm else None
        summary = run(client, era=args.era)
        print(f"unit_map ({summary['era']}): {summary['total']} rows, {summary['kept_verified']} kept as verified, "
              f"model stage {summary['model_stage']} ({summary['pending_states']} states had pending names); "
              f"{summary['all_rows']} rows in the file")
        if summary["verified_not_in_input"]:
            print(f"  WARNING: {summary['verified_not_in_input']} verified row(s) have a (state, unit) key the input "
                  "no longer produces; kept as they are, check the normaliser or the raw file")
        for stage, count in sorted(summary["by_stage"].items(), key=lambda kv: -kv[1]):
            print(f"  {stage:<11} {count:>5}  {count / summary['total']:>6.1%}")
        print(f"review file -> {CSV_PATH}")
    elif args.command == "report":
        print(report())
    else:
        print(apply(store.connect()))


if __name__ == "__main__":
    main()
