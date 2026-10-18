"""Re-basing: the observed panel re-expressed on today's districts, every invented number marked.

A filed row does not mean the same land in every year: Karimnagar's 2016 row covers seven of today's
districts, its 2017 row one. For each year this module works out which of today's districts sit inside
which filed row (from the verified lineage), divides each such row among them by a share, and writes one
row per district and year to ``panel_rebased`` and ``data/panel_rebased.parquet``. The pieces add back to
the filed row exactly, so national totals never change; rows carry today's state, so state totals are on
today's states.

Vocabulary. A *holder* is the district whose filed row contains another district's area in a year; that
other district is *pooled* in it. A *share* is the pooled district's fraction of the holder's row.

The share of a lineage edge is measured where both sides report separately: for a split, the first
``BASIS_YEARS`` years from the split year; for a late pooling (``reported_together`` with a year), the last
years before it. Each year's ratio is the pooled unit's total over the holder unit's total. The pooled
unit is the member as it stood the year before the edge took effect (its own pooled children count with
it; a district that joined it later does not), the holder unit is the holder's as the lineage stood just
before a split or just after a join, and a year counts only when filed rows cover each unit exactly. The mean of the ratios is the share; ``share_basis``
says how many years it rests on (``crime:3y`` .. ``crime:1y``). With no such year the old map's area share
stands in (``area``). With neither, nothing is estimated and the report says so.

A district carved from several parents takes a part from each; its filed total is divided among the
parents by the old map's area overlap before shares are measured (equal parts, flagged, when the map
cannot tell; a parent the map gives no land moves nothing and is listed). A chain (Lower Subansiri held
Kurung Kumey, which held Kra Daadi) is peeled outer edge first. Within one holder, siblings that split in
different years are peeled in year order, each from what remained; late poolings are peeled latest first,
each from what remained, independently of the splits, because every share was measured against the
holder's unit as it stood at that edge's own year.

Never estimated: map-only districts (never a row of their own in any file; the product says "reported
with ..."), a year a district simply failed to file (stays empty), and a row a district filed before the
lineage says it existed (stays observed; the report lists it).

Stable units are rebuilt here from the shares actually used: an edge joins two districts into one group
only when its share is at least ``GROUP_THRESHOLD``; below it the transfer is applied and no group forms.
A late pooling always groups (the pooled district's recent years are all estimates), and a map-only
district that moves between holders links them when its area share of the old holder is large enough.

    python -m agents.rebase build     # from the store's panel_observed and district_lineage
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd

from agents import lineage
from agents.schemas import CORE_CATEGORIES
from data_ingestion.normalize import norm

ROOT = Path(__file__).resolve().parents[1]
PARQUET_PATH = ROOT / "data" / "panel_rebased.parquet"
UNITS_PATH = lineage.UNITS_PATH
AREA_PATH = lineage.OVERLAP_CSV
REPORTS_DIR = lineage.REPORTS_DIR
GROUP_THRESHOLD = 0.2
BASIS_YEARS = 3
MIN_OVERLAP = 0.05           # an old polygon counts as a district's own only if at least this much of it lies there today
SPLIT, JOIN = "split", "reported_together"
BASES = ("observed", "remainder", "crime:3y", "crime:2y", "crime:1y", "area")        # strongest first
COLUMNS = ["unit_id", "state_code", "district_code", "year", *CORE_CATEGORIES, "total",
           "is_estimated", "share", "share_basis", "holder_code"]
N = len(CORE_CATEGORIES)
WEIGHTS_PATH = ROOT / "data" / "split_weights.csv"      # the human's record of how a multi-parent child divides
WEIGHT_COLUMNS = ["child_code", "child_name", "parent_code", "parent_name", "kind", "weight", "basis", "proposed_by",
                  "verified", "reviewer_note"]
WEIGHT_TOLERANCE = 0.01


@dataclass(frozen=True)
class Pooling:
    """One verified lineage fact as a rule: ``member`` sits inside ``holder``'s row while ``active``."""
    member: int
    holder: int
    kind: str          # SPLIT: pooled before ``year``; JOIN: pooled from ``year``
    year: int

    def active(self, y: int) -> bool:
        return y < self.year if self.kind == SPLIT else y >= self.year

    @property
    def parent_child(self) -> tuple[int, int]:       # the (parent, child) order lineage.csv uses
        return (self.holder, self.member) if self.kind == SPLIT else (self.member, self.holder)

    def label(self, names: dict[int, str]) -> str:
        p, c = self.parent_child
        return f"{names.get(p, p)} -> {names.get(c, c)} ({self.kind} {self.year})"


@dataclass
class Share:
    share: float | None
    basis: str | None          # 'crime:<n>y', 'area' or None
    years: list[int]
    area_share: float | None   # fraction of the holder's old polygon now in the member, when the old map has both


# ---- inputs ------------------------------------------------------------------------------------------------

def poolings_from_store(conn) -> tuple[list[Pooling], list[tuple]]:
    """Verified split and reported_together edges as poolings. A split with no year says nothing about which
    years are pooled, so it is returned in the second list instead of used."""
    out, skipped = [], []
    for r in conn.execute("SELECT parent_code, child_code, kind, year FROM district_lineage WHERE verified = 1 "
                          "AND kind IN (?, ?)", (SPLIT, JOIN)):
        p, c, kind, year = int(r[0]), int(r[1]), r[2], r[3]
        if kind == SPLIT:
            (out.append(Pooling(c, p, SPLIT, int(year))) if year is not None else skipped.append((p, c, kind)))
        else:
            out.append(Pooling(p, c, JOIN, int(year) if year is not None else 0))
    return out, skipped


def districts_from_store(conn) -> dict[int, dict]:
    return {int(r[0]): {"state_code": int(r[1]), "name": r[2], "aliases": json.loads(r[3] or "[]"), "in_files": int(r[4] or 0)}
            for r in conn.execute("SELECT district_code, state_code, name, aliases_json, in_files FROM districts")}


def observed_from_store(conn) -> pd.DataFrame:
    rows = conn.execute("SELECT * FROM panel_observed ORDER BY state_code, district_code, year").fetchall()
    return pd.DataFrame([dict(r) for r in rows], columns=["unit_id", "state_code", "district_code", "year", "era",
                                                           *CORE_CATEGORIES, "total", "n_records"])


def read_weights(path: Path = WEIGHTS_PATH) -> dict[tuple[int, int, str], float]:
    """Verified rows of data/split_weights.csv as (child, parent, kind) -> weight: the human's division of a child
    among its parents where the old map cannot be trusted. The weights of one child and kind must sum to 1 within
    WEIGHT_TOLERANCE; a row that does not is a typo and stops the build with the child's name."""
    path = Path(path)
    if not path.exists():
        return {}
    out: dict[tuple[int, int, str], float] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("verified", "").strip() != "1":
                continue
            key = (int(r["child_code"]), int(r["parent_code"]), r["kind"].strip())
            if key in out:
                raise ValueError(f"split_weights: {r['child_name']} <- {r['parent_name']} ({r['kind']}) appears twice")
            out[key] = float(r["weight"])
    sums: dict[tuple[int, str], float] = defaultdict(float)
    for (child, _parent, kind), w in out.items():
        if w < 0:
            raise ValueError(f"split_weights: a negative weight for child {child}")
        sums[(child, kind)] += w
    for (child, kind), s in sums.items():
        if abs(s - 1.0) > WEIGHT_TOLERANCE:
            raise ValueError(f"split_weights: the {kind} weights of child {child} sum to {s:.3f}, not 1")
    return out


def units_changes(old: list[dict] | None, new: list[dict]) -> dict:
    """Stable units that appeared or disappeared since the last build, by member set, so a changed weight or edge
    is seen in the report before anyone has to notice it in the panel."""
    before = {frozenset(u["members"]) for u in (old or [])}
    after = {frozenset(u["members"]) for u in new}
    return {"appeared": sorted(sorted(g) for g in after - before), "disappeared": sorted(sorted(g) for g in before - after),
            "compared": old is not None}


# ---- the computation -----------------------------------------------------------------------------------------

class Rebaser:
    """Shares, then one pass per year. ``panel`` is the observed panel (district rows keyed by district_code,
    unallocated rows with district_code NULL); ``poolings`` the verified lineage; ``districts`` today's
    districts (code -> state_code, name, aliases); ``area`` the old-map overlap table or None."""

    def __init__(self, panel: pd.DataFrame, poolings: list[Pooling], districts: dict[int, dict],
                 area: pd.DataFrame | None = None, *, threshold: float = GROUP_THRESHOLD, basis_years: int = BASIS_YEARS,
                 weights: dict[tuple[int, int, str], float] | None = None):
        self.panel, self.districts = panel, districts
        self.overrides = dict(weights or {})                    # (member, holder, kind) -> weight from the decision record
        self.area = None
        if area is not None and len(area):
            self.area = area.copy()
            self.area["_poly"] = list(zip(self.area["old_state_norm"].astype(str), self.area["old_norm"].astype(str)))
        self.threshold, self.basis_years = threshold, basis_years
        self.names = {code: d["name"] for code, d in districts.items()}
        dist = panel[panel["district_code"].notna()]
        self.years = sorted(int(y) for y in panel["year"].unique()) if len(panel) else []
        self.filed: dict[tuple[int, int], np.ndarray] = {
            (int(r.district_code), int(r.year)): np.array([float(getattr(r, c)) for c in CORE_CATEGORIES])
            for r in dist.itertuples()}
        self.filed_years: dict[int, set[int]] = defaultdict(set)
        for d, y in self.filed:
            self.filed_years[d].add(y)
        self.map_only = {d for d in districts if d not in self.filed_years}
        known = [p for p in poolings if p.member in districts and p.holder in districts]
        self.map_only_poolings = [p for p in known if p.member in self.map_only]
        self.poolings = [p for p in known if p.member not in self.map_only and p.holder not in self.map_only]
        self.out_edges: dict[int, list[Pooling]] = defaultdict(list)
        self.in_edges: dict[int, list[Pooling]] = defaultdict(list)
        for p in self.poolings:
            self.out_edges[p.holder].append(p)
            self.in_edges[p.member].append(p)
        self._tops: dict[tuple[int, int], frozenset] = {}
        self._contents: dict[int, dict[int, set[int]]] = {}
        self._polygons: dict[int, set[tuple[str, str]]] = {}   # code -> the old map's (state, name) polygons that were it
        self._claims: dict[str, set[int]] = defaultdict(set)      # normalised name or alias -> the districts bearing it
        for code in districts:
            for n in self._holder_names(code):
                self._claims[n].add(code)
        self.diag: dict[str, list] = defaultdict(list)
        self.weights = self._weights()
        self.shares: dict[Pooling, Share] = {}
        self.coverage: list[dict] = []

    # -- the old map ------------------------------------------------------------------------------------------

    def _holder_names(self, code: int) -> set[str]:
        d = self.districts[code]
        return {norm(d["name"]), *(norm(a) for a in d.get("aliases", []))}

    def _holder_polygons(self, code: int) -> set[tuple[str, str]]:
        """The old map's polygons that were this district: a polygon that bears its name or alias and overlaps it
        today, or one that is at least half of it today and bears no other district's name (so 'Rangareddi' and
        'Chittaurgarh' are found, while Peddapalli does not inherit 'Karimnagar' and a stray 'Junagadh' polygon that
        barely touches Junagadh is left out)."""
        if code not in self._polygons:
            own = self.area[(self.area["new_code"] == code) & (self.area["share"] >= MIN_OVERLAP)]
            names = self._holder_names(code)
            named = {poly for poly in own["_poly"] if poly[1] in names}
            adopted = {poly for poly in own[own["new_share"] >= 0.5]["_poly"] if not (self._claims.get(poly[1], set()) - {code})}
            self._polygons[code] = named | adopted
        return self._polygons[code]

    def _area_rows(self, holder: int, member: int) -> pd.DataFrame | None:
        """Overlap rows between the holder's old polygon(s) and the member. None when the old map has no polygon
        for the holder at all; an empty frame when it has one that does not touch the member (a true zero)."""
        if self.area is None or self.area.empty:
            return None
        polygons = self._holder_polygons(holder)
        if not polygons:
            return None
        mine = self.area["_poly"].map(lambda poly: poly in polygons)
        return self.area[(self.area["new_code"] == member) & mine]

    def area_share(self, holder: int, member: int) -> float | None:
        """Fraction of the holder's old polygon that is the member today: the area analogue of a share."""
        hit = self._area_rows(holder, member)
        return None if hit is None else min(float(hit["share"].sum()), 1.0)

    def area_cover(self, holder: int, member: int) -> float | None:
        """Fraction of the member's polygon that was the holder's old polygon: how a multi-parent child divides."""
        hit = self._area_rows(holder, member)
        return None if hit is None else float(hit["new_share"].sum())

    def _weights(self) -> dict[Pooling, float]:
        """A member with several holders of one kind is split among them by area cover; equal parts, flagged,
        when the old map lacks a holder. A member with one holder has weight 1. A verified row of the decision
        record (data/split_weights.csv) replaces the map's division for its member: every holder of that member
        must then have a row, and the rows are used as given. A row naming no edge is reported, not used."""
        by_member: dict[tuple[int, str], list[Pooling]] = defaultdict(list)
        for p in self.poolings:
            by_member[(p.member, p.kind)].append(p)
        known = {(p.member, p.holder, p.kind) for p in self.poolings + self.map_only_poolings}
        self.diag["orphan_weights"] = [k for k in self.overrides if k not in known]
        w: dict[Pooling, float] = {}
        for (m, kind), ps in by_member.items():
            given = {p: self.overrides[(m, p.holder, kind)] for p in ps if (m, p.holder, kind) in self.overrides}
            if given:
                if len(given) != len(ps):
                    raise ValueError(f"split_weights: {self.names.get(m, m)} has a weight for {len(given)} of its {len(ps)} "
                                     f"{kind} holders; give one for each")
                total = sum(given.values())
                for p in ps:
                    w[p] = given[p] / total
                self.diag["weight_overrides"].append({"member": m, "holders": {p.holder: round(w[p], 4) for p in ps}})
                continue
            if len(ps) == 1:
                w[ps[0]] = 1.0
                continue
            cover = {p: self.area_cover(p.holder, m) for p in ps}
            total = sum(v for v in cover.values() if v is not None)
            if all(v is not None for v in cover.values()) and total > 0:
                for p in ps:
                    w[p] = cover[p] / total
            else:
                for p in ps:
                    w[p] = 1.0 / len(ps)
                self.diag["equal_weights"].append({"member": m, "holders": [p.holder for p in ps]})
        return w

    # -- who holds whom, by lineage and by data -----------------------------------------------------------------

    def lineage_set(self, x: int, y: int) -> dict[int, float]:
        """``x`` and everything the lineage says is pooled in it in year ``y``, with the weight of each part: 1 for
        a district wholly inside, the area weight for a part, products along a chain. A district reached through
        two members of the set (a child of two parents both inside) carries the sum of its parts."""
        members, stack = {x}, [x]
        while stack:
            h = stack.pop()
            for p in self.out_edges[h]:
                if p.active(y) and p.member not in members:
                    members.add(p.member)
                    stack.append(p.member)
        out = {x: 1.0}
        for _ in range(len(members)):                           # settle weights in dependency order (a DAG this size converges fast)
            nxt = {x: 1.0}
            for h in members:
                for p in self.out_edges[h]:
                    if p.active(y) and p.member in members and h in out:
                        nxt[p.member] = nxt.get(p.member, 0.0) + out[h] * self.weights[p]
            if nxt == out:
                break
            out = nxt
        return out

    def tops(self, d: int, y: int) -> frozenset:
        """The filed districts whose rows contain ``d`` in year ``y`` (itself when it filed); empty = missing."""
        key = (d, y)
        if key in self._tops:
            return self._tops[key]
        self._tops[key] = frozenset()                            # guards a cycle in the edges
        if y in self.filed_years.get(d, ()):
            res = frozenset({d})
        else:
            res = frozenset().union(*(self.tops(p.holder, y) for p in self.in_edges[d] if p.active(y)))
        self._tops[key] = res
        return res

    def contents(self, t: int, y: int) -> set[int]:
        if y not in self._contents:
            inv: dict[int, set[int]] = defaultdict(set)
            for d in set(self.filed_years) | set(self.in_edges):
                for top in self.tops(d, y):
                    inv[top].add(d)
            self._contents[y] = inv
        return self._contents[y].get(t, set())

    def value(self, X: dict[int, float], y: int) -> np.ndarray | None:
        """Counts of the unit ``X`` in year ``y`` from filed rows, or None when filed rows do not cover it exactly:
        a member missing or held outside X, or a filed member whose row also holds a non-member."""
        vec, filed_in = np.zeros(N), []
        for d, w in X.items():
            tops = self.tops(d, y)
            if not tops or not (tops & X.keys()):
                return None
            if d in tops:
                vec += w * self.filed[(d, y)]
                filed_in.append(d)
        for t in filed_in:
            if any(e not in X for e in self.contents(t, y)):
                return None
        return vec

    # -- shares --------------------------------------------------------------------------------------------------

    def measure(self) -> dict[Pooling, Share]:
        for p in self.poolings:
            t = p.year
            M = self.lineage_set(p.member, t - 1)                 # the member's unit just before the edge took effect
            H = self.lineage_set(p.holder, t - 1 if p.kind == SPLIT else t)
            H_rest = {d: w for d, w in H.items() if d not in M}
            a = self.weights[p]
            area_share = self.area_share(p.holder, p.member)
            if a == 0:                                            # the old map gives this holder no land in the member
                self.shares[p] = Share(0.0, "area", [], area_share)
                self.diag["zero_weight"].append({"edge": p})
                continue
            candidates = ([y for y in self.years if y >= t] if p.kind == SPLIT else [y for y in reversed(self.years) if y < t])
            ratios, used = [], []
            for y in candidates:
                vm, vh = self.value(M, y), self.value(H_rest, y)
                if vm is None or vh is None:
                    continue
                num, den = a * vm.sum(), vh.sum() + a * vm.sum()
                if den <= 0:
                    continue
                ratios.append(num / den)
                used.append(y)
                if len(ratios) == self.basis_years:
                    break
            if ratios:
                self.shares[p] = Share(float(np.mean(ratios)), f"crime:{len(ratios)}y", used, area_share)
            elif area_share is not None:
                self.shares[p] = Share(area_share, "area", [], area_share)
            else:
                self.shares[p] = Share(None, None, [], None)
        return self.shares

    # -- one year ---------------------------------------------------------------------------------------------------

    def apply_year(self, y: int) -> list[dict]:
        filed_now = {d for d, ys in self.filed_years.items() if y in ys}
        incoming: dict[int, list[Pooling]] = defaultdict(list)
        for p in self.poolings:
            if p.member in filed_now or not p.active(y) or self.weights[p] == 0:
                continue
            if self.shares[p].share is None:
                self.diag["no_estimate"].append({"edge": p, "year": y})
                continue
            incoming[p.member].append(p)
        for m, ps in list(incoming.items()):
            if len({p.kind for p in ps}) > 1:                   # a split and a late pooling both active: keep the split
                incoming[m] = [p for p in ps if p.kind == SPLIT]
                self.diag["conflicting_edges"].append({"member": m, "year": y})
        while True:                                              # every holder needs a row this year: filed, or estimated itself
            short = {m: [p.holder for p in ps if p.holder not in filed_now and p.holder not in incoming] for m, ps in incoming.items()}
            dropped = [m for m, holders in short.items() if holders]
            if not dropped:
                break
            for m in dropped:                                    # one holder short: no row at all rather than a silently partial one
                self.diag["holder_unresolved"].append({"member": m, "year": y, "holders": short[m]})
                del incoming[m]
        nodes = filed_now | set(incoming)
        outgoing: dict[int, list[Pooling]] = defaultdict(list)
        for ps in incoming.values():
            for p in ps:
                outgoing[p.holder].append(p)
        indeg = {n: len(incoming.get(n, [])) for n in nodes}
        amount = {d: (self.filed[(d, y)].copy() if d in filed_now else np.zeros(N)) for d in nodes}
        received: dict[int, list[tuple[Pooling, np.ndarray, float]]] = defaultdict(list)
        gave: set[int] = set()
        ready, order = sorted(n for n in nodes if indeg[n] == 0), []

        def give(p: Pooling, part: np.ndarray, base: np.ndarray) -> None:
            amount[p.member] += part
            received[p.member].append((p, part, float(base.sum())))
            indeg[p.member] -= 1
            if indeg[p.member] == 0:
                ready.append(p.member)

        while ready:                                             # holders before members, so a member's amount is complete
            x = ready.pop(0)
            order.append(x)
            row, given = amount[x].copy(), np.zeros(N)
            base = row.copy()                                    # late poolings: the latest to join is peeled first, from the whole row
            for p in sorted((p for p in outgoing[x] if p.kind == JOIN), key=lambda p: (-p.year, p.member)):
                part = self.shares[p].share * base
                give(p, part, base)
                base, given = base - part, given + part
            base = row.copy()                                    # splits: earliest first; one year's siblings from the same remainder
            for _year, group in groupby(sorted((p for p in outgoing[x] if p.kind == SPLIT), key=lambda p: (p.year, p.member)),
                                        key=lambda p: p.year):
                parts = [(p, self.shares[p].share * base) for p in group]
                for p, part in parts:
                    give(p, part, base)
                    given = given + part
                base = base - sum(part for _, part in parts)
            rem = row - given
            if (rem < -1e-6 * max(float(row.sum()), 1.0)).any():
                self.diag["over_assigned"].append({"holder": x, "year": y, "short": float(-rem[rem < 0].sum())})
            amount[x] = np.clip(rem, 0.0, None)                  # a negative beyond rounding is reported above; none is stored
            if given.sum() > 0:
                gave.add(x)
        if len(order) < len(nodes):
            self.diag["cycles"].append({"year": y, "codes": sorted(nodes - set(order))})
        rows = []
        for x in order:
            vec = amount[x]
            if x in filed_now and x not in gave:
                basis, est, share, holder = "observed", 0, 1.0, None
            elif x in filed_now:
                filed_total = float(self.filed[(x, y)].sum())
                basis, est, share, holder = "remainder", 1, (float(vec.sum()) / filed_total if filed_total > 0 else 1.0), None
            else:
                parts = received[x]
                basis = max((self.shares[p].basis for p, _, _ in parts), key=BASES.index)     # the weakest basis among its parts
                est = 1
                bases_total = sum(b for _, _, b in parts)                                      # share: received over the holders' bases
                share = sum(float(part.sum()) for _, part, _ in parts) / bases_total if bases_total > 0 else 0.0
                holder = max(parts, key=lambda t: float(t[1].sum()))[0].holder                 # the largest part's holder
            rows.append({"unit_id": f"d{x}", "state_code": self.districts[x]["state_code"], "district_code": x, "year": y,
                         **{c: float(v) for c, v in zip(CORE_CATEGORIES, vec)}, "total": float(vec.sum()),
                         "is_estimated": est, "share": min(max(share, 0.0), 1.0), "share_basis": basis, "holder_code": holder})
        missing = sorted(d for d in self.filed_years if d not in order)
        self.coverage.append({"year": y, "observed": sum(r["share_basis"] == "observed" for r in rows),
                              "adjusted": sum(r["share_basis"] == "remainder" for r in rows),
                              "estimated": sum(r["is_estimated"] == 1 and r["share_basis"] != "remainder" for r in rows),
                              "missing": len(missing), "missing_codes": missing})
        return rows

    def run(self) -> pd.DataFrame:
        if not self.shares:
            self.measure()
        self.coverage = []
        for key in ("no_estimate", "conflicting_edges", "holder_unresolved", "over_assigned", "cycles"):
            self.diag.pop(key, None)
        rows = [r for y in self.years for r in self.apply_year(y)]
        for r in self.panel[self.panel["district_code"].isna()].itertuples():      # the states' unallocated buckets, as filed
            rows.append({"unit_id": r.unit_id, "state_code": int(r.state_code), "district_code": None, "year": int(r.year),
                         **{c: float(getattr(r, c)) for c in CORE_CATEGORIES}, "total": float(r.total),
                         "is_estimated": 0, "share": 1.0, "share_basis": "observed", "holder_code": None})
        df = pd.DataFrame(rows, columns=COLUMNS)
        df["district_code"] = df["district_code"].astype("Int64")
        df["holder_code"] = df["holder_code"].astype("Int64")
        return df.sort_values(["state_code", "district_code", "year"], na_position="last").reset_index(drop=True)

    # -- what the report needs ----------------------------------------------------------------------------------

    def filed_before_edge(self) -> list[dict]:
        out = []
        for p in self.poolings:
            if p.kind == SPLIT:
                early = sorted(y for y in self.filed_years[p.member] if y < p.year)
                if early:
                    out.append({"edge": p, "years": early})
        return out

    def crime_vs_area(self) -> list[dict]:
        out = []
        for p, s in self.shares.items():
            if s.share is None or s.area_share is None or not s.basis.startswith("crime"):
                continue
            if (s.share == 0 and s.area_share >= 0.05) or (s.share > 0 and s.area_share > 0 and not 0.5 <= s.share / s.area_share <= 2):
                out.append({"edge": p, "share": s.share, "area_share": s.area_share, "basis": s.basis})
        return sorted(out, key=lambda r: -(r["share"] / r["area_share"] if r["area_share"] else 0))

    def map_only_notes(self) -> list[dict]:
        """For each map-only district: who reports it, by period."""
        by_member: dict[int, list[Pooling]] = defaultdict(list)
        for p in self.map_only_poolings:
            by_member[p.member].append(p)
        notes = []
        for m in sorted(self.map_only):
            ps = sorted(by_member.get(m, []), key=lambda p: (p.year, p.holder))
            before = [p for p in ps if p.kind == SPLIT]
            after = [p for p in ps if p.kind == JOIN]
            moves = bool(before) and {p.holder for p in before} != {p.holder for p in after}
            notes.append({"code": m, "name": self.names[m], "state_code": self.districts[m]["state_code"],
                          "inside_before": [(p.holder, p.year) for p in before], "inside_from": [(p.holder, p.year) for p in after],
                          "moves": moves})
        return notes

    # -- groups -----------------------------------------------------------------------------------------------------

    def group_edges(self) -> list[dict]:
        """Every pooling with its share and whether it joins a group; map-only moves add their two edges."""
        first_year = min(self.years) if self.years else 0
        out = []
        for p in self.poolings:
            s = self.shares[p]
            if self.weights[p] == 0:
                grouped, why = False, "the old map gives this holder no land in the member: nothing moves"
            elif p.kind == JOIN and p.year > first_year:
                grouped, why = True, "late pooling: the member's land sits in the holder's row from then on"
            elif s.share is None:
                grouped, why = False, "no share: nothing estimated"
            else:
                grouped = s.share >= self.threshold
                why = f"share {s.share:.2f} {'>=' if grouped else '<'} {self.threshold}"
            out.append({"parent_code": p.parent_child[0], "child_code": p.parent_child[1], "kind": p.kind, "year": p.year,
                        "share": s.share, "share_basis": s.basis, "share_years": s.years, "area_share": s.area_share,
                        "weight": self.weights[p], "grouped": grouped, "why": why})
        for note in self.map_only_notes():
            if not note["moves"]:
                continue
            old = max(note["inside_before"], key=lambda t: self.area_share(t[0], note["code"]) or 0)
            share = self.area_share(old[0], note["code"])
            grouped = share is not None and share >= self.threshold
            why = (f"map-only {note['name']} moves between holders; area share of {self.names[old[0]]} "
                   f"{'unknown' if share is None else f'{share:.2f}'} {'>=' if grouped else '<'} {self.threshold}")
            for p in [q for q in self.map_only_poolings if q.member == note["code"]]:
                out.append({"parent_code": p.parent_child[0], "child_code": p.parent_child[1], "kind": p.kind, "year": p.year,
                            "share": None, "share_basis": None, "share_years": [], "area_share": share, "weight": 1.0,
                            "grouped": grouped, "why": why})
        return out

    def build_units(self, edges: list[dict] | None = None) -> list[dict]:
        edges = self.group_edges() if edges is None else edges
        rows = [{"verified": "1", "kind": e["kind"], "parent_code": str(e["parent_code"]), "child_code": str(e["child_code"]),
                 "year": str(e["year"])} for e in edges if e["grouped"]]
        nodes = [{"district_code": code, "name": d["name"], "in_files": d.get("in_files", 1)} for code, d in self.districts.items()]
        last = max(self.years) if self.years else 0
        totals = {d: float(v.sum()) for (d, y), v in self.filed.items() if y == last}
        return lineage.build_units(rows, nodes, totals)


# ---- outputs -----------------------------------------------------------------------------------------------------

INT_COLUMNS = {"state_code", "district_code", "year", "is_estimated", "holder_code"}


def _sql_value(column: str, value):
    if pd.isna(value):
        return None
    if isinstance(value, str):
        return value
    return int(value) if column in INT_COLUMNS else float(value)


def write_table(conn, df: pd.DataFrame) -> None:
    rows = [tuple(_sql_value(c, v) for c, v in zip(COLUMNS, row)) for row in df.itertuples(index=False)]
    with conn:
        conn.execute("DELETE FROM panel_rebased")
        conn.executemany("INSERT INTO panel_rebased (" + ", ".join(COLUMNS) + ") VALUES (" + ", ".join("?" * len(COLUMNS)) + ")", rows)


def _pct(n: float, d: float) -> str:
    return f"{100 * n / d:.1f}%" if d else "-"


def write_report(rb: Rebaser, df: pd.DataFrame, units: list[dict], path: Path, changes: dict | None = None) -> str:
    names = rb.names
    dist = df[df["district_code"].notna()]
    n = len(dist)
    counts = dist["share_basis"].value_counts()
    lines = [f"# Re-basing ({date.today().isoformat()})", "",
             f"{n:,} district-years on today's districts, {rb.years[0] if rb.years else ''}-{rb.years[-1] if rb.years else ''}: "
             f"{counts.get('observed', 0):,} observed as filed ({_pct(counts.get('observed', 0), n)}), "
             f"{counts.get('remainder', 0):,} filed rows reduced by the parts of pooled districts, "
             f"{n - counts.get('observed', 0) - counts.get('remainder', 0):,} estimated rows for districts that were pooled.",
             "", "Every row adds back to its filed row, so national totals are unchanged; rows carry today's state, so "
                 "state totals are on today's states.", "",
             "## Share basis of estimated rows", "", "| basis | rows |", "|---|---|"]
    lines += [f"| {b} | {int(counts.get(b, 0)):,} |" for b in BASES[2:]]
    lines += ["", "## Coverage by year", "", "| year | observed | adjusted | estimated | missing |", "|---|---|---|---|---|"]
    lines += [f"| {c['year']} | {c['observed']} | {c['adjusted']} | {c['estimated']} | {c['missing']} |" for c in rb.coverage]
    missing_all = defaultdict(list)
    for c in rb.coverage:
        for d in c["missing_codes"]:
            missing_all[d].append(c["year"])
    lines += ["", f"Missing: a district that filed in other years but not this one, with no lineage fact to pool it anywhere; "
                  f"{len(missing_all)} district(s) affected."]
    for d, ys in sorted(missing_all.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:15]:
        lines.append(f"- {d} {names.get(d, '')}: {', '.join(map(str, ys))}")

    no_ev = sorted({r['edge'] for r in rb.diag.get("no_estimate", [])}, key=lambda p: p.parent_child)
    lines += ["", "## Edges with no share evidence (nothing estimated)", ""]
    lines += [f"- {p.label(names)}" for p in no_ev] or ["None."]
    hm = defaultdict(list)
    for d in rb.diag.get("holder_missing", []):
        hm[d["edge"]].append(d["year"])
    lines += ["", f"## Edges whose holder filed nothing that year ({len(hm)}; the member stays missing)", ""]
    lines += [f"- {p.label(names)}: {', '.join(map(str, sorted(ys)))}" for p, ys in sorted(hm.items(), key=lambda kv: kv[0].parent_child)] or ["None."]
    unresolved = defaultdict(list)
    for d in rb.diag.get("holder_unresolved", []):
        unresolved[d["member"]].append(d["year"])
    lines += ["", f"## Districts left without a row because a holder had none that year ({len(unresolved)})", ""]
    lines += [f"- {names.get(m)}: {', '.join(map(str, sorted(ys)))}" for m, ys in sorted(unresolved.items())] or ["None."]
    zero = [d["edge"] for d in rb.diag.get("zero_weight", [])]
    lines += ["", f"## Edges the old map gives no land ({len(zero)}; nothing moves, the member's other parents carry it)", ""]
    lines += [f"- {p.label(names)}" for p in zero] or ["None."]
    odd = rb.diag.get("conflicting_edges", []) + rb.diag.get("cycles", [])
    lines += ["", f"## Conflicting edges and cycles ({len(odd)})", ""]
    lines += [f"- {d}" for d in odd] or ["None."]
    part_only = [p for p in rb.poolings if p.kind == JOIN and any(y >= p.year for y in rb.filed_years.get(p.member, ()))]
    lines += ["", f"## Late poolings of a part ({len(part_only)}): the pooled district still files, so only a part of it sits in the "
                  "holder's row; nothing is estimated, the two are grouped", ""]
    lines += [f"- {p.label(names)}" for p in part_only] or ["None."]

    cva = rb.crime_vs_area()
    lines += ["", f"## Crime share vs area share, disagreement beyond 2x ({len(cva)} edges, information only)", "",
              "| edge | crime share | basis | area share |", "|---|---|---|---|"]
    lines += [f"| {r['edge'].label(names)} | {r['share']:.2f} | {r['basis']} | {r['area_share']:.2f} |" for r in cva]

    early = rb.filed_before_edge()
    lines += ["", f"## Districts that filed before their lineage year ({len(early)}; rows kept as observed)", ""]
    lines += [f"- {r['edge'].label(names)}: filed in {', '.join(map(str, r['years']))}" for r in early] or ["None."]

    eq = rb.diag.get("equal_weights", [])
    lines += ["", f"## Multi-parent districts divided equally (old map cannot tell) ({len(eq)})", ""]
    lines += [f"- {names.get(e['member'])} <- {', '.join(names.get(h, str(h)) for h in e['holders'])}" for e in eq] or ["None."]

    ov = rb.diag.get("weight_overrides", [])
    lines += ["", f"## Multi-parent districts divided by the decision record, data/split_weights.csv ({len(ov)})", ""]
    lines += [f"- {names.get(o['member'])} <- " + ", ".join(f"{names.get(h, str(h))} {w:.2f}" for h, w in o["holders"].items())
              for o in ov] or ["None."]
    orphans = rb.diag.get("orphan_weights", [])
    lines += ["", f"## Weight rows naming no verified edge ({len(orphans)}; ignored)", ""]
    lines += [f"- {names.get(c, str(c))} <- {names.get(p, str(p))} ({kind})" for c, p, kind in orphans] or ["None."]

    over = rb.diag.get("over_assigned", [])
    lines += ["", f"## Holders whose parts exceeded the filed row (clamped to zero) ({len(over)})", ""]
    lines += [f"- {names.get(o['holder'])} {o['year']}: short by {o['short']:.1f}" for o in over] or ["None."]

    notes = rb.map_only_notes()
    lines += ["", f"## Map-only districts ({len(notes)}): no rows, shown as reported with their holders", ""]
    for nt in notes:
        before = ", ".join(f"{names.get(h)} to {y - 1}" for h, y in nt["inside_before"])
        after = ", ".join(f"{names.get(h)}" + (f" from {y}" if y > (rb.years[0] if rb.years else 0) else "") for h, y in nt["inside_from"])
        lines.append(f"- {nt['code']} {nt['name']}: " + "; ".join(s for s in (before, after) if s) + (" (moves: holders grouped if the area share allows)" if nt["moves"] else ""))

    if changes is not None:
        lines += ["", f"## Stable units changed since the last build ({len(changes['appeared'])} appeared, "
                      f"{len(changes['disappeared'])} disappeared)", ""]
        if not changes["compared"]:
            lines.append("No earlier stable-units file to compare with.")
        else:
            lines += [f"- now: {', '.join(names.get(c, str(c)) for c in g)}" for g in changes["appeared"]]
            lines += [f"- gone: {', '.join(names.get(c, str(c)) for c in g)}" for g in changes["disappeared"]]
            lines += [] if changes["appeared"] or changes["disappeared"] else ["None."]

    sizes = lineage.size_distribution(units)
    grouped = [e for e in rb.group_edges() if e["grouped"]]
    lines += ["", f"## Stable units: {len(units)} groups from {len(grouped)} grouping edges (threshold {rb.threshold})", "",
              "| size | groups |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in sizes.items()]
    text = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def build_rebased(conn, *, parquet_path: Path = PARQUET_PATH, units_path: Path = UNITS_PATH, reports_dir: Path = REPORTS_DIR,
                  area_path: Path = AREA_PATH, weights_path: Path = WEIGHTS_PATH, threshold: float = GROUP_THRESHOLD) -> dict:
    """The orchestrator's step: panel_observed + district_lineage (+ the split weights record) -> panel_rebased, the
    parquet file, stable units (share-gated) and two reports. Returns the numbers the trace records."""
    poolings, skipped = poolings_from_store(conn)
    area = pd.read_csv(area_path) if Path(area_path).exists() else None
    rb = Rebaser(observed_from_store(conn), poolings, districts_from_store(conn), area, threshold=threshold,
                 weights=read_weights(weights_path))
    df = rb.run()
    write_table(conn, df)
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    edges = rb.group_edges()
    units = rb.build_units(edges)
    old_units = json.loads(Path(units_path).read_text(encoding="utf-8"))["units"] if Path(units_path).exists() else None
    changes = units_changes(old_units, units)
    meta = lineage.write_stable_units(units, Path(units_path), extra={
        "threshold": threshold, "basis_years": rb.basis_years,
        "edges": [{k: (v if not isinstance(v, float) else round(v, 4)) for k, v in e.items()} for e in edges],
        "map_only": [{"code": nt["code"], "name": nt["name"], "inside_before": nt["inside_before"], "inside_from": nt["inside_from"]}
                     for nt in rb.map_only_notes()]})
    reports_dir = Path(reports_dir)
    today = date.today().isoformat()
    nodes = [{"district_code": c, "name": d["name"], "in_files": d.get("in_files", 1)} for c, d in rb.districts.items()]
    lineage.write_report(units, reports_dir / f"{today}_stable_units.md", nodes)
    write_report(rb, df, units, reports_dir / f"{today}_rebasing.md", changes)
    dist = df[df["district_code"].notna()]
    counts = dist["share_basis"].value_counts()
    return {"rows": int(len(df)), "district_rows": int(len(dist)), "observed": int(counts.get("observed", 0)),
            "adjusted": int(counts.get("remainder", 0)),
            "estimated": int(len(dist) - counts.get("observed", 0) - counts.get("remainder", 0)),
            "basis": {b: int(counts.get(b, 0)) for b in BASES[2:] if counts.get(b, 0)},
            "missing_district_years": int(sum(c["missing"] for c in rb.coverage)),
            "edges": len(rb.poolings), "edges_no_evidence": len({r["edge"] for r in rb.diag.get("no_estimate", [])}),
            "splits_without_year": len(skipped), "map_only": len(rb.map_only),
            "groups": meta["n_groups"], "largest_group": meta["largest_group_size"], "grouping_edges": meta["n_edges_used"],
            "weight_overrides": len(rb.diag.get("weight_overrides", [])), "orphan_weights": len(rb.diag.get("orphan_weights", [])),
            "units_appeared": len(changes["appeared"]), "units_disappeared": len(changes["disappeared"]),
            "parquet": str(parquet_path.relative_to(ROOT)) if parquet_path.is_relative_to(ROOT) else str(parquet_path)}


def main(argv: list[str] | None = None) -> None:
    from agents import store
    ap = argparse.ArgumentParser(description="Re-base the observed panel on today's districts.")
    ap.add_argument("command", choices=["build"])
    ap.parse_args(argv)
    out = build_rebased(store.connect())
    for k, v in out.items():
        print(f"{k:<24} {v}")


if __name__ == "__main__":
    main()
