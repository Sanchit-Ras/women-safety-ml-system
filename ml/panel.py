"""The table the forecasting models learn from: one row per stable unit and year.

A stable unit (agents/rebase.py, data/stable_units.json) is a group of today's districts whose combined total
is real in every year: a district and the parent it was carved from, for instance. Training on districts would
count an estimated child and its adjusted parent twice and let a parent's later years leak into its child's
earlier ones. A unit's series is a sum of filed rows, exact except where an estimated piece crosses the unit's
boundary, and data/rebased_pieces.csv says how much did.

Columns: unit_id, state_code (the largest member's), year, the seven categories, total, n_members, outside_est
(cases estimated across the unit's boundary that year, into it or out of it), outside_share (outside_est / total).
``unit_neighbours`` maps a unit to the units whose districts touch any of its districts on today's map.

    python -m ml.panel          # a summary of the table
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import shape
from shapely.strtree import STRtree

from agents.schemas import CORE_CATEGORIES

ROOT = Path(__file__).resolve().parents[1]
REBASED_PATH = ROOT / "data" / "panel_rebased.parquet"
UNITS_PATH = ROOT / "data" / "stable_units.json"
PIECES_PATH = ROOT / "data" / "rebased_pieces.csv"
MAP_PATH = ROOT / "data" / "districts_simplified.geojson"
TOUCH = 0.002        # degrees; the simplified map's boundaries do not meet exactly, so polygons this close count as touching
COLUMNS = ["unit_id", "state_code", "year", *CORE_CATEGORIES, "total", "n_members", "outside_est", "outside_share"]


def load_units(path: Path = UNITS_PATH) -> dict[str, list[int]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {u["unit_id"]: [int(m) for m in u["members"]] for u in data["units"]}


def build_panel(rebased: pd.DataFrame, units: dict[str, list[int]], pieces: pd.DataFrame | None = None) -> pd.DataFrame:
    """Unit-years from the re-based district rows: members summed, the unit's state taken from its largest member,
    and the estimated cases that crossed the unit's boundary that year (in or out) from the pieces log."""
    member_of = {m: u for u, ms in units.items() for m in ms}
    dist = rebased[rebased["district_code"].notna()].copy()
    dist["district_code"] = dist["district_code"].astype(int)
    dist = dist[dist["district_code"].isin(member_of)]
    dist["unit_id"] = dist["district_code"].map(member_of)
    size = dist.groupby(["unit_id", "district_code"])["total"].sum()
    largest = {u: key[1] for u, key in size.groupby(level=0).idxmax().items()}
    state_of = dist.drop_duplicates("district_code").set_index("district_code")["state_code"].astype(int)
    unit_state = {u: int(state_of[code]) for u, code in largest.items()}
    g = dist.groupby(["unit_id", "year"], as_index=False)[[*CORE_CATEGORIES, "total"]].sum()
    g["n_members"] = g["unit_id"].map({u: len(ms) for u, ms in units.items()})
    g["state_code"] = g["unit_id"].map(unit_state)
    flows: dict[tuple[str, int], float] = defaultdict(float)
    if pieces is not None and len(pieces):
        for r in pieces.itertuples(index=False):
            mu, hu = member_of.get(int(r.member_code)), member_of.get(int(r.holder_code))
            if mu == hu:
                continue                                       # a piece moved inside one unit leaves its total exact
            if mu is not None:
                flows[(mu, int(r.year))] += float(r.amount)      # estimated into the unit
            if hu is not None:
                flows[(hu, int(r.year))] += float(r.amount)      # estimated out of the unit
    g["outside_est"] = [flows.get((u, int(y)), 0.0) for u, y in zip(g["unit_id"], g["year"])]
    g["outside_share"] = np.where(g["total"] > 0, g["outside_est"] / g["total"].replace(0, np.nan), 0.0)
    return g[COLUMNS].sort_values(["unit_id", "year"]).reset_index(drop=True)


def district_neighbours(map_path: Path = MAP_PATH, touch: float = TOUCH) -> dict[int, set[int]]:
    """Districts whose polygons touch (within ``touch`` degrees) on today's map, by LGD code."""
    data = json.loads(Path(map_path).read_text(encoding="utf-8"))
    codes, geoms = [], []
    for f in data["features"]:
        code = f["properties"].get("dist_lgd")
        if code in (None, 0) or f.get("geometry") is None:
            continue
        geom = shape(f["geometry"])
        if geom.is_empty:
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)                                  # the simplified map has a few self-touching rings
        codes.append(int(code))
        geoms.append(geom.buffer(touch))
    tree = STRtree(geoms)
    out: dict[int, set[int]] = defaultdict(set)
    for i, geom in enumerate(geoms):
        for j in tree.query(geom, predicate="intersects"):
            if codes[j] != codes[i]:
                out[codes[i]].add(codes[j])
                out[codes[j]].add(codes[i])
    return dict(out)


def unit_neighbours(units: dict[str, list[int]], district_nbrs: dict[int, set[int]]) -> dict[str, list[str]]:
    member_of = {m: u for u, ms in units.items() for m in ms}
    return {u: sorted({member_of[d] for m in ms for d in district_nbrs.get(m, ()) if d in member_of} - {u})
            for u, ms in units.items()}


def build(rebased_path: Path = REBASED_PATH, units_path: Path = UNITS_PATH, pieces_path: Path = PIECES_PATH,
          map_path: Path = MAP_PATH) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    units = load_units(units_path)
    pieces = pd.read_csv(pieces_path) if Path(pieces_path).exists() else None
    return build_panel(pd.read_parquet(rebased_path), units, pieces), unit_neighbours(units, district_neighbours(map_path))


def main() -> None:
    panel, nbrs = build()
    years = panel.groupby("unit_id")["year"].nunique()
    print(f"{panel['unit_id'].nunique()} units, {len(panel):,} unit-years, {panel['year'].min()}-{panel['year'].max()}; "
          f"{int((years == years.max()).sum())} units with every year")
    print(f"unit-years with an outside estimate: {int((panel['outside_share'] > 0).sum()):,} "
          f"(over 5%: {int((panel['outside_share'] > 0.05).sum()):,}; over 20%: {int((panel['outside_share'] > 0.2).sum()):,})")
    print(f"neighbours per unit: median {np.median([len(v) for v in nbrs.values()]):.0f}, none for {sum(not v for v in nbrs.values())}")


if __name__ == "__main__":
    main()
