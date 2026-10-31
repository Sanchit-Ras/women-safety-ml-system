"""Batch scoring for the live app. Run once whenever the data or the bake-off changes; the API only reads the result.

Writes two files, both committed:
  backend/artifacts/app_data.json      every district's measured numbers and notes, the watch list for the year after
                                       the data, and the model card the app shows (copied from the latest bake-off report)
  frontend/public/districts.geojson    today's district map, simplified for the browser, one shape per LGD code

The watch list: the ranking method the bake-off chose (its rule: highest precision at 10% on the validation years)
is trained on every usable window, then scores each stable unit's window for the next year. The top WATCH_SHARE of
scored units are on the list; a district's status is its unit's, because a unit is the smallest group whose past is
real. Why batch and not per request: the data changes once a year, and the API then needs no model at all.

    python -m ml.export_app
"""
from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from agents.schemas import CORE_CATEGORIES
from ml import bakeoff
from ml import panel as unit_panel

ROOT = Path(__file__).resolve().parents[1]
APP_DATA = ROOT / "backend" / "artifacts" / "app_data.json"
MAP_OUT = ROOT / "frontend" / "public" / "districts.geojson"
REGISTRY = ROOT / "data" / "registry.json"
LGD = ROOT / "data" / "lgd_districts.csv"
WATCH_SHARE = 0.10
MAP_TOLERANCE = 0.01      # degrees (about 1 km): enough detail for a country-wide map
MAP_GRID = 0.001          # coordinates snapped to 3 decimals
CATEGORY_LABELS = {"rape": "Rape", "kidnapping_abduction": "Kidnapping and abduction", "dowry_deaths": "Dowry deaths",
                   "assault_on_women": "Assault on women (outraging modesty)", "insult_to_modesty": "Insult to the modesty of women",
                   "cruelty_by_husband": "Cruelty by husband or relatives", "importation_of_girls": "Importation of girls"}


def latest_bakeoff(reports_dir: Path = bakeoff.REPORTS) -> tuple[Path, dict]:
    files = sorted(reports_dir.glob("*_bakeoff.json"))
    if not files:
        raise FileNotFoundError(f"no bake-off report in {reports_dir}: run python -m ml.bakeoff first")
    return files[-1], json.loads(files[-1].read_text(encoding="utf-8"))


def params_of(tag: str) -> dict:
    """'hotspot depth 3, 600 trees' -> the grid entry it names."""
    for g in bakeoff.XGB_GRID:
        if tag.endswith(f"depth {g['max_depth']}, {g['n_estimators']} trees"):
            return dict(g)
    raise ValueError(f"no grid entry matches {tag!r}")


def score_next(panel: pd.DataFrame, neighbours: dict[str, list[str]], params: dict, share: float = WATCH_SHARE) -> tuple[pd.DataFrame, dict]:
    """Train the hotspot classifier on every usable labelled window and score each unit's window for the year after
    the data; the top ``share`` of scored units (at least one) are watched. Returns the scored rows and training facts."""
    rows = bakeoff.usable(bakeoff.windows(panel, neighbours, include_next=True))
    train, nxt = rows[rows["y"].notna()].reset_index(drop=True), rows[rows["y"].isna()].reset_index(drop=True)
    nxt = nxt.assign(score=bakeoff.hotspot(train, nxt, params))
    nxt = nxt.sort_values(["score", "unit_id"], ascending=[False, True], kind="stable").reset_index(drop=True)
    k = max(1, math.ceil(share * len(nxt)))
    nxt["rank"] = np.arange(1, len(nxt) + 1)
    nxt["watch"] = nxt["rank"] <= k
    facts = {"trained_windows": int(len(train)), "trained_label_years": [int(train["label_year"].min()), int(train["label_year"].max())],
             "units_scored": int(len(nxt)), "units_watched": int(k)}
    return nxt[["unit_id", "label_year", "score", "rank", "watch"]], facts


def names() -> tuple[dict[int, dict], dict[int, str]]:
    """District code -> name and state code, and state code -> name: the registry's spelling where it has one (the
    names the pipeline uses), the LGD list for districts that are only on the map."""
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    states = {int(s["state_code"]): s["name"] for s in reg["states"]}
    dists = {int(d["district_code"]): {"name": d["name"], "state_code": int(d["state_code"])} for d in reg["districts"]}
    for r in pd.read_csv(LGD).itertuples(index=False):
        code = int(r.district_code)
        if code and code not in dists:
            dists[code] = {"name": r.name, "state_code": int(r.state_code)}
        states.setdefault(int(r.state_code), str(r.state_name).title())
    return dists, states


FILED, ADJUSTED, ESTIMATED = 0, 1, 2      # the third value of a series point: how that year's number came about


def kind_of(basis: str) -> int:
    return FILED if basis == "observed" else ADJUSTED if basis == "remainder" else ESTIMATED


def build(rebased: pd.DataFrame, units_data: dict, scored: pd.DataFrame, dist_names: dict[int, dict], state_names: dict[int, str],
          map_codes: set[int], pieces: pd.DataFrame | None = None) -> dict[str, dict]:
    """One entry per district on today's map that the data or the lineage knows: measured numbers, status and notes.
    A series point is [year, total, kind]: kind 0 = as filed, 1 = adjusted (the filed row minus the parts that are
    today's districts carved out of it), 2 = estimated (a share of the row of the district it was part of)."""
    gave = {}
    if pieces is not None and len(pieces):
        gave = pieces.groupby("holder_code")["member_code"].agg(lambda s: sorted({int(x) for x in s})).to_dict()
        took = pieces.groupby("member_code")["holder_code"].agg(lambda s: sorted({int(x) for x in s})).to_dict()
    else:
        took = {}
    units = {u["unit_id"]: [int(m) for m in u["members"]] for u in units_data["units"]}
    unit_of = {m: u for u, ms in units.items() for m in ms}
    map_only = {int(n["code"]): n for n in units_data["meta"].get("map_only", [])}
    by_unit = {r.unit_id: r for r in scored.itertuples(index=False)}
    dist = rebased[rebased["district_code"].notna()].copy()
    dist["district_code"] = dist["district_code"].astype(int)
    state_now = dist.drop_duplicates("district_code").set_index("district_code")["state_code"].astype(int).to_dict()
    nm = lambda c: dist_names.get(c, {}).get("name", str(c))                                   # noqa: E731
    out = {}
    for code in sorted((set(dist["district_code"]) | set(map_only)) & map_codes):
        rows = dist[dist["district_code"] == code].sort_values("year")
        unit = unit_of.get(code)
        others = [m for m in units.get(unit, []) if m != code]
        state_code = state_now.get(code, dist_names.get(code, {}).get("state_code"))
        notes = []
        if code in map_only:
            holders = sorted({h for h, _y in map_only[code]["inside_from"]} | {h for h, _y in map_only[code]["inside_before"]})
            notes.append(f"Has no row of its own in the NCRB files: its crimes are counted under {', '.join(nm(h) for h in holders)}.")
        if others:
            notes.append(f"Assessed together with {', '.join(nm(m) for m in others)}: their past cannot be separated, so they share one status.")
        kinds = [kind_of(b) for b in rows["share_basis"]]
        est = [y for y, k in zip(rows["year"], kinds) if k == ESTIMATED]
        adj = [y for y, k in zip(rows["year"], kinds) if k == ADJUSTED]
        if est:
            src = ", ".join(nm(h) for h in took.get(code, [])) or "the district it was part of"
            notes.append(f"{len(est)} of its {len(rows)} years ({min(est)}-{max(est)}) are estimated as a share of {src}, which it was "
                         "part of then; they are drawn hollow on the chart.")
        if adj:
            kids = ", ".join(nm(m) for m in gave.get(code, [])) or "districts carved out of it later"
            notes.append(f"{len(adj)} of its {len(rows)} years ({min(adj)}-{max(adj)}) are its filed numbers minus the part that is "
                         f"today's {kids}; they are drawn lighter on the chart.")
        s = by_unit.get(unit)
        if s is not None:
            status, score, rank = ("watch" if s.watch else "no_signal"), round(float(s.score), 4), int(s.rank)
        else:
            status, score, rank = "not_scored", None, None
            if code not in map_only:
                notes.append("Not scored: its unit lacks six recent years of comparable, mostly filed numbers.")
        latest = rows.iloc[-1] if len(rows) else None
        out[str(code)] = {
            "code": code, "name": nm(code), "state_code": state_code, "state": state_names.get(state_code, str(state_code)),
            "unit_id": unit, "unit_members": sorted(units.get(unit, [code])), "status": status, "score": score, "rank": rank,
            "in_files": code not in map_only,
            "latest_year": int(latest["year"]) if latest is not None else None,
            "latest_total": round(float(latest["total"]), 1) if latest is not None else None,
            "latest": {c: round(float(latest[c]), 1) for c in CORE_CATEGORIES} if latest is not None else None,
            "series": [[int(r.year), round(float(r.total), 1), kind_of(r.share_basis)] for r in rows.itertuples(index=False)],
            "notes": notes}
    return out


def model_card(report_path: Path, report: dict, facts: dict, watch_year: int, data_years: list[int]) -> dict:
    rec = report["ranking_recommended"]
    rk = report["ranking"][rec]
    t = report["scores"]["test"]
    best = report["best_baseline"]
    point = report["recommended"]
    return {
        "generated": date.today().isoformat(), "data_years": data_years, "watch_year": watch_year, "watch_share": WATCH_SHARE,
        "model": {"name": f"{rec} classifier (XGBoost)", "setting": report["chosen"][rec], "features": len(report["features"]), **facts},
        "evidence": {"report": f"ml/reports/{report_path.stem}.md", "test_years": report["test_years"],
                     "precision_at_10": round(rk["test"], 3), "ci95": [round(x, 3) for x in rk["test_ci95"]], "chance": 0.1},
        "size_of_change": {"best_baseline": best, "baseline_mae": round(t[best]["mae"], 1), "model": point,
                           "model_mae": round(t[point]["mae"], 1),
                           "statement": "No method forecast the size of next year's change better than assuming no change, so the app "
                                        "shows no forecast percentage."},
        "limitations": ["The watch list measures growth, not how unsafe a place is: a district going from 5 cases to 10 can be on "
                        "it while a district with thousands of cases is not, so small districts appear on it often.",
                        "A nationwide shock such as the 2020 lockdown cannot be foreseen from any district's history.",
                        "Counts are crimes reported to the police; a rise can mean more reporting as well as more crime."],
        "categories": CATEGORY_LABELS}


def clockwise(geom):
    """Outer rings clockwise, holes anticlockwise: the winding d3-geo needs. With the other winding a spherical
    projection reads a district as the whole globe minus the district and paints the map over."""
    if isinstance(geom, Polygon):
        return orient(geom, sign=-1.0)
    if isinstance(geom, MultiPolygon):
        return MultiPolygon([orient(p, sign=-1.0) for p in geom.geoms])
    return MultiPolygon([orient(p, sign=-1.0) for p in getattr(geom, "geoms", []) if isinstance(p, Polygon)])


def simplify_map(src: Path = unit_panel.MAP_PATH, tolerance: float = MAP_TOLERANCE, grid: float = MAP_GRID,
                 dist_names: dict[int, dict] | None = None) -> dict:
    """Today's districts, one feature per LGD code (parts merged), simplified, snapped to a grid and wound clockwise
    for the browser."""
    data = json.loads(Path(src).read_text(encoding="utf-8"))
    parts: dict[int, list] = {}
    for f in data["features"]:
        code = f["properties"].get("dist_lgd")
        if code in (None, 0) or f.get("geometry") is None:
            continue
        geom = shape(f["geometry"])
        parts.setdefault(int(code), []).append(geom if geom.is_valid else geom.buffer(0))
    features = []
    for code in sorted(parts):
        geom = clockwise(shapely.set_precision(unary_union(parts[code]).simplify(tolerance, preserve_topology=True), grid))
        if geom.is_empty:
            continue
        name = (dist_names or {}).get(code, {}).get("name")
        features.append({"type": "Feature", "properties": {"code": code, **({"name": name} if name else {})}, "geometry": mapping(geom)})
    return {"type": "FeatureCollection", "features": features}


def main() -> None:
    report_path, report = latest_bakeoff()
    params = params_of(report["chosen"][report["ranking_recommended"]])
    panel, neighbours = unit_panel.build()
    scored, facts = score_next(panel, neighbours, params)
    rebased = pd.read_parquet(unit_panel.REBASED_PATH)
    units_data = json.loads(unit_panel.UNITS_PATH.read_text(encoding="utf-8"))
    dist_names, state_names = names()
    geo = simplify_map(dist_names=dist_names)
    map_codes = {f["properties"]["code"] for f in geo["features"]}
    pieces = pd.read_csv(unit_panel.PIECES_PATH) if unit_panel.PIECES_PATH.exists() else None
    districts = build(rebased, units_data, scored, dist_names, state_names, map_codes, pieces)
    years = [int(rebased["year"].min()), int(rebased["year"].max())]
    watch_year = int(scored["label_year"].iloc[0]) if len(scored) else years[1] + 1
    app = {"meta": model_card(report_path, report, facts, watch_year, years),
           "states": {str(c): n for c, n in sorted(state_names.items()) if any(d["state_code"] == c for d in districts.values())},
           "districts": districts}
    APP_DATA.parent.mkdir(parents=True, exist_ok=True)
    APP_DATA.write_text(json.dumps(app, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    MAP_OUT.parent.mkdir(parents=True, exist_ok=True)
    MAP_OUT.write_text(json.dumps(geo, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    watched = sorted((d for d in districts.values() if d["status"] == "watch"), key=lambda d: (d["rank"], d["name"]))
    print(f"{len(districts)} districts; {facts['units_scored']} units scored, {facts['units_watched']} watched "
          f"({len(watched)} districts) for {watch_year}; trained on {facts['trained_windows']:,} windows")
    print(f"app data -> {APP_DATA} ({APP_DATA.stat().st_size / 1e6:.2f} MB)\nmap      -> {MAP_OUT} ({MAP_OUT.stat().st_size / 1e6:.2f} MB, "
          f"{len(geo['features'])} shapes)")
    print("top 10:", ", ".join(f"{d['name']} ({d['state']})" for d in watched[:10]))


if __name__ == "__main__":
    main()
