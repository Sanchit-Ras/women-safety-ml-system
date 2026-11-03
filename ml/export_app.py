"""Batch scoring for the live app. Run once whenever the data or the bake-off changes; the API only reads the result.

Writes four files, all committed:
  backend/artifacts/app_data.json      every district's measured numbers, plain-language summary and notes, the watch
                                       list for the year after the data, and the model card the app shows (copied from
                                       the latest bake-off report)
  frontend/public/districts.geojson    today's district map, simplified for the browser, one shape per LGD code
  frontend/public/data/district_crimes_against_women.csv
                                       the re-based panel as one plain table, for anyone to download
  frontend/public/snapshot.json        the API's own answers, saved once, so the site draws at once instead of waiting
                                       for the free server to wake up (backend/tests check it still equals the API)

The summary: a district's reported cases in TREND_FROM and in the latest year, beside its state's and India's change
over the same years, its rank in its state, and its most common offence. Worked out here, once, so the API and the
site only display it.

The watch list: the ranking method the bake-off chose (its rule: highest precision at 10% on the validation years)
is trained on every usable window, then scores each stable unit's window for the next year. The top WATCH_SHARE of
scored units are on the list; a district's status is its unit's, because a unit is the smallest group whose past is
real. Why batch and not per request: the data changes once a year, and the API then needs no model at all.

    python -m ml.export_app
"""
from __future__ import annotations

import json
import math
import sys
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
DATASET = ROOT / "frontend" / "public" / "data" / "district_crimes_against_women.csv"
REGISTRY = ROOT / "data" / "registry.json"
LGD = ROOT / "data" / "lgd_districts.csv"
WATCH_SHARE = 0.10
TREND_FROM = 2017         # from 2017 NCRB's district file carries today's district codes, so almost every district has
                          # numbers as filed in every year since; the window also starts after the 2013 change in the law
MIN_CASES = 30            # below this many cases in the first year, a percentage turns on a handful of cases
BANDS = (("fell", -0.10), ("steady", 0.10), ("rose", 0.50), ("rose_sharply", math.inf))   # the map's colours: change below each bound
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


def score_next(panel: pd.DataFrame, neighbours: dict[str, list[str]], params: dict, groups: int = 1,
               share: float = WATCH_SHARE) -> tuple[pd.DataFrame, dict]:
    """Train the hotspot classifier (top tenth within ``groups`` size groups) on every usable labelled window and score
    each unit's window for the year after the data. Units are split into the same number of size groups by their latest
    year's cases; the top ``share`` of each group (at least one) are watched. Returns the scored rows, each with its
    group, its rank in the group and the group's size range, and the training facts."""
    rows = bakeoff.usable(bakeoff.windows(panel, neighbours, include_next=True))
    train, nxt = rows[rows["y"].notna()].reset_index(drop=True), rows[rows["y"].isna()].reset_index(drop=True)
    nxt = nxt.assign(score=bakeoff.hotspot(train, nxt, params, groups), group=bakeoff.size_groups(nxt, groups) + 1)
    nxt = nxt.sort_values(["score", "unit_id"], ascending=[False, True], kind="stable").reset_index(drop=True)
    nxt["rank"] = nxt.groupby("group").cumcount() + 1
    size = nxt.groupby("group")["unit_id"].transform("size")
    nxt["group_size"] = size
    nxt["watch"] = nxt["rank"] <= np.maximum(1, np.ceil(share * size))
    nxt["group_min"] = nxt.groupby("group")["count_prev"].transform("min")
    nxt["group_max"] = nxt.groupby("group")["count_prev"].transform("max")
    facts = {"trained_windows": int(len(train)), "trained_label_years": [int(train["label_year"].min()), int(train["label_year"].max())],
             "units_scored": int(len(nxt)), "units_watched": int(nxt["watch"].sum()), "size_groups": int(groups)}
    return nxt[["unit_id", "label_year", "score", "group", "rank", "group_size", "group_min", "group_max", "watch", "count_prev"]], facts


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
HOW_COUNTED = {FILED: "filed", ADJUSTED: "adjusted", ESTIMATED: "estimated"}


def kind_of(basis: str) -> int:
    return FILED if basis == "observed" else ADJUSTED if basis == "remainder" else ESTIMATED


def change_of(before: float, after: float) -> float | None:
    return None if not before else round((after - before) / before, 4)


def band_of(change: float | None, from_total: float, min_cases: int = MIN_CASES) -> str:
    """The map's colour for a change; 'few_cases' when the first year had too few cases for a fair percentage."""
    if change is None or from_total < min_cases:
        return "few_cases"
    return next(name for name, below in BANDS if change < below)


def summaries(rebased: pd.DataFrame, units: dict[str, list[int]], start: int = TREND_FROM,
              min_cases: int = MIN_CASES) -> tuple[dict[int, dict], dict]:
    """The plain-language facts the site shows for each district with numbers:
      trend  reported cases in ``start`` and in the latest year, and the change, beside its state's and India's change
             over the same years. Its own numbers when both years are as filed; otherwise those of its unit (the
             districts whose past cannot be separated: their combined numbers are real in every year), scope 'area';
             otherwise its own numbers from the first later year it was counted as it is today, beside its state's and
             India's change over those same, shorter years. None when its borders changed in the latest year itself.
      rank   its place in its state by reported cases in the latest year: counts, not rates
      top    its most common offence in the latest year as a share of its cases, beside the same share for its state
    Also returns the comparison's settings, India's total in every year (the site's first chart, so a visitor sees all
    the years before the recent change) and the trend of India and of each state, for the site's overview."""
    end = int(rebased["year"].max())
    state_rows = {int(s): rows for s, rows in rebased.groupby("state_code")}

    def totals(rows: pd.DataFrame, since: int) -> tuple[float, float]:
        return tuple(float(rows.loc[rows["year"] == y, "total"].sum()) for y in (since, end))

    def trend(rows: pd.DataFrame, since: int = start) -> dict:
        before, after = totals(rows, since)
        return {"from_total": round(before, 1), "to_total": round(after, 1), "change": change_of(before, after)}

    india = trend(rebased)
    states = {s: trend(rows) for s, rows in state_rows.items()}
    dist = rebased[rebased["district_code"].notna()].assign(district_code=lambda d: d["district_code"].astype(int),
                                                           kind=lambda d: d["share_basis"].map(kind_of))
    first, last = (dist[dist["year"] == y].set_index("district_code") for y in (start, end))
    state_last = rebased[rebased["year"] == end].groupby("state_code")[[*CORE_CATEGORIES, "total"]].sum()
    unit_of = {m: u for u, ms in units.items() for m in ms}
    out = {}
    for code, rows in dist.groupby("district_code"):
        state = int(rows["state_code"].iloc[-1])
        members = units.get(unit_of.get(code), [code])
        filed = set(rows.loc[rows["kind"] == FILED, "year"])
        later = sorted(y for y in filed if start < y < end)
        scope, since = None, start
        if {start, end} <= filed:
            scope = "district"
            before, after = float(first.at[code, "total"]), float(last.at[code, "total"])
        elif len(members) > 1 and first.index.isin(members).any() and last.index.isin(members).any():
            scope = "area"
            before, after = (float(f.loc[f.index.isin(members), "total"].sum()) for f in (first, last))
        elif end in filed and later:
            scope, since = "district", later[0]
            before, after = totals(rows, since)
        t = None
        if scope:
            change = change_of(before, after)
            t = {"scope": scope, "from_year": since, "to_year": end, "from_total": round(before, 1), "to_total": round(after, 1),
                 "change": change, "band": band_of(change, before, min_cases),
                 "state_change": states[state]["change"] if since == start else change_of(*totals(state_rows[state], since)),
                 "india_change": india["change"] if since == start else change_of(*totals(rebased, since))}
        rank = top = None
        if code in last.index:
            peers = last[last["state_code"] == state]
            rank = {"year": end, "rank": int((peers["total"] > last.at[code, "total"]).sum()) + 1, "of": int(len(peers))}
            row = last.loc[code]
            if row["total"] > 0:
                cat = max(CORE_CATEGORIES, key=lambda c: row[c])          # the first of equals, in the categories' order
                s = state_last.loc[state]
                top = {"category": cat, "share": round(float(row[cat] / row["total"]), 4),
                       "state_share": round(float(s[cat] / s["total"]), 4) if s["total"] else None}
        out[int(code)] = {"trend": t, "rank": rank, "top": top}
    per_year = rebased.groupby("year")["total"].sum()
    meta = {"from_year": start, "to_year": end, "min_cases": min_cases,
            "bands": [{"band": name, "below": None if math.isinf(below) else below} for name, below in BANDS],
            "india": india, "india_series": [[int(y), round(float(t), 1), FILED] for y, t in per_year.items()],
            "states": {str(s): v for s, v in sorted(states.items())}}
    return out, meta


def dataset(rebased: pd.DataFrame, dist_names: dict[int, dict], state_names: dict[int, str]) -> pd.DataFrame:
    """The re-based panel as one plain table to download: a row per district and year, the seven offence categories
    that exist in every year's files, their total, and how that year's numbers came about. Rows of reporting units that
    belong to no district (a few state-level units before 2016) are left out."""
    d = rebased[rebased["district_code"].notna()]
    codes, states = d["district_code"].astype(int), d["state_code"].astype(int)
    out = pd.DataFrame({"state_code": states, "state": states.map(lambda c: state_names.get(c, str(c))),
                        "district_code": codes, "district": codes.map(lambda c: dist_names.get(c, {}).get("name", str(c))),
                        "year": d["year"].astype(int), **{c: d[c].round(1) for c in CORE_CATEGORIES}, "total": d["total"].round(1),
                        "how_counted": d["share_basis"].map(lambda b: HOW_COUNTED[kind_of(b)])})
    return out.sort_values(["state", "district", "district_code", "year"], kind="stable").reset_index(drop=True)


def write_snapshot() -> Path:
    """The API's own answers for every page of the site, asked in process and saved as one static file. The backend
    decides the shape (backend/app/snapshot.py); its tests fail if the committed file no longer equals the API."""
    sys.path.insert(0, str(ROOT / "backend"))
    from app import snapshot                                                   # the backend's package, imported only here
    snapshot.write()
    return snapshot.PATH


def build(rebased: pd.DataFrame, units_data: dict, scored: pd.DataFrame, dist_names: dict[int, dict], state_names: dict[int, str],
          map_codes: set[int], pieces: pd.DataFrame | None = None, facts: dict[int, dict] | None = None) -> dict[str, dict]:
    """One entry per district on today's map that the data or the lineage knows: measured numbers, summary, status
    and notes. A series point is [year, total, kind]: kind 0 = as filed, 1 = adjusted (the filed row minus the parts
    that are today's districts carved out of it), 2 = estimated (a share of the row of the district it was part of)."""
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
            status, score = ("watch" if s.watch else "no_signal"), round(float(s.score), 4)
            peer = {"group": int(s.group), "groups": int(scored["group"].max()), "rank": int(s.rank), "size": int(s.group_size),
                    "min_cases": round(float(s.group_min)), "max_cases": round(float(s.group_max)), "cases": round(float(s.count_prev))}
        else:
            status, score, peer = "not_scored", None, None
            if code not in map_only:
                notes.append("Not scored: its unit lacks six recent years of comparable, mostly filed numbers.")
        latest = rows.iloc[-1] if len(rows) else None
        out[str(code)] = {
            "code": code, "name": nm(code), "state_code": state_code, "state": state_names.get(state_code, str(state_code)),
            "unit_id": unit, "unit_members": sorted(units.get(unit, [code])), "status": status, "score": score, "peer": peer,
            "summary": (facts or {}).get(code, {"trend": None, "rank": None, "top": None}),
            "in_files": code not in map_only,
            "latest_year": int(latest["year"]) if latest is not None else None,
            "latest_total": round(float(latest["total"]), 1) if latest is not None else None,
            "latest": {c: round(float(latest[c]), 1) for c in CORE_CATEGORIES} if latest is not None else None,
            "series": [[int(r.year), round(float(r.total), 1), kind_of(r.share_basis)] for r in rows.itertuples(index=False)],
            "notes": notes}
    return out


def model_card(report_path: Path, report: dict, facts: dict, watch_year: int, data_years: list[int]) -> dict:
    rec = report["ranking_recommended"]
    g = report["size_groups"]
    gt = g["test"]
    t = report["scores"]["test"]
    best = report["best_baseline"]
    point = report["recommended"]
    return {
        "generated": date.today().isoformat(), "data_years": data_years, "watch_year": watch_year, "watch_share": WATCH_SHARE,
        "model": {"name": f"{rec} classifier (XGBoost)", "setting": report["chosen"][rec], "features": len(report["features"]), **facts},
        "evidence": {"report": f"ml/reports/{report_path.stem}.md", "test_years": report["test_years"], "size_groups": g["chosen"],
                     "precision_at_10": round(gt["model"], 3), "ci95": [round(x, 3) for x in gt["ci95"]], "chance": 0.1,
                     "size_only": round(max(gt["smallest_first"], gt["largest_first"]), 3),
                     "by_group": [{"group": b["group"], "min_cases": round(b["min_cases"]), "max_cases": round(b["max_cases"]),
                                   "precision": None if b["precision"] is None else round(b["precision"], 3)} for b in g["by_group"]]},
        "size_of_change": {"best_baseline": best, "baseline_mae": round(t[best]["mae"], 1), "model": point,
                           "model_mae": round(t[point]["mae"], 1),
                           "statement": "No method forecast the size of next year's change better than assuming no change, so the app "
                                        "shows no forecast percentage."},
        "limitations": ["The watch list measures growth compared with districts of similar size, not how unsafe a place is: a "
                        "district with few cases can be on it while a district with thousands of cases is not.",
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
    scored, facts = score_next(panel, neighbours, params, groups=report["size_groups"]["chosen"])
    rebased = pd.read_parquet(unit_panel.REBASED_PATH)
    units_data = json.loads(unit_panel.UNITS_PATH.read_text(encoding="utf-8"))
    dist_names, state_names = names()
    geo = simplify_map(dist_names=dist_names)
    map_codes = {f["properties"]["code"] for f in geo["features"]}
    pieces = pd.read_csv(unit_panel.PIECES_PATH) if unit_panel.PIECES_PATH.exists() else None
    units = {u["unit_id"]: [int(m) for m in u["members"]] for u in units_data["units"]}
    district_facts, trend = summaries(rebased, units)
    districts = build(rebased, units_data, scored, dist_names, state_names, map_codes, pieces, district_facts)
    table = dataset(rebased, dist_names, state_names)
    years = [int(rebased["year"].min()), int(rebased["year"].max())]
    watch_year = int(scored["label_year"].iloc[0]) if len(scored) else years[1] + 1
    meta = {**model_card(report_path, report, facts, watch_year, years), "trend": trend,
            "dataset": {"path": f"/{DATASET.relative_to(ROOT / 'frontend' / 'public').as_posix()}", "rows": len(table),
                        "districts": int(table["district_code"].nunique()), "columns": list(table.columns)}}
    app = {"meta": meta,
           "states": {str(c): n for c, n in sorted(state_names.items()) if any(d["state_code"] == c for d in districts.values())},
           "districts": districts}
    APP_DATA.parent.mkdir(parents=True, exist_ok=True)
    APP_DATA.write_text(json.dumps(app, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    MAP_OUT.parent.mkdir(parents=True, exist_ok=True)
    MAP_OUT.write_text(json.dumps(geo, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    DATASET.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(DATASET, index=False, float_format="%.10g", encoding="utf-8", lineterminator="\n")
    snap = write_snapshot()
    watched = sorted((d for d in districts.values() if d["status"] == "watch"), key=lambda d: (-d["score"], d["name"]))
    print(f"{len(districts)} districts; {facts['units_scored']} units scored, {facts['units_watched']} watched "
          f"({len(watched)} districts) for {watch_year}; trained on {facts['trained_windows']:,} windows")
    for label, path in (("app data", APP_DATA), ("map", MAP_OUT), ("dataset", DATASET), ("snapshot", snap)):
        print(f"{label:<9}-> {path} ({path.stat().st_size / 1e6:.2f} MB)")
    print("top 10:", ", ".join(f"{d['name']} ({d['state']})" for d in watched[:10]))
    print(f"India {trend['from_year']}-{trend['to_year']}: {trend['india']['from_total']:,.0f} -> {trend['india']['to_total']:,.0f} "
          f"({trend['india']['change']:+.1%})")


if __name__ == "__main__":
    main()
