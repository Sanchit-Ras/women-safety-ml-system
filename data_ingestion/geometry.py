"""Old district map vs current LGD map: where each old district's land is now.

The 2001-2014 file names districts as they were before 2014; the 2016+ files
carry LGD codes for the districts as they are now. Many districts were split,
merged or renamed in between. The old GADM map (594 polygons, old names) and
the current LGD map (785 polygons, codes) let us answer "where is that old
district today" by geography instead of by guessing from the name.

For every old polygon this module records the fraction of its land inside each
modern polygon (``share``) and the fraction of that modern polygon it fills
(``new_share``). A rename shows up as one row with share 1.0; a split as
several rows whose shares add up to 1.0. The table is evidence for the
resolver (old name -> code) and the lineage step (splits), and later the area
cross-check for crime shares.

Areas are computed in WGS84 degrees, not square kilometres: pyproj is not
installed and one degree of longitude shrinks towards the poles. That
distorts *absolute* areas, but the shares within one old polygon are ratios of
areas measured on the same patch of the earth, so at district scale they are
only slightly affected. The note is written into the meta file too.

    python -m data_ingestion.geometry overlap    # -> data/area_overlap.csv + area_overlap_meta.json, prints checks
    python -m data_ingestion.geometry lgd-list   # -> data/lgd_districts.csv (every LGD polygon, in_files flag)
    python -m data_ingestion.geometry simplify   # -> data/districts_simplified.geojson for the web map
    python -m data_ingestion.geometry all        # the three in that order
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path

import pandas as pd
import shapely
from shapely import STRtree, make_valid, unary_union
from shapely.geometry import mapping, shape

from data_ingestion.normalize import norm, norm_state

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OLD_PATH = RAW / "india_district.geojson"
NEW_PATH = RAW / "LGD_Districts.geojson"
REGISTRY_PATH = ROOT / "data" / "registry.json"
OVERLAP_PATH = ROOT / "data" / "area_overlap.csv"
META_PATH = ROOT / "data" / "area_overlap_meta.json"
LGD_LIST_PATH = ROOT / "data" / "lgd_districts.csv"
SIMPLIFIED_PATH = ROOT / "data" / "districts_simplified.geojson"

MIN_SHARE = 0.01                       # below 1% of an old polygon is a boundary sliver, not a real overlap
COVER_THRESHOLD = 0.95                 # an old polygon whose kept shares add up to less than this is reported
DOUBLE_THRESHOLD = 1.05                # ... and more than this means two modern polygons overlap each other
SIMPLIFY_TOLERANCES = (0.005, 0.01, 0.02)
SIMPLIFY_LIMIT_BYTES = 12 * 1024 * 1024
COORD_DECIMALS = 5                     # ~1 m; far below the coarsest simplify tolerance, so it costs no shape

# The old map predates three renames. Applied after norm_state so its own aliases run first.
OLD_STATE_ALIASES: dict[str, str] = {
    "orissa": "odisha",
    "uttaranchal": "uttarakhand",
    "andaman and nicobar": "andaman and nicobar islands",
}

OVERLAP_COLUMNS = ["old_state", "old_name", "old_norm", "old_state_norm",
                   "new_code", "new_name", "new_state_code", "share", "new_share"]
LGD_LIST_COLUMNS = ["district_code", "name", "state_code", "state_name", "in_files"]
AREA_NOTE = ("Areas are computed in WGS84 degrees (no projection library installed). Absolute areas are "
             "distorted by latitude; the shares within one old polygon are ratios of areas on the same patch "
             "of ground and are only slightly affected at district scale.")


# ---- loading ------------------------------------------------------------------------------

def _read_geojson(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:            # json.load: the text is never kept, only the parsed tree
        return json.load(fh)


def old_state_norm(value: str) -> str:
    s = norm_state(value)
    return OLD_STATE_ALIASES.get(s, s)


def valid_polygon(geometry: dict):
    """Parse a GeoJSON geometry and make it valid. A self-touching or spiked ring would make
    ``intersection`` raise or return nonsense. ``make_valid`` may hand back a GeometryCollection
    (the polygon plus the zero-area lines it cut off) or, for a collinear ring, a bare line.
    Only the polygonal parts are a district; a ring with no area becomes an empty Polygon so the
    rest of the module can rely on ``geom_type`` and ``is_empty`` instead of checking for lines."""
    geom = make_valid(shape(geometry))
    parts = [g for g in shapely.get_parts(geom) if g.geom_type in ("Polygon", "MultiPolygon")]
    return unary_union(parts) if parts else shapely.Polygon()


def load_old_from(obj: dict) -> list[dict]:
    """Old map features -> {state, name, state_norm, name_norm, geom}."""
    out = []
    for f in obj["features"]:
        p = f["properties"]
        out.append({"state": p["NAME_1"], "name": p["NAME_2"], "state_norm": old_state_norm(p["NAME_1"]),
                    "name_norm": norm(p["NAME_2"]), "geom": valid_polygon(f["geometry"])})
    return out


def load_new_from(obj: dict) -> list[dict]:
    """LGD map features -> {code, name, state_code, state_name, name_norm, geom}."""
    out = []
    for f in obj["features"]:
        p = f["properties"]
        out.append({"code": int(p["dist_lgd"]), "name": p["dtname"], "state_code": int(p["state_lgd"]),
                    "state_name": p["stname"], "name_norm": norm(p["dtname"]), "geom": valid_polygon(f["geometry"])})
    return out


def load_old(path: Path = OLD_PATH) -> list[dict]:
    return load_old_from(_read_geojson(Path(path)))


def load_new(path: Path = NEW_PATH) -> list[dict]:
    return load_new_from(_read_geojson(Path(path)))


# ---- the overlap table ----------------------------------------------------------------------

def overlap(old: list[dict], new: list[dict], min_share: float = MIN_SHARE, progress_every: int = 100) -> pd.DataFrame:
    """One row per (old polygon, modern polygon) pair that shares at least ``min_share`` of the
    old polygon's area. Pure over the two lists, so tests feed unit squares."""
    new_geoms = [d["geom"] for d in new]
    new_areas = shapely.area(new_geoms) if new else []
    tree = STRtree(new_geoms)
    rows = []
    for i, o in enumerate(old):
        if progress_every and i and i % progress_every == 0:
            print(f"  overlap: {i}/{len(old)} old polygons", flush=True)
        old_area = o["geom"].area
        if old_area == 0:
            continue
        hits = tree.query(o["geom"], predicate="intersects")        # bounding boxes, then exact test
        inter_areas = shapely.area(shapely.intersection(o["geom"], [new_geoms[j] for j in hits]))
        for j, inter in zip(hits, inter_areas):
            share = inter / old_area
            if inter <= 0 or share < min_share:      # 'intersects' also matches edge contact: no land, no row
                continue
            n = new[j]
            rows.append({"old_state": o["state"], "old_name": o["name"], "old_norm": o["name_norm"],
                         "old_state_norm": o["state_norm"], "new_code": n["code"], "new_name": n["name"],
                         "new_state_code": n["state_code"], "share": float(share),
                         "new_share": float(inter / new_areas[j]) if new_areas[j] else 0.0})
    df = pd.DataFrame(rows, columns=OVERLAP_COLUMNS)
    return (df.sort_values(["old_state", "old_name", "share"], ascending=[True, True, False], kind="stable")
              .reset_index(drop=True))


def checks(df: pd.DataFrame, old: list[dict] = (), new: list[dict] = ()) -> dict:
    """Facts about the table, measured not assumed. Pass ``old`` and ``new`` to also name the
    polygons that never appear in it: the table alone cannot know what is missing from it."""
    sums = df.groupby(["old_state", "old_name"])["share"].sum() if len(df) else pd.Series(dtype=float)
    seen_old = set(sums.index)
    seen_new = set(zip(df["new_code"], df["new_name"])) if len(df) else set()
    return {
        "old_polygons": len(old),
        "old_with_rows": int(len(sums)),
        "share_sum": {"min": float(sums.min()) if len(sums) else None,
                      "max": float(sums.max()) if len(sums) else None,
                      f"below_{COVER_THRESHOLD}": int((sums < COVER_THRESHOLD).sum()),
                      f"above_{DOUBLE_THRESHOLD}": int((sums > DOUBLE_THRESHOLD).sum())},
        "old_without_overlap": [{"state": o["state"], "name": o["name"]}
                                for o in old if (o["state"], o["name"]) not in seen_old],
        "new_without_cover": [{"code": n["code"], "name": n["name"], "state_name": n["state_name"]}
                              for n in new if (n["code"], n["name"]) not in seen_new],
    }


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows({c: r[c] for c in columns} for r in rows)


def write_overlap(old: list[dict], new: list[dict], min_share: float = MIN_SHARE, *,
                  out_path: Path = OVERLAP_PATH, meta_path: Path = META_PATH) -> tuple[pd.DataFrame, dict]:
    """Compute the table, write it and a meta file that records how it was made and what the checks found."""
    df = overlap(old, new, min_share)
    rows = df.assign(share=df["share"].round(6), new_share=df["new_share"].round(6)).to_dict("records")
    _write_csv(out_path, rows, OVERLAP_COLUMNS)
    meta = {"generated": date.today().isoformat(), "old_source": OLD_PATH.name, "new_source": NEW_PATH.name,
            "old_polygons": len(old), "new_polygons": len(new), "rows": int(len(df)), "min_share": min_share,
            "checks": checks(df, old, new), "area_note": AREA_NOTE}
    Path(meta_path).write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    return df, meta


def read_overlap(path: Path = OVERLAP_PATH) -> pd.DataFrame:
    """The committed table, for the resolver and the lineage step."""
    return pd.read_csv(path, dtype={"new_code": int, "new_state_code": int})


# ---- exports for the registry check and the web map ------------------------------------------

def write_lgd_list(new: list[dict], registry_path: Path = REGISTRY_PATH, out_path: Path = LGD_LIST_PATH) -> int:
    """Every LGD polygon with a flag saying whether its code appears in the crime files (the registry).
    One row per polygon: the map has two features with code 0 (unassigned territory), kept as they are."""
    reg = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    in_files = {int(d["district_code"]) for d in reg["districts"]}
    rows = sorted(({"district_code": d["code"], "name": d["name"], "state_code": d["state_code"],
                    "state_name": d["state_name"], "in_files": int(d["code"] in in_files)} for d in new),
                  key=lambda r: (r["state_code"], r["district_code"], r["name"]))
    _write_csv(Path(out_path), rows, LGD_LIST_COLUMNS)
    return len(rows)


def _round_coords(coords):
    if isinstance(coords, (int, float)):
        return round(coords, COORD_DECIMALS)
    return [_round_coords(c) for c in coords]


def simplify(new: list[dict], tolerance_deg: float = SIMPLIFY_TOLERANCES[0], out_path: Path = SIMPLIFIED_PATH) -> int:
    """Write a FeatureCollection with only the four join properties and simplified geometries;
    return its size in bytes. Coordinates are rounded because a 15-digit float per vertex is
    most of the file and none of the shape. A feature with no area is left out and named, so
    the web map never receives something that is not a district polygon."""
    skipped = [d["code"] for d in new if d["geom"].is_empty]
    if skipped:
        print(f"  simplify: skipped {len(skipped)} feature(s) with no area: codes {skipped}", flush=True)
    features = []
    for d in new:
        if d["geom"].is_empty:
            continue
        geom = mapping(d["geom"].simplify(tolerance_deg, preserve_topology=True))
        features.append({"type": "Feature",
                         "properties": {"dist_lgd": d["code"], "dtname": d["name"],
                                        "state_lgd": d["state_code"], "stname": d["state_name"]},
                         "geometry": {"type": geom["type"], "coordinates": _round_coords(geom["coordinates"])}})
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, separators=(",", ":")),
                        encoding="utf-8")
    return out_path.stat().st_size


def simplify_within(new: list[dict], out_path: Path = SIMPLIFIED_PATH, tolerances=SIMPLIFY_TOLERANCES,
                    limit_bytes: int = SIMPLIFY_LIMIT_BYTES) -> tuple[float, int]:
    """Try each tolerance in order until the file fits under the limit; return (tolerance, bytes) of the
    file left on disk. The last tolerance is written even when it still does not fit, so the caller sees it."""
    tolerance = size = None
    for tolerance in tolerances:
        size = simplify(new, tolerance, out_path)
        print(f"  simplify: tolerance {tolerance} deg -> {size / 1024 / 1024:.1f} MB", flush=True)
        if size <= limit_bytes:
            break
    return tolerance, size


# ---- CLI --------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Old map vs LGD map: area overlap table, LGD list, simplified map.")
    ap.add_argument("command", choices=["overlap", "lgd-list", "simplify", "all"])
    ap.add_argument("--min-share", type=float, default=MIN_SHARE)
    ap.add_argument("--tolerance", type=float, default=SIMPLIFY_TOLERANCES[0], help="first simplify tolerance, degrees")
    args = ap.parse_args(argv)

    new = load_new(NEW_PATH)                                       # 93 MB: read once, whatever the command
    print(f"loaded {len(new)} LGD polygons from {NEW_PATH.name}")
    if args.command in ("overlap", "all"):
        old = load_old(OLD_PATH)
        print(f"loaded {len(old)} old polygons from {OLD_PATH.name}")
        df, meta = write_overlap(old, new, args.min_share)
        print(f"overlap: {len(df)} rows -> {OVERLAP_PATH}")
        print(json.dumps(meta["checks"], indent=1, ensure_ascii=False))
        print(f"meta -> {META_PATH}")
    if args.command in ("lgd-list", "all"):
        n = write_lgd_list(new)
        print(f"lgd-list: {n} rows -> {LGD_LIST_PATH}")
    if args.command in ("simplify", "all"):
        tolerances = [args.tolerance] + [t for t in SIMPLIFY_TOLERANCES if t > args.tolerance]
        tolerance, size = simplify_within(new, SIMPLIFIED_PATH, tolerances)
        fits = "fits" if size <= SIMPLIFY_LIMIT_BYTES else "still above"
        print(f"simplify: tolerance {tolerance} deg, {size:,} bytes ({fits} the {SIMPLIFY_LIMIT_BYTES // 1024 // 1024} MB "
              f"limit) -> {SIMPLIFIED_PATH}")


if __name__ == "__main__":
    main()
