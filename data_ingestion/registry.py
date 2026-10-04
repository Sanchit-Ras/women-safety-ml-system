"""Build the canonical geography from the 2016+ files: states, districts, circles.

No AI is involved. The 2016 and 2017+ files carry an official LGD code on
every row, so the list of districts is *read*, never guessed. Rules:

* one row per district code; display name = the spelling used in the latest year
* aliases = every spelling seen for that code, normalised
* circles = every police reporting unit ("registration circle") seen under the
  code, with first/last year; this is how old-file units such as
  "guntakal rly." will find their district
* a code that stops appearing before the last year is queued for a human: it
  was re-coded, merged or renamed, and only a verified lineage row can say which

Writes data/registry.json (committed) and the states / districts / circles
tables. Run with ``python -m data_ingestion.registry``.
"""
from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import pandas as pd

from agents import store
from data_ingestion.normalize import norm, norm_state

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "registry.json"
LGD_CSV = ROOT / "data" / "lgd_districts.csv"          # written by data_ingestion.geometry: every polygon's code
SOURCES = {
    "2016": RAW / "districtwise-crime-against-women-2016.csv",
    "2017_plus": RAW / "districtwise-crime-against-women-2017-onwards.csv",
}
KEY = ["year", "state_name", "state_code", "district_name", "district_code", "registration_circles"]


def load_keys(sources: dict[str, Path] = SOURCES) -> pd.DataFrame:
    """Only the six key columns of the 2016+ files, all years stacked."""
    return pd.concat([pd.read_csv(p, usecols=KEY) for p in sources.values()], ignore_index=True)


def _latest(g: pd.DataFrame, col: str):
    """The value used in the latest year; if that year has several spellings, the most common one."""
    last = g.loc[g["year"] == g["year"].max(), col].dropna()
    pool = last if len(last) else g[col].dropna()
    return pool.mode().iloc[0]


def build_registry(keys: pd.DataFrame) -> dict:
    keys = keys.dropna(subset=["state_code", "district_code", "district_name"]).copy()
    for col in ("year", "state_code", "district_code"):
        keys[col] = keys[col].astype(int)
    last_year = int(keys["year"].max())

    states = []
    for code, g in keys.groupby("state_code"):
        states.append({"state_code": int(code), "name": str(_latest(g, "state_name")),
                       "aliases": sorted({norm_state(n) for n in g["state_name"].dropna()})})

    districts = []
    for code, g in keys.groupby("district_code"):
        with_circle = g.dropna(subset=["registration_circles"])
        circles = []
        for cnorm, cg in with_circle.groupby(with_circle["registration_circles"].map(norm)):
            circles.append({"norm": cnorm, "name": str(_latest(cg, "registration_circles")),
                            "first_seen_year": int(cg["year"].min()), "last_seen_year": int(cg["year"].max())})
        districts.append({
            "district_code": int(code),
            "state_code": int(_latest(g, "state_code")),            # a district can change state (Leh -> Ladakh)
            "state_codes_seen": sorted({int(s) for s in g["state_code"]}),
            "name": str(_latest(g, "district_name")),
            "aliases": sorted({norm(n) for n in g["district_name"]}),
            "first_seen_year": int(g["year"].min()),
            "last_seen_year": int(g["year"].max()),
            "circles": sorted(circles, key=lambda c: c["norm"]),
        })

    # ---- checks: facts the rest of the pipeline relies on. Measured every build, never assumed.
    name_codes = (keys.assign(dn=keys["district_name"].map(norm))
                      .groupby(["state_code", "dn"])["district_code"].nunique())
    ambiguous_names = [{"state_code": int(s), "name": n, "codes": int(k)} for (s, n), k in name_codes[name_codes > 1].items()]
    disappeared = [{k: d[k] for k in ("district_code", "name", "state_code", "first_seen_year", "last_seen_year")}
                   for d in districts if d["last_seen_year"] < last_year]
    moved_state = [d["district_code"] for d in districts if len(d["state_codes_seen"]) > 1]

    return {
        "meta": {"generated": date.today().isoformat(), "sources": sorted(p.name for p in SOURCES.values()),
                 "years": [int(keys["year"].min()), last_year], "states": len(states), "districts": len(districts),
                 "circles": sum(len(d["circles"]) for d in districts)},
        "checks": {"names_with_more_than_one_code": ambiguous_names,
                   "codes_that_disappear_before_last_year": disappeared,
                   "codes_that_changed_state": moved_state},
        "states": states,
        "districts": districts,
    }


def write_registry(conn, reg: dict, lgd_list: list[dict] = ()) -> dict:
    """Load the registry into the store and queue every unresolved fact for a human.
    ``lgd_list`` (data/lgd_districts.csv) adds the districts that are on the LGD map but in no data file,
    so the unit map and lineage can point at them; they carry ``in_files = 0`` and no seen-years."""
    states = [(s["state_code"], s["name"], json.dumps(s["aliases"])) for s in reg["states"]]
    districts = [(d["district_code"], d["state_code"], d["name"], json.dumps(d["aliases"]),
                  d["first_seen_year"], d["last_seen_year"]) for d in reg["districts"]]
    circles = [(d["district_code"], c["norm"], c["name"], c["first_seen_year"], c["last_seen_year"])
               for d in reg["districts"] for c in d["circles"]]
    known_states = {s["state_code"] for s in reg["states"]}
    seen = {d["district_code"] for d in reg["districts"]}
    map_only = [(int(n["district_code"]), int(n["state_code"]), n["name"]) for n in lgd_list
                if int(n["district_code"]) not in seen and int(n["district_code"]) != 0       # 0 = unassigned polygons
                and int(n["state_code"]) in known_states]
    store.replace_registry(conn, states, districts, circles, map_only=map_only)

    queued, last_year = 0, reg["meta"]["years"][1]
    for d in reg["checks"]["codes_that_disappear_before_last_year"]:
        queued += store.add_escalation(conn, kind="unit", ref={"district_code": d["district_code"]}, violations=[{
            "rule": "code_disappeared",
            "message": (f"code {d['district_code']} '{d['name']}' (state {d['state_code']}) is last seen in "
                        f"{d['last_seen_year']} but the files run to {last_year}. Needs a verified lineage row: "
                        "recode (same district, new code), merge, or rename.")}])
    for a in reg["checks"]["names_with_more_than_one_code"]:
        queued += store.add_escalation(conn, kind="unit", ref={"state_code": a["state_code"], "name": a["name"]},
                                       violations=[{"rule": "name_maps_to_many_codes",
                                                    "message": f"'{a['name']}' appears under {a['codes']} different codes"}])
    return {"states": len(states), "districts": len(districts), "map_only_districts": len(map_only),
            "circles": len(circles), "escalations_queued": queued}


def load_lgd_list(path: Path = LGD_CSV) -> list[dict]:
    """The LGD map's district list, if the geometry step has produced it; empty otherwise."""
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    reg = build_registry(load_keys())
    conn = store.connect()
    summary = write_registry(conn, reg, load_lgd_list())
    OUT.write_text(json.dumps(reg, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"registry: {summary}")
    print(f"years {reg['meta']['years']}")
    for name, items in reg["checks"].items():
        print(f"  {name}: {len(items)}")
        for it in items[:10]:
            print(f"     {it}")
    print(f"written -> {OUT}")


if __name__ == "__main__":
    main()
