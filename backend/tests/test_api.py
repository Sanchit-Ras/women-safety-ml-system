"""The API against the committed data file: every endpoint, unknown codes, CORS for the deployed frontend, and the
promise the app makes: no forecast percentage anywhere."""
import json
import re

from app import API_VERSION
from app.services import data


def test_health_and_deploy_check(client):
    assert client.get("/health").json() == {"status": "ok"}
    d = client.get("/deploy-check").json()
    assert d["api_version"] == API_VERSION == client.get("/openapi.json").json()["info"]["version"]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", d["data_generated"]) and d["watch_year"] > 2022


def test_districts_lists_every_district_once_under_its_state(client):
    body = client.get("/districts").json()
    codes = [d["code"] for s in body["states"] for d in s["districts"]]
    assert len(codes) == len(set(codes)) == len(data.get()["districts"])
    assert [s["name"] for s in body["states"]] == sorted(s["name"] for s in body["states"])
    delhi = next(s for s in body["states"] if s["name"] == "Delhi")
    assert {"East", "North East", "Shahdara"} <= {d["name"] for d in delhi["districts"]}


def test_a_district_carries_its_numbers_status_and_notes(client):
    d = client.get("/district/671").json()                         # Shahdara: estimated before 2017, pooled with its parents
    assert d["name"] == "Shahdara" and d["state"] == "Delhi" and d["status"] in ("watch", "no_signal")
    assert {m["name"] for m in d["unit_members"]} == {"East", "North East"}
    years = [p[0] for p in d["series"]]
    assert years == sorted(years) and years[0] == 2001 and years[-1] == d["latest_year"] == 2022
    assert all(p[2] == 2 for p in d["series"] if p[0] < 2017) and all(p[2] == 0 for p in d["series"] if p[0] >= 2017)
    assert d["latest_total"] == sum(d["latest"].values()) and any("estimated" in n for n in d["notes"])
    j = client.get("/district/689").json()                         # Jangaon: on the map, no row of its own
    assert (j["status"], j["in_files"], j["series"], j["latest"]) == ("not_scored", False, [], None)
    assert "Warangal" in j["notes"][0]


def test_a_district_carries_both_comparisons_and_the_list_their_map_colours(client):
    m = client.get("/model-info").json()
    w = m["trends"]
    assert (w["long"]["from_year"], w["recent"]["from_year"]) == (m["data_years"][0], 2017)
    assert [p[0] for p in m["india_series"]] == list(range(m["data_years"][0], m["data_years"][1] + 1))
    labels = {n: {b["label"] for b in w[n]["bands"]} | {w[n]["few_cases"]} for n in w}
    d = client.get("/district/671").json()                         # Shahdara: filed from 2017, estimated before
    by_year = {p[0]: p[1] for p in d["series"]}
    recent, long = d["summary"]["recent"], d["summary"]["long"]
    assert (recent["scope"], recent["from_year"], recent["to_year"]) == ("district", 2017, 2022) and recent["band"] in labels["recent"]
    assert (recent["from_total"], recent["to_total"]) == (by_year[2017], by_year[2022])
    assert round((recent["to_total"] - recent["from_total"]) / recent["from_total"], 4) == recent["change"]
    assert (long["scope"], long["from_year"]) == ("area", 2001) and long["band"] in labels["long"]   # 2001 only as a whole area
    assert long["india_change"] == w["long"]["india"]["change"] and recent["india_change"] == w["recent"]["india"]["change"]
    assert d["summary"]["rank"]["of"] == 11 and d["latest"][d["summary"]["top"]["category"]] == max(d["latest"].values())
    assert client.get("/district/689").json()["summary"] == {"long": None, "recent": None, "rank": None, "top": None}
    listed = {x["code"]: x for s in client.get("/districts").json()["states"] for x in s["districts"]}
    assert (listed[671]["long_band"], listed[671]["recent_band"]) == (long["band"], recent["band"])
    assert listed[689]["long_band"] is listed[689]["recent_band"] is None
    warangal = client.get("/district/522").json()["summary"]["long"]               # counted with today's borders from 2019
    assert warangal["from_year"] > 2001 and listed[522]["long_band"] is None             # so not coloured on the 22-year map
    for n in w:
        assert {x[f"{n}_band"] for x in listed.values()} <= labels[n] | {None}
    assert m["dataset"]["path"].endswith(".csv") and m["dataset"]["rows"] > 16000


def test_unknown_or_malformed_codes(client):
    assert client.get("/district/999999").status_code == 404
    assert client.get("/district/abc").status_code == 422


def test_watchlist_is_ordered_and_agrees_with_the_districts(client):
    w = client.get("/watchlist").json()
    scores = [d["score"] for d in w["districts"]]
    assert w["districts"] and scores == sorted(scores, reverse=True)
    watched = {c for c, d in data.get()["districts"].items() if d["status"] == "watch"}
    assert {str(d["code"]) for d in w["districts"]} == watched
    for item in w["districts"]:
        d = client.get(f"/district/{item['code']}").json()
        assert d["status"] == "watch" and d["peer"] == item["peer"]
        assert d["peer"]["rank"] <= -(-d["peer"]["size"] // 10)                # inside the top tenth of its size group


def test_districts_of_one_unit_share_one_status_and_size_group():
    by_unit = {}
    for d in data.get()["districts"].values():
        by_unit.setdefault(d["unit_id"], set()).add((d["status"], json.dumps(d["peer"], sort_keys=True)))
    assert all(len(v) == 1 for v in by_unit.values())


def test_size_groups_cover_every_scored_district_without_overlap():
    peers = [d["peer"] for d in data.get()["districts"].values() if d["peer"]]
    groups = {p["group"]: (p["min_cases"], p["max_cases"]) for p in peers}
    assert sorted(groups) == list(range(1, peers[0]["groups"] + 1))
    bounds = [groups[g] for g in sorted(groups)]
    assert all(lo <= hi for lo, hi in bounds) and all(a[1] <= b[0] for a, b in zip(bounds, bounds[1:]))
    assert all(p["min_cases"] <= p["cases"] <= p["max_cases"] for p in peers)


def test_model_info_states_the_evidence_and_the_limits(client):
    m = client.get("/model-info").json()
    e = m["evidence"]
    assert e["chance"] == 0.1 and e["ci95"][0] > e["chance"] and e["ci95"][0] <= e["precision_at_10"] <= e["ci95"][1]
    assert m["model"]["units_watched"] <= m["model"]["units_scored"] and m["limitations"]
    assert m["size_of_change"]["model_mae"] >= m["size_of_change"]["baseline_mae"]   # why no percentage is shown


def test_no_response_carries_a_forecast_percentage(client):
    bodies = [client.get(p).text for p in ("/districts", "/watchlist", "/district/671", "/district/502")]
    for b in bodies:
        assert "predicted_change" not in b and "risk_level" not in b
    assert "predicted_change" not in json.dumps(data.get())


def test_cors_allows_the_vercel_frontend_and_only_reads(client):
    r = client.get("/health", headers={"Origin": "https://women-safety-ml-system-jwlh.vercel.app"})
    assert r.headers.get("access-control-allow-origin") == "https://women-safety-ml-system-jwlh.vercel.app"
    r = client.get("/health", headers={"Origin": "https://evil.example.com"})
    assert "access-control-allow-origin" not in r.headers


def test_the_old_prediction_endpoints_are_gone(client):
    assert client.get("/predict_district", params={"state": "x", "district": "y"}).status_code == 404
    assert client.post("/predict", json={}).status_code in (404, 405)
