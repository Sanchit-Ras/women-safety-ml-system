"""The website draws from a static copy of the API's answers (app/snapshot.py). It must equal what the API returns:
a data change not followed by a new copy fails here, instead of the site quietly showing old numbers."""
import json

from app import snapshot


def test_the_static_snapshot_equals_the_api(client):
    snap = json.loads(snapshot.PATH.read_text(encoding="utf-8"))
    for key, path in (("deploy_check", "/deploy-check"), ("districts", "/districts"), ("watchlist", "/watchlist"),
                      ("model_info", "/model-info")):
        assert snap[key] == client.get(path).json(), f"{key} differs: run python -m ml.export_app"
    listed = {str(d["code"]) for s in snap["districts"]["states"] for d in s["districts"]}
    assert set(snap["district"]) == listed
    stale = [c for c, d in snap["district"].items() if d != client.get(f"/district/{c}").json()]
    assert not stale, f"{len(stale)} districts differ (first: {stale[:3]}): run python -m ml.export_app"
