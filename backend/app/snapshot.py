"""The static copy of the API's answers that the website draws from first.

The API runs on a free server that sleeps when nobody uses it, and waking it takes about half a minute. Every answer
is computed ahead of time anyway (see services/data.py), so the same answers are saved once into the frontend as a
static file, served from Vercel's CDN with the page. The site draws from it at once; the live API is for programs.

The file holds exactly what the API returns, asked in process through FastAPI's test client, so the two cannot
differ in shape: tests/test_snapshot.py fails if the committed file no longer equals the API's answers.
Written by ``python -m ml.export_app`` (which calls ``write``); needs httpx, a development tool not in the image.
"""
from __future__ import annotations

import json
from pathlib import Path

PATH = Path(__file__).resolve().parents[2] / "frontend" / "public" / "snapshot.json"


def build() -> dict:
    from fastapi.testclient import TestClient     # imported here: the image has no httpx and never calls this

    from app.main import app
    from app.services import data

    data.get.cache_clear()                         # read the data file as it is now, not as an earlier import saw it
    with TestClient(app) as client:
        def get(path: str):
            r = client.get(path)
            r.raise_for_status()
            return r.json()

        districts = get("/districts")
        codes = [d["code"] for s in districts["states"] for d in s["districts"]]
        return {"deploy_check": get("/deploy-check"), "districts": districts, "watchlist": get("/watchlist"),
                "model_info": get("/model-info"), "district": {str(c): get(f"/district/{c}") for c in codes}}


def write(path: Path = PATH) -> dict:
    snap = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snap, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return snap
