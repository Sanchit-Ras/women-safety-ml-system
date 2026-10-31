from fastapi import APIRouter, HTTPException

from app.schemas import District, Districts, Watchlist
from app.services import data

router = APIRouter()


@router.get("/districts", response_model=Districts)
def districts():
    return {"watch_year": data.meta()["watch_year"], "states": data.states()}


@router.get("/district/{code}", response_model=District)
def district(code: int):
    d = data.district(code)
    if d is None:
        raise HTTPException(status_code=404, detail=f"no district with code {code}")
    return d


@router.get("/watchlist", response_model=Watchlist)
def watchlist():
    return {"watch_year": data.meta()["watch_year"], "districts": data.watchlist()}
