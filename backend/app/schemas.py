"""Response shapes: they validate what the API sends and document it at /docs."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

Status = Literal["watch", "no_signal", "not_scored"]


class DistrictRef(BaseModel):
    code: int
    name: str


class DistrictListItem(DistrictRef):
    status: Status


class State(BaseModel):
    code: int
    name: str
    districts: list[DistrictListItem]


class Districts(BaseModel):
    watch_year: int
    states: list[State]


class Peer(BaseModel):
    """The size group a district is compared within: group g of n, by last year's cases (a district assessed together
    with others counts their combined cases), and its rank in that group."""
    group: int
    groups: int
    rank: int
    size: int
    min_cases: int
    max_cases: int
    cases: int


class District(BaseModel):
    code: int
    name: str
    state_code: int
    state: str
    status: Status
    score: Optional[float]
    peer: Optional[Peer]
    in_files: bool
    latest_year: Optional[int]
    latest_total: Optional[float]
    latest: Optional[dict[str, float]]
    series: list[tuple[int, float, int]]          # [year, total, kind]: 0 as filed, 1 adjusted, 2 estimated
    unit_members: list[DistrictRef]
    notes: list[str]


class WatchItem(BaseModel):
    code: int
    name: str
    state: str
    score: float
    peer: Peer
    latest_year: Optional[int]
    latest_total: Optional[float]


class Watchlist(BaseModel):
    watch_year: int
    districts: list[WatchItem]
