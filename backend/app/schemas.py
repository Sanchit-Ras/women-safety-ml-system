"""Response shapes: they validate what the API sends and document it at /docs."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

Status = Literal["watch", "no_signal", "not_scored"]
Band = Literal["fell", "steady", "rose", "rose_sharply", "few_cases"]


class DistrictRef(BaseModel):
    code: int
    name: str


class DistrictListItem(DistrictRef):
    status: Status
    trend: Optional[Band]                          # the map's colour: how reported cases changed (Trend.band)


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


class Trend(BaseModel):
    """Reported cases in the first and the latest year of the comparison, and the change, beside the state's and India's
    change over the same years. Scope 'area': the district's own numbers are not as filed in both years, so these are
    the combined numbers of the districts it is assessed with (unit_members). A from_year later than the usual one: the
    first year the district was counted with today's borders. Band 'few_cases': too few cases in the first year for a
    fair percentage."""
    scope: Literal["district", "area"]
    from_year: int
    to_year: int
    from_total: float
    to_total: float
    change: Optional[float]
    band: Band
    state_change: Optional[float]
    india_change: Optional[float]


class Rank(BaseModel):
    """Place in its state by reported cases in that year: counts, not rates, so larger districts rank higher."""
    year: int
    rank: int
    of: int


class Top(BaseModel):
    """The most common offence in the latest year, as a share of the district's cases and of its state's."""
    category: str
    share: float
    state_share: Optional[float]


class Summary(BaseModel):
    trend: Optional[Trend]
    rank: Optional[Rank]
    top: Optional[Top]


class District(BaseModel):
    code: int
    name: str
    state_code: int
    state: str
    status: Status
    score: Optional[float]
    peer: Optional[Peer]
    summary: Summary
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
