"""Canonical record shapes shared by every module in the ingestion layer.

Every source era (2001-2014, 2016, 2017+) is converted into ``CrimeRecord``
before anything downstream (crosswalk, resolver, validator, ML features)
touches it. Keeping one shape here is what stops three source schemas from
leaking into the rest of the codebase.

``ResolutionResult`` and ``Violation`` arrive with the resolver and validator
modules respectively; they are not defined until they have a consumer.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, computed_field

Era = Literal["2001_2014", "2016", "2017_plus"]

# The seven categories that exist, with the same legal definition, in all
# three source eras. This is the lowest common denominator: later eras can be
# aggregated *down* to these, but 2001-2014 can never be split *up* into the
# finer 2017+ categories. Anything modelled over the full time series must be
# built from these and nothing else.
CORE_CATEGORIES: tuple[str, ...] = (
    "rape",
    "kidnapping_abduction",
    "dowry_deaths",
    "assault_on_women",
    "insult_to_modesty",
    "cruelty_by_husband",
    "importation_of_girls",
)


class CrimeRecord(BaseModel):
    """One reporting unit, one year, the seven core categories.

    ``district`` is the name exactly as the source spells it. Canonical IDs are
    attached later by the resolver and never guessed here, so that a record can
    always be traced back to the row it came from.
    """

    state: str
    district: str
    registration_circle: Optional[str] = None
    state_code: Optional[int] = None
    district_code: Optional[int] = None
    year: int = Field(ge=2001, le=2030)

    rape: int = Field(ge=0)
    kidnapping_abduction: int = Field(ge=0)
    dowry_deaths: int = Field(ge=0)
    assault_on_women: int = Field(ge=0)
    insult_to_modesty: int = Field(ge=0)
    cruelty_by_husband: int = Field(ge=0)
    importation_of_girls: int = Field(ge=0)

    era: Era
    source_file: str
    source_row: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total(self) -> int:
        return sum(getattr(self, c) for c in CORE_CATEGORIES)
