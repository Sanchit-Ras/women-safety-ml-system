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

Era = Literal["2001_2014", "2015", "2016", "2017_plus"]     # mirrored by store.ERA_ROWS, which the database enforces

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

# Harmonised fine categories for the 2016+ files. The first seven are the core
# categories; the rest exist only from 2016 (or 2017) and are kept and shown for
# those years, never stretched back to 2001. One decision per source column:
# which of these does it belong to. A column may take a core name only if the
# 2001-2014 NCRB head would have counted the same cases.
EXTENDED_CATEGORIES: tuple[str, ...] = CORE_CATEGORIES + (
    "attempt_to_rape",              # IPC 376/511, a separate NCRB head since 2014
    "murder_with_rape",             # IPC 376A (2017+)
    "acid_attack",                  # IPC 326A
    "attempt_to_acid_attack",       # IPC 326B
    "abetment_of_suicide",          # IPC 306
    "miscarriage_offences",         # IPC 312-318
    "human_trafficking",            # IPC 370 / 370A
    "unnatural_offences",           # IPC 377 (2016 file only)
    "dowry_prohibition_act",        # Dowry Prohibition Act, 1961
    "domestic_violence_act",        # Protection of Women from Domestic Violence Act, 2005
    "immoral_traffic_act",          # Immoral Traffic (Prevention) Act, 1956
    "indecent_representation_act",  # Indecent Representation of Women (Prohibition) Act, 1986
    "pocso_act",                    # POCSO Act, 2012, all heads (2017+)
    "cyber_crimes",                 # IT Act and other women-centric cyber crimes (2017+)
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
