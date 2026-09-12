# Data sources and lineage

All crime data below is district-level *Crime Against Women* statistics published by
India's National Crime Records Bureau (NCRB) in its annual *Crime in India* reports,
as redistributed in CSV form by the India Data Portal
(<https://ckandev.indiadataportal.com/dataset/crime-statistics>). We did not
transcribe any of it ourselves; the ingestion layer's job is to *harmonize* these
files, not to create them.

| File | Years | Rows | Schema era | Notes |
|---|---|---|---|---|
| `raw/crimes_against_women_2001-2014.csv` | 2001–2014 | 10,677 | **Era A** — 7 categories | Reporting unit is the police district / railway unit, not the administrative district. Includes `TOTAL` rows per state that must be dropped. State names are spelled inconsistently across years. |
| `raw/districtwise-crime-against-women-2016.csv` | 2016 | 817 | **Era B** — 19 categories | Adds `state_code`, `district_code`, `registration_circles`. |
| `raw/districtwise-crime-against-women-2017-onwards.csv` | 2017–2022 | 5,322 | **Era C** — 42 categories | Same key columns as Era B; categories split by victim age (above/below 18) and add POCSO / cyber offences. |
| `raw/india_district.geojson` | — | 594 features | — | Administrative district boundaries (`NAME_1` = state, `NAME_2` = district). Used for the map and as the canonical district list. |

## Known gaps and caveats (stated up front, on purpose)

- **2015 is missing.** It exists only inside the NCRB *Crime in India 2015* PDF. Filling it is an
  optional later module (PDF extraction); the model tolerates a one-year hole.
- **Three schemas.** The category set changed in 2016 and again in 2017. A verified crosswalk
  maps every era down to the seven categories that exist in all three, so a 22-year time series
  measures the same quantity in every year. Finer-grained columns are preserved but excluded
  from the modelled total.
- **Unit of analysis.** NCRB counts by *police reporting unit* (`registration_circles` in Eras B/C;
  the `DISTRICT` column in Era A, which also contains entries like `guntakal rly.` and `grp`).
  The GeoJSON is administrative districts. Units must be aggregated to districts before joining.
- **District boundaries changed.** Telangana was carved out of Andhra Pradesh in 2014 and many
  districts were split or renamed between 2014 and 2017 (e.g. `Y.S.R.` = `Cuddapah`,
  `Ananthapuramu` = `Anantapur`). Resolving these is the resolver module's job.
- **Zero-count years.** 204 district-years in Era A report zero crimes in total; a percentage
  change from zero is undefined and is treated as missing, not as 0 or infinity.

## Regenerating the profile

`python -m data_ingestion.profile` reads every file above and writes `data/PROFILE.md` with
row counts, unit counts per year, and cross-era name overlap. Re-run it whenever a raw file changes.
