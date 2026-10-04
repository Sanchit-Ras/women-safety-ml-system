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
| `raw/ncrb_district_wise_crimes_against_women_2015.xlsx` | 2015 | 815 district rows | **Era 2015** — 60 heads | NCRB's own workbook, *Crime in India 2015 → Additional Tables → District Wise Reports*: <https://www.ncrb.gov.in/uploads/2022/July/11/custom/crime-in-india/5-District-wise-Crimes-committed-against-Women_2015.xlsx> (279,426 bytes, sha256 `f0c049c0c0521888…`, downloaded 2026-10-04). One sheet, header in row 2, `State: <name>` marker rows, `Total District(s)` rows per state, totals and their parts side by side (`Kidnapping & Abduction_Total` + five parts). No LGD codes: units are police districts, resolved like Era A. |
| `raw/districtwise-crime-against-women-2016.csv` | 2016 | 817 | **Era B** — 19 categories | Adds `state_code`, `district_code`, `registration_circles`. |
| `raw/districtwise-crime-against-women-2017-onwards.csv` | 2017–2022 | 5,322 | **Era C** — 42 categories | Same key columns as Era B; categories split by victim age (above/below 18) and add POCSO / cyber offences. |
| `raw/india_district.geojson` | — | 594 features | — | Old (pre-2014) administrative boundaries (`NAME_1` = state, `NAME_2` = district). Kept for area-overlap checks between old and current districts; no longer the map. |
| `raw/LGD_Districts.geojson` (not committed: 93 MB) | — | 785 features | — | Current district boundaries from [bharatlas](https://bharatlas.com/view/lgd_districts), LGD 2024 snapshot, CC0-1.0 / CC-BY-4.0. Every feature carries the LGD codes (`dist_lgd`, `state_lgd`), so the map joins by code. All 754 codes in our files are present; the 30 extra districts were created after 2022. Re-download: `curl -L -o data/raw/LGD_Districts.geojson https://pub-0429b8e3b5a946e69ea007df844a6f1c.r2.dev/admin/districts/LGD_Districts.geojson`. A simplified copy for the map will be committed later. |
| `reference/ncrb_published_national.csv` | 2015–2022 | 40 figures | — | NCRB's own published national totals per crime head (MoSPI *Women and Men in India 2020* tables 6.1–6.2, plus news reports of Crime in India 2020–2022). The fidelity check adds our district rows up and compares them with these. |
| `column_fixes.csv` | 2018–2022 | 5 rows | — | **Known defect in the source file:** from 2018 the five "below 18" columns of the 2017+ file are rotated by one position (values sit under the neighbouring header). Found by the fidelity check, proven to the case against NCRB's published rape, assault, insult, attempt and kidnapping figures. The raw file is never edited; loaders apply these verified moves at load time. |

## Known gaps and caveats (stated up front, on purpose)

- **2015 comes from a different shape of file.** The India Data Portal has no 2015 district file; NCRB
  publishes it as an Excel workbook (row above). Its district rows add up to slightly less than the
  published national figures (rape 34,637 vs 34,651; kidnapping 59,081 vs 59,283; cruelty 113,256 vs
  113,403; assault and insult exact): NCRB's district tables leave out a few state-level reporting
  units. Within the fidelity tolerance for every core head.
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
