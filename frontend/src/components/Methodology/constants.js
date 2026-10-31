export const STEPS = [
  {
    title: "Ingestion",
    desc: "Four NCRB file formats (2001–2014, the 2015 workbook, 2016 and 2017–2022) are read into one schema, and a reviewed crosswalk maps each era's columns onto seven offence categories.",
    step: "01",
  },
  {
    title: "Name Resolution",
    desc: "Lookups, fuzzy matching and the old district map go first; Gemini answers what is left, inside guards: a fixed candidate list, a completeness check with one repair call, and a calibrated confidence cut.",
    step: "02",
  },
  {
    title: "Lineage and Re-basing",
    desc: "Every district split is recorded with its year. Earlier years are divided by measured crime shares (old-map area as fallback), and districts whose past cannot be separated are grouped into stable units.",
    step: "03",
  },
  {
    title: "Evaluation Harness",
    desc: "Frozen answer keys (119 reviewed names, 100 records re-read from the source files), precision with Wilson intervals, calibration, hallucination counts, and the cost and latency of every model call.",
    step: "04",
  },
  {
    title: "Forecasting Bake-off",
    desc: "Eleven methods tested year by year on 2016–2022, each year forecast only from earlier years; settings chosen on 2016–2018 alone, and a win counted only if a bootstrap interval clears the best baseline.",
    step: "05",
  },
  {
    title: "Continuous Evaluation",
    desc: "On every push, GitHub rebuilds the whole panel from the committed files and checks the metrics against their floors. Prompts are versioned files, and the app is batch-scored.",
    step: "06",
  },
];
