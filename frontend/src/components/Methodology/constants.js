export const STEPS = [
  {
    title: "Data Engineering",
    desc: "Aggregated district-level crime records across 14 years and constructed year-over-year percentage change features to normalize regional reporting differences.",
    step: "01",
  },
  {
    title: "Stationarity Transformation",
    desc: "Applied Z-score normalization to stabilize variance across districts, enabling the model to learn patterns independent of absolute crime volume.",
    step: "02",
  },
  {
    title: "Sliding Window Supervision",
    desc: "Converted temporal crime sequences into supervised learning samples using a 6-year lag window to forecast future crime change.",
    step: "03",
  },
  {
    title: "District Holdout Validation",
    desc: "Used GroupShuffleSplit to exclude entire districts during training — preventing geographic memorization and ensuring true generalization.",
    step: "04",
  },
  {
    title: "Model Architecture",
    desc: "Benchmarked Random Forest and XGBoost regressors. Tree ensembles were selected due to their robustness against non-linear socio-crime relationships.",
    step: "05",
  },
  {
    title: "Overfitting Diagnostics",
    desc: "Compared training, validation, and future-test R² scores to confirm the model learned transferable crime dynamics.",
    step: "06",
  },
];
