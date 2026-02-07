export default function Methodology() {
  const steps = [
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

  return (
    <div className="bg-white py-24">
      <div className="max-w-7xl mx-auto px-6">
        <div className="mb-16 md:text-center max-w-3xl mx-auto">
          <h2 className="text-4xl font-display font-bold text-slate-900 mb-6">
            Research Methodology
          </h2>
          <p className="text-lg text-slate-600 leading-relaxed">
            This project was engineered using research-grade machine learning
            practices to ensure predictive reliability rather than superficial
            accuracy.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {steps.map((step, i) => (
            <div
              key={i}
              className="group relative bg-slate-50 p-8 rounded-3xl border border-slate-100 hover:bg-white hover:shadow-xl transition-all duration-300"
            >
              <div className="absolute top-6 right-6 text-6xl font-display font-bold text-slate-200 group-hover:text-blue-100 transition-colors select-none">
                {step.step}
              </div>

              <div className="relative z-10">
                <h3 className="text-xl font-bold text-slate-900 mb-4 pr-12 group-hover:text-blue-700 transition-colors">
                  {step.title}
                </h3>
                <p className="text-slate-600 leading-relaxed text-sm">
                  {step.desc}
                </p>
              </div>

              <div className="absolute bottom-0 left-0 w-full h-1 bg-linear-to-r from-blue-500 to-purple-500 transform scale-x-0 group-hover:scale-x-100 transition-transform duration-300 rounded-b-3xl"></div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
