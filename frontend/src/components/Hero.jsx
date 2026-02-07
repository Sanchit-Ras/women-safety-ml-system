export default function Hero() {
  return (
    <div className="relative bg-slate-900 text-white overflow-hidden">
      {/* Background Decor */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden z-0 opacity-20">
        <div className="absolute -top-24 -left-24 w-96 h-96 bg-blue-500 rounded-full blur-3xl"></div>
        <div className="absolute top-1/2 right-0 w-80 h-80 bg-purple-500 rounded-full blur-3xl"></div>
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-6 py-24 md:py-32 flex flex-col items-center text-center">
        <span className="bg-blue-900/50 text-blue-200 text-sm font-semibold px-4 py-1.5 rounded-full border border-blue-700/50 mb-6 animate-fade-in">
          Women's Safety Machine Learning Project
        </span>

        <h1 className="text-5xl md:text-7xl font-display font-bold leading-tight mb-6 bg-clip-text text-transparent bg-linear-to-r from-white to-slate-400">
          Mapping Safety Through <br />
          <span className="text-blue-500">Machine Intelligence</span>
        </h1>

      <p className="mt-6 text-lg text-slate-600 max-w-3xl mx-auto">
        A research-driven machine learning project forecasting district-level
        crime risk across India using time-series modeling and spatial analysis.
      </p>

      <p className="mt-2 text-sm text-slate-500">
        Built using 14+ years of crime data • District-level forecasting •
        Research-grade validation
      </p>
      <p className="mt-3 text-red-600 font-semibold">
        Built using Random Forest, XGBoost, District Holdout Validation, and
        Temporal Forecasting.
      </p>
    </div>
  </div>
  );
}
