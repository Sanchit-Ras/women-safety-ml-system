import { useEffect, useState } from "react";
import RiskExplanation from "../components/RiskExplanation";
import PredictionMeaning from "../components/PredictionsMeaning";

export default function Predictor() {
  const [data, setData] = useState({});
  const [state, setState] = useState("");
  const [district, setDistrict] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showInfo, setShowInfo] = useState(false);

  // Fetch districts
  useEffect(() => {
    fetch(`${import.meta.env.VITE_API_URL}/districts`)
      .then((res) => res.json())
      .then((data) => setData(data))
      .catch(() => setError("Failed to load districts"));
  }, []);

  const predict = async () => {
    if (!state || !district) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const res = await fetch(
        `${import.meta.env.VITE_API_URL}/predict_district?state=${state}&district=${district}`,
      );

      const json = await res.json();

      if (json.error) {
        setError(json.error);
      } else {
        setResult(json);
      }
    } catch {
      setError("Prediction failed. Please try again.");
    }

    setLoading(false);
  };

  const riskColor = (level) => {
    switch (level) {
      case "Low Risk":
        return "bg-green-100 text-green-800 border-green-200";
      case "Moderate Risk":
        return "bg-yellow-100 text-yellow-800 border-yellow-200";
      case "High Risk":
        return "bg-orange-100 text-orange-800 border-orange-200";
      case "Very High Risk":
        return "bg-red-100 text-red-800 border-red-200";
      default:
        return "bg-gray-100 text-gray-800 border-gray-200";
    }
  };

  return (
    <div className="min-h-screen bg-linear-to-br from-slate-50 to-blue-50 py-12 px-4 sm:px-6 lg:px-8 font-sans">
      <div className="max-w-4xl mx-auto">
        <div className="bg-white rounded-3xl shadow-xl overflow-hidden border border-slate-100">
          {/* Header Section */}
          <div className="bg-slate-900 text-white p-8 sm:p-10 text-center">
            <h1 className="text-3xl sm:text-4xl font-display font-bold mb-3 tracking-tight">
              Women's Safety Predictor
            </h1>
            <p className="text-slate-300 text-lg max-w-2xl mx-auto font-light">
              Forecasting district-level crime risks using advanced Time-Series
              Machine Learning
            </p>
          </div>

          <div className="p-8 sm:p-10 space-y-8">
            {/* Context Panel */}
            <div className="bg-blue-50 border border-blue-100 rounded-2xl p-6 text-slate-700 leading-relaxed shadow-sm">
              <div className="flex gap-3 mb-3">
                <span className="text-2xl">📊</span>
                <p className="font-medium text-blue-900 text-lg">Model Scope</p>
              </div>
              <p className="mb-2">
                Predicts the{" "}
                <span className="font-semibold text-blue-900">
                  percentage change/trend in crimes against women
                </span>{" "}
                for the year{" "}
                <span className="font-semibold text-blue-900">2014-2015</span>.
              </p>
              <div className="grid md:grid-cols-2 gap-4 mt-4 text-sm">
                <div className="bg-white/60 p-3 rounded-lg">
                  <span className="font-semibold text-slate-900 block mb-1">
                    Architecture
                  </span>
                  XGBoost regression pipeline trained on official NCRB data
                  (2001–2014).
                </div>
                <div className="bg-white/60 p-3 rounded-lg">
                  <span className="font-semibold text-slate-900 block mb-1">
                    Risk Scoring
                  </span>
                  Relative ranking using national quartile thresholds for
                  geospatial consistency.
                </div>
              </div>
            </div>

            <RiskExplanation />

            {/* How it Works Accordion */}
            <div className="border border-slate-200 rounded-2xl overflow-hidden">
              <button
                onClick={() => setShowInfo(!showInfo)}
                className="w-full flex items-center justify-between p-4 bg-slate-50 hover:bg-slate-100 transition-colors font-semibold text-slate-700"
              >
                <span>How This Prediction Works</span>
                <span
                  className={`transform transition-transform duration-200 ${showInfo ? "rotate-180" : ""}`}
                >
                  ▼
                </span>
              </button>

              {showInfo && (
                <div className="p-6 bg-white space-y-4 text-sm text-slate-600 border-t border-slate-200">
                  <div className="grid gap-4 sm:grid-cols-2">
                    <div>
                      <strong className="block text-slate-900 mb-1">
                        Data Source
                      </strong>
                      Official NCRB crime records (2001-2014).
                    </div>
                    <div>
                      <strong className="block text-slate-900 mb-1">
                        Feature Engineering
                      </strong>
                      Sliding-window time-series to capture momentum &
                      volatility.
                    </div>
                    <div>
                      <strong className="block text-slate-900 mb-1">
                        Model
                      </strong>
                      XGBoost regression for percentage change forecasting.
                    </div>
                    <div>
                      <strong className="block text-slate-900 mb-1">
                        Validation Strategy
                      </strong>
                      District holdout testing to ensure generalization.
                    </div>
                  </div>
                </div>
              )}
            </div>

            <PredictionMeaning />

            {/* Selection Form */}
            <div className="bg-slate-50 p-6 rounded-2xl border border-slate-200 shadow-inner">
              <h3 className="text-lg font-semibold text-slate-800 mb-4">
                Select Location
              </h3>
              <div className="grid sm:grid-cols-2 gap-4 mb-6">
                <div>
                  <label className="block text-sm font-medium text-slate-500 mb-1.5 ml-1">
                    State
                  </label>
                  <div className="relative">
                    <select
                      className="w-full appearance-none bg-white border border-slate-300 text-slate-700 py-3 px-4 pr-8 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent shadow-sm"
                      onChange={(e) => {
                        setState(e.target.value);
                        setDistrict("");
                      }}
                      value={state || "Select State"}
                    >
                      <option disabled>Select State</option>
                      {Object.keys(data).map((s) => (
                        <option key={s} value={s}>
                          {s.toUpperCase()}
                        </option>
                      ))}
                    </select>
                    <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-slate-500">
                      <svg
                        className="fill-current h-4 w-4"
                        xmlns="http://www.w3.org/2000/svg"
                        viewBox="0 0 20 20"
                      >
                        <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z" />
                      </svg>
                    </div>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-500 mb-1.5 ml-1">
                    District
                  </label>
                  <div className="relative">
                    <select
                      className="w-full appearance-none bg-white border border-slate-300 text-slate-700 py-3 px-4 pr-8 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent shadow-sm disabled:bg-slate-100 disabled:text-slate-400"
                      onChange={(e) => setDistrict(e.target.value)}
                      disabled={!state}
                      value={district || "Select District"}
                    >
                      <option disabled>Select District</option>
                      {data[state]?.map((d) => (
                        <option key={d} value={d}>
                          {d.toUpperCase()}
                        </option>
                      ))}
                    </select>
                    <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-slate-500">
                      <svg
                        className="fill-current h-4 w-4"
                        xmlns="http://www.w3.org/2000/svg"
                        viewBox="0 0 20 20"
                      >
                        <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z" />
                      </svg>
                    </div>
                  </div>
                </div>
              </div>

              <button
                onClick={predict}
                disabled={!state || !district || loading}
                className={`w-full py-4 rounded-xl font-bold text-lg shadow-lg transform transition-all duration-200 
                  ${
                    !state || !district || loading
                      ? "bg-slate-300 text-slate-500 cursor-not-allowed"
                      : "bg-linear-to-r from-blue-600 to-indigo-600 text-white hover:from-blue-700 hover:to-indigo-700 hover:shadow-xl hover:-translate-y-0.5"
                  }`}
              >
                {loading ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg
                      className="animate-spin h-5 w-5 text-white"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      ></circle>
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      ></path>
                    </svg>
                    Analyzing Data...
                  </span>
                ) : (
                  "Predict Crime Risk"
                )}
              </button>

              {error && (
                <div className="mt-4 p-4 bg-red-50 border border-red-200 text-red-600 rounded-xl flex items-center gap-3">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-6 w-6"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                  {error}
                </div>
              )}
            </div>

            {/* Results Section */}
            {result && !loading && (
              <div className="mt-10 animate-fade-in-up">
                <div className="text-center p-8 bg-slate-50 border border-slate-200 rounded-3xl">
                  <p className="text-slate-500 font-medium uppercase tracking-wider text-sm mb-4">
                    Predicted Crime Trend (2014 → 2015)
                  </p>

                  <div className="flex flex-col items-center justify-center">
                    <span
                      className={`text-6xl sm:text-7xl font-display font-bold mb-4 ${result.predicted_change > 0 ? "text-red-500" : "text-green-500"}`}
                    >
                      {result.predicted_change > 0 ? "+" : ""}
                      {result.predicted_change.toFixed(2)}%
                    </span>

                    <div
                      className={`px-6 py-2 rounded-full font-bold text-sm tracking-wide border uppercase flex items-center gap-2 ${riskColor(result.risk_level)}`}
                    >
                      <span className="w-2 h-2 rounded-full bg-current"></span>
                      {result.risk_level}
                    </div>
                  </div>

                  <p className="mt-6 text-slate-500 text-sm max-w-lg mx-auto">
                    This projection is based on historical patterns. Use this as
                    an early warning signal rather than a definitive outcome.
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="bg-slate-50 border-t border-slate-100 p-4 text-center">
            <p className="text-xs text-slate-400 font-medium">
              Model: XGBoost • Features: Sliding Window Time-Series •
              Evaluation: District Holdout Validation
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
