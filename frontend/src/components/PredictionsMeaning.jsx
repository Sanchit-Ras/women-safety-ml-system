export default function PredictionMeaning() {
  return (
    <div className="bg-purple-50 border border-purple-100 rounded-2xl p-6 mb-8 shadow-sm">
      <div className="flex items-center gap-3 mb-4">
        <div className="bg-purple-100 p-2 rounded-lg">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-6 w-6 text-purple-600"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
        </div>
        <h2 className="text-lg font-bold text-slate-800">
          What This Prediction Means
        </h2>
      </div>

      <p className="text-slate-700 mb-4 leading-relaxed">
        This system predicts the{" "}
        <strong className="font-semibold text-purple-900 border-b-2 border-purple-200">
          expected percentage change in crimes against women
        </strong>{" "}
        for a district in the upcoming year using historical trends from
        2001–2014.
      </p>

      <ul className="space-y-3 mb-6">
        <li className="flex items-start gap-3 bg-white p-3 rounded-lg border border-purple-100 shadow-sm">
          <span className="text-red-500 text-lg mt-0.5">•</span>
          <span className="text-slate-700 text-sm">
            <span className="font-semibold text-red-600">Positive values</span>{" "}
            indicate a potential increase in crime.
          </span>
        </li>
        <li className="flex items-start gap-3 bg-white p-3 rounded-lg border border-purple-100 shadow-sm">
          <span className="text-green-500 text-lg mt-0.5">•</span>
          <span className="text-slate-700 text-sm">
            <span className="font-semibold text-green-600">
              Negative values
            </span>{" "}
            suggest a possible decline.
          </span>
        </li>
      </ul>

      <p className="mt-3 text-slate-700">
        Risk levels are determined relative to nationwide patterns rather than
        absolute values. Even if crime is declining in a district, it may still
        be categorized as higher risk if the rate of improvement is
        significantly slower than the national trend.
      </p>

      <div className="bg-white border rounded-lg p-4 mt-4">
        <p className="text-sm text-slate-700 font-semibold">
          How Relative Risk Works:
        </p>

        <p className="text-sm text-slate-600 mt-1">
          If most districts experience a 5–8% decline in crime, but one district
          improves by only 1%, that district may still require attention — not
          because crime increased, but because progress is lagging behind
          national improvements.
        </p>
      </div>

      <div className="mt-4 flex items-start gap-2 p-3 bg-yellow-50 border border-yellow-100 rounded-lg text-xs text-yellow-800">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="h-4 w-4 mt-0.5 shrink-0"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
          />
        </svg>
        <p>
          Predictions are trend-based estimates and should be used as an early
          warning signal, not a guarantee.
        </p>
      </div>
    </div>
  );
}
