export default function RiskExplanation() {
  return (
    <div className="bg-blue-50 border border-blue-100 rounded-2xl p-6 mb-6 shadow-sm">
      <div className="flex items-center gap-3 mb-4">
        <div className="bg-blue-100 p-2 rounded-lg">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-6 w-6 text-blue-600"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
            />
          </svg>
        </div>
        <h2 className="text-lg font-bold text-slate-800">
          How Risk is Calculated
        </h2>
      </div>

      <p className="text-slate-700 mb-4 leading-relaxed">
        Risk levels are determined using{" "}
        <strong className="font-semibold text-blue-900 border-b-2 border-blue-200">
          relative crime trends across India
        </strong>
        , not just absolute crime reduction.
      </p>

      <div className="bg-white p-4 rounded-xl border border-blue-100 text-sm text-slate-600 space-y-2">
        <p>
          Districts are grouped into statistical quartiles based on predicted
          crime change.
        </p>
        <p>
          A district may show a small improvement but still rank as
          <strong className="text-yellow-600"> Moderate Risk</strong> if its
          improvement is weaker than the most districts.
        </p>
      </div>

      <div className="mt-4 flex items-center justify-between text-xs font-semibold text-slate-500 bg-blue-100/50 p-3 rounded-lg">
        <span>Lower Relative Risk</span>
        <span className="text-slate-300">→</span>
        <span className="text-yellow-600">Moderate</span>
        <span className="text-slate-300">→</span>
        <span className="text-orange-600">Higher</span>
        <span className="text-slate-300">→</span>
        <span className="text-red-600">Critical</span>
      </div>
    </div>
  );
}
