// Every number here comes from a committed report: evals/reports/2026-10-18.md and ml/reports/2026-10-18_bakeoff.md.
function Row({ label, value }) {
  return (
    <li className="flex justify-between gap-4">
      <span>{label}</span>
      <span className="font-mono font-bold text-slate-900 text-right">{value}</span>
    </li>
  );
}

function Bar({ label, value, color }) {
  return (
    <div>
      <div className="flex justify-between text-sm font-semibold mb-2">
        <span className="text-slate-500">{label}</span>
        <span className="text-slate-900">{Math.round(value * 100)}%</span>
      </div>
      <div className="w-full bg-slate-200 rounded-full h-2">
        <div className={`${color} h-2 rounded-full`} style={{ width: `${value * 100}%` }}></div>
      </div>
    </div>
  );
}

export default function Performance() {
  return (
    <div className="bg-white/50 py-24 border-t border-slate-100">
      <div className="max-w-7xl mx-auto px-6">
        <h2 className="text-4xl font-display font-bold text-center text-slate-900 mb-4">
          Measured Results
        </h2>
        <p className="text-center text-slate-600 max-w-2xl mx-auto mb-16">
          Every number is measured on held-out data and kept in a dated report in the repository.
        </p>

        <div className="grid md:grid-cols-2 gap-8 lg:gap-12 max-w-5xl mx-auto">
          <div className="bg-slate-50 p-8 rounded-3xl border border-slate-200">
            <h3 className="text-2xl font-bold text-slate-800 mb-6 flex items-center gap-3">
              <span className="w-3 h-3 rounded-full bg-green-500"></span>
              Data Pipeline
            </h3>
            <div className="space-y-6">
              <Bar label="Old place names matched (from 56.6%)" value={0.985} color="bg-green-500" />
              <ul className="space-y-3 pt-4 border-t border-slate-200 text-slate-600 text-sm">
                <Row label="Records placed on a district" value="99.6% of 17,140" />
                <Row label="Records re-read from the source files" value="100 of 100 match" />
                <Row label="Model answers outside the allowed list" value="0 of 76" />
                <Row label="Names a person must check (with the cut)" value="124 → 42" />
              </ul>
            </div>
          </div>

          <div className="bg-white p-8 rounded-3xl border border-blue-100 shadow-xl relative overflow-hidden">
            <div className="absolute top-0 right-0 bg-blue-600 text-white text-xs font-bold px-3 py-1 rounded-bl-xl">
              WATCH LIST MODEL
            </div>
            <h3 className="text-2xl font-bold text-blue-900 mb-6 flex items-center gap-3">
              <span className="w-3 h-3 rounded-full bg-blue-500 animate-pulse"></span>
              Forecasting
            </h3>
            <div className="space-y-6">
              <Bar label="Flagged districts truly in the fastest-growing tenth" value={0.367} color="bg-blue-600" />
              <Bar label="The same, picking at random" value={0.1} color="bg-slate-400" />
              <ul className="space-y-3 pt-4 border-t border-slate-100 text-slate-600 text-sm">
                <Row label="95% interval" value="31–42%" />
                <Row label="Test" value="2019–2022, past only" />
                <Row label="Size of next year's change" value="not forecastable" />
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
