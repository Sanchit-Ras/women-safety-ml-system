import { pct } from "../lib/api";

export default function WatchExplanation({ meta }) {
  const e = meta.evidence;
  return (
    <div className="grid md:grid-cols-2 gap-4">
      <div className="bg-blue-50 border border-blue-100 rounded-2xl p-6 text-slate-700 leading-relaxed">
        <p className="font-semibold text-blue-900 text-lg mb-2">What the watch list is</p>
        <p className="mb-3">
          The districts most likely to be among the <strong>fastest-growing tenth</strong> in {meta.watch_year} for reported crimes
          against women, ranked by a model trained on NCRB data from {meta.data_years[0]} to {meta.data_years[1]}.
        </p>
        <p className="text-sm bg-white/70 rounded-xl p-3">
          Tested on {e.test_years[0]}-{e.test_years.at(-1)}, each year forecast only from earlier years: of the districts it
          flagged, <strong>{pct(e.precision_at_10)}</strong> really were in the fastest-growing tenth (95% interval{" "}
          {pct(e.ci95[0])}-{pct(e.ci95[1])}). Picking at random: {pct(e.chance)}.
        </p>
      </div>
      <div className="bg-amber-50 border border-amber-100 rounded-2xl p-6 text-slate-700 leading-relaxed">
        <p className="font-semibold text-amber-900 text-lg mb-2">What it is not</p>
        <ul className="list-disc pl-5 space-y-1.5 text-sm">
          {meta.limitations.map((l) => <li key={l}>{l}</li>)}
          <li>{meta.size_of_change.statement}</li>
        </ul>
      </div>
    </div>
  );
}
