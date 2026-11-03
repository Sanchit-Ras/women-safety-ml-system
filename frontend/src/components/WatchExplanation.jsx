import { pct } from "../lib/format";

export default function WatchExplanation({ meta }) {
  const e = meta.evidence;
  const first = e.by_group[0];
  const last = e.by_group.at(-1);
  return (
    <div className="grid md:grid-cols-2 gap-4">
      <div className="bg-blue-50 border border-blue-100 rounded-2xl p-6 text-slate-700 leading-relaxed">
        <p className="font-semibold text-blue-900 text-lg mb-2">What the watch list is</p>
        <p className="mb-3">
          Districts are split into {e.size_groups} groups by how many crimes against women were reported in {meta.data_years[1]}.
          Within each group, the watch list holds the tenth most likely to grow fastest in {meta.watch_year}, so every district is
          compared only with districts of similar size.
        </p>
        <p className="text-sm bg-white/70 rounded-xl p-3">
          Tested on {e.test_years[0]}-{e.test_years.at(-1)}, each year forecast only from earlier years: of the districts it
          flagged, <strong>{pct(e.precision_at_10)}</strong> really were among the fastest-growing tenth of their group (95% interval{" "}
          {pct(e.ci95[0])}-{pct(e.ci95[1])}). By guessing: {pct(e.chance)}. By size alone: {pct(e.size_only)}. It works best for smaller
          districts ({pct(first.precision)} for those with up to {first.max_cases.toLocaleString("en-IN")} cases) and least for the
          largest ({pct(last.precision)} for those with {last.min_cases.toLocaleString("en-IN")} or more).
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
