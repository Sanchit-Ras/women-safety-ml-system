import Sparkline from "./Sparkline";

const BADGE = {
  watch: "bg-red-100 text-red-800 border-red-200",
  no_signal: "bg-slate-100 text-slate-700 border-slate-200",
  not_scored: "bg-slate-50 text-slate-500 border-slate-200",
};

function statusText(d, year) {
  if (d.status === "watch") return `On the ${year} watch list`;
  if (d.status === "no_signal") return `Not on the ${year} watch list`;
  return "Not scored";
}

export default function DistrictCard({ district: d, meta }) {
  const cats = meta.categories;
  const max = d.latest ? Math.max(1, ...Object.values(d.latest)) : 1;
  const p = d.peer;
  const groupHit = p && meta.evidence.by_group.find((b) => b.group === p.group)?.precision;

  return (
    <div className="bg-white border border-slate-200 rounded-3xl p-6 sm:p-8 space-y-6 shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h3 className="text-2xl font-display font-bold text-slate-900">{d.name}</h3>
          <p className="text-slate-500">{d.state}</p>
        </div>
        <span className={`px-4 py-1.5 rounded-full border text-sm font-semibold ${BADGE[d.status]}`}>
          {statusText(d, meta.watch_year)}
        </span>
      </div>

      {p && (
        <p className={`text-sm rounded-xl p-4 border ${d.status === "watch" ? "text-red-900 bg-red-50 border-red-100" : "text-slate-700 bg-slate-50 border-slate-200"}`}>
          Compared with the {p.size} districts of similar size (those with {p.min_cases.toLocaleString("en-IN")} to{" "}
          {p.max_cases.toLocaleString("en-IN")} cases in {meta.data_years[1]}
          {d.unit_members.length > 0 && (
            <>; it is assessed with {d.unit_members.map((m) => m.name).join(", ")}, {p.cases.toLocaleString("en-IN")} cases together</>
          )}), it ranks <strong>{p.rank}</strong> for how likely it is
          to grow fastest in {meta.watch_year}; the top {Math.max(1, Math.ceil(p.size * meta.watch_share))} are on the watch list.
          {groupHit != null && (
            <> In past tests, for districts this size, about {Math.round(groupHit * 100)} in 100 flagged districts really were among
            the fastest-growing tenth (by guessing: 10).</>
          )}{" "}
          This compares growth, not how many crimes a district has.
        </p>
      )}

      {d.latest ? (
        <div className="grid lg:grid-cols-2 gap-8">
          <div>
            <p className="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-1">Reported in {d.latest_year}</p>
            <p className="text-4xl font-display font-bold text-slate-900 mb-4">{Math.round(d.latest_total).toLocaleString("en-IN")}</p>
            <ul className="space-y-2">
              {Object.entries(cats).map(([key, label]) => (
                <li key={key} className="text-sm">
                  <div className="flex justify-between text-slate-600"><span>{label}</span><span className="font-mono">{Math.round(d.latest[key])}</span></div>
                  <div className="h-1.5 bg-slate-100 rounded-full"><div className="h-1.5 bg-blue-500 rounded-full" style={{ width: `${(d.latest[key] / max) * 100}%` }} /></div>
                </li>
              ))}
            </ul>
          </div>
          <div>
            <p className="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-3">Total, {d.series[0][0]} to {d.latest_year}</p>
            <Sparkline series={d.series} />
          </div>
        </div>
      ) : (
        <p className="text-slate-600">No numbers of its own in the NCRB files.</p>
      )}

      {d.notes.length > 0 && (
        <ul className="text-sm text-slate-600 space-y-1.5 border-t border-slate-100 pt-4 list-disc pl-5">
          {d.notes.map((n) => <li key={n}>{n}</li>)}
        </ul>
      )}
    </div>
  );
}
