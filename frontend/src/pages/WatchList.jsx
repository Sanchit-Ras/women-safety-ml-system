import { useMemo, useRef } from "react";
import { useSearchParams } from "react-router-dom";
import DistrictCard from "../components/DistrictCard";
import DistrictMap from "../components/DistrictMap";
import WatchExplanation from "../components/WatchExplanation";
import { REPO } from "../lib/api";
import { STATUS_COLOR } from "../lib/bands";
import { pct } from "../lib/format";
import { districtList, useMap, useSnapshot, useTitle } from "../lib/useData";

const LEGEND = [
  { color: STATUS_COLOR.watch, label: "On the watch list" },
  { color: STATUS_COLOR.no_signal, label: "Not on it" },
  { color: STATUS_COLOR.not_scored, label: "Not scored" },
];

export default function WatchList() {
  const { data: snap, error, retry } = useSnapshot();
  const geo = useMap();
  const [params, setParams] = useSearchParams();
  const code = Number(params.get("district")) || null;
  const cardRef = useRef(null);
  const list = useMemo(() => districtList(snap), [snap]);
  const statusOf = useMemo(() => Object.fromEntries(list.map((d) => [d.code, d.status])), [list]);
  const district = code ? snap?.district[String(code)] : null;
  const meta = snap?.model_info;
  useTitle(`Next year's watch list${meta ? ` (${meta.watch_year})` : ""}: an experiment`);

  // The watch list by size group, largest first, each with how often the list was right for that size in past tests
  const groups = useMemo(() => {
    if (!snap) return [];
    const byGroup = {};
    snap.watchlist.districts.forEach((d) => {
      const g = (byGroup[d.peer.group] ??= { group: d.peer.group, min: d.peer.min_cases, max: d.peer.max_cases, items: [] });
      g.items.push(d);
    });
    const hit = Object.fromEntries(snap.model_info.evidence.by_group.map((b) => [b.group, b.precision]));
    return Object.values(byGroup).sort((a, b) => b.group - a.group).map((g) => ({ ...g, hit: hit[g.group] }));
  }, [snap]);

  const select = (c) => {
    setParams(c ? { district: String(c) } : {});
    if (c) setTimeout(() => cardRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 60);
  };

  return (
    <div className="min-h-screen bg-slate-50 pt-20 pb-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 space-y-6">
        <header className="max-w-3xl pt-4 sm:pt-8">
          <h1 className="text-3xl sm:text-4xl font-display font-bold text-slate-900 tracking-tight">
            Next year's watch list{meta ? ` (${meta.watch_year})` : ""}
          </h1>
          <p className="mt-3 text-base sm:text-lg text-slate-600 leading-relaxed">
            An experiment: which districts are likely to be among the fastest-growing tenth of districts their size next year.
            {meta && (
              <> In tests on {meta.evidence.test_years[0]}–{meta.evidence.test_years.at(-1)}, {pct(meta.evidence.precision_at_10)} of its
              flags were right, against {pct(meta.evidence.chance)} by guessing and {pct(meta.evidence.size_only)} by size alone. It
              ranks growth, not how unsafe a place is.</>
            )}
          </p>
        </header>

        {error ? (
          <div className="p-4 bg-red-50 text-red-700 rounded-xl border border-red-100">
            Could not load the data. <button onClick={retry} className="underline font-semibold">Try again</button>
          </div>
        ) : !snap ? (
          <div className="text-slate-400 py-20 text-center">Loading the data...</div>
        ) : (
          <>
            <details className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
              <summary className="cursor-pointer font-semibold text-slate-800 select-none">How to read the watch list, and what it is not</summary>
              <div className="mt-4"><WatchExplanation meta={meta} /></div>
            </details>

            <div className="grid lg:grid-cols-12 gap-6 items-start">
              <section className="lg:col-span-7 bg-white border border-slate-200 rounded-2xl p-4 sm:p-6 shadow-sm">
                <DistrictMap
                  geo={geo}
                  colorOf={(c) => STATUS_COLOR[statusOf[c]] ?? STATUS_COLOR.not_scored}
                  legend={LEGEND}
                  selected={code}
                  onSelect={select}
                  label={`Map of India's districts with the ${meta.watch_year} watch list in red`}
                />
              </section>

              <section className="lg:col-span-5 bg-white rounded-2xl border border-slate-200 overflow-hidden shadow-sm">
                <p className="px-5 py-3 font-semibold text-slate-800 border-b border-slate-100">
                  On the list ({snap.watchlist.districts.length} districts)
                </p>
                <div className="max-h-[32rem] overflow-y-auto text-sm">
                  {groups.map((g) => (
                    <div key={g.group}>
                      <p className="px-5 py-2 bg-slate-50 text-xs text-slate-500 border-y border-slate-100">
                        Size group {g.group}: {g.min.toLocaleString("en-IN")}-{g.max.toLocaleString("en-IN")} cases
                        {g.hit != null && <> · right {Math.round(g.hit * 100)} in 100 in past tests</>}
                      </p>
                      <ul className="divide-y divide-slate-100">
                        {g.items.map((d) => (
                          <li key={d.code}>
                            <button
                              onClick={() => select(d.code)}
                              className={`w-full text-left px-5 py-2 hover:bg-red-50 flex justify-between gap-3 ${d.code === code ? "bg-red-50" : ""}`}
                            >
                              <span>{d.name}<span className="text-slate-400">, {d.state}</span></span>
                              <span className="text-slate-500 font-mono">{d.latest_total == null ? "-" : Math.round(d.latest_total)}</span>
                            </button>
                          </li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
                <p className="px-5 py-2 text-xs text-slate-400 border-t border-slate-100">
                  Grouped by size, largest first; most likely first within a group. Number: the district's own cases in{" "}
                  {meta.data_years[1]}. By guessing, 10 in 100 would be right.
                </p>
              </section>
            </div>

            <div ref={cardRef} className="scroll-mt-20">
              {district && <DistrictCard key={district.code} district={district} meta={meta} watchOpen />}
            </div>

            <p className="text-center text-xs text-slate-500">
              Model: {meta.model.name}, batch-scored once a year.{" "}
              <a className="text-blue-700 hover:underline" href={`${REPO}/blob/main/${meta.evidence.report}`} target="_blank" rel="noopener noreferrer">
                The evidence report
              </a>
            </p>
          </>
        )}
      </div>
    </div>
  );
}
