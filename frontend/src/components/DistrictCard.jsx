import { useState } from "react";
import { Link } from "react-router-dom";
import Sparkline from "./Sparkline";
import { bandColor } from "../lib/bands";
import { changeText, downloadCsv, listText, num, ordinal, pct } from "../lib/format";

const HOW = ["filed", "adjusted", "estimated"];

// The trend in plain words: what changed, over which years, and the same years for the state and India.
function Trend({ d, t, firstYear }) {
  if (!t) {
    return (
      <p className="text-slate-600">
        No fair comparison over time: since {firstYear}, {d.name} has no two years counted with today's borders (its
        borders changed recently, or a year is missing). The chart and the notes below show what is known.
      </p>
    );
  }
  const others = d.unit_members.map((m) => m.name);
  const area = t.scope === "area";
  const later = t.from_year > firstYear;
  return (
    <div className="space-y-2">
      {area && (
        <p className="text-sm text-slate-600">
          Its borders changed after {t.from_year}, so it is compared together with {listText(others)}: their combined
          numbers are real for every year.
        </p>
      )}
      {t.band === "few_cases" ? (
        <p className="text-slate-700">
          {num(t.from_total)} cases in {t.from_year} and {num(t.to_total)} in {t.to_year}: too few for a fair percentage.
        </p>
      ) : (
        <p className="text-lg text-slate-800">
          <span className="inline-block w-3 h-3 rounded-sm mr-2 align-middle" style={{ background: bandColor(t.band) }} />
          <strong className="text-2xl font-display">{changeText(t.change)}</strong> since {t.from_year}
          {later && <span className="text-slate-500">, the first year it was counted with today's borders</span>}
          <span className="text-slate-500"> ({num(t.from_total)} to {num(t.to_total)}{area ? ", together" : ""})</span>
        </p>
      )}
      <p className="text-sm text-slate-600">
        Over the same years: {d.state} {changeText(t.state_change)}, India {changeText(t.india_change)}.
      </p>
    </div>
  );
}

function WatchStatus({ d, meta, open }) {
  const p = d.peer;
  const hit = p && meta.evidence.by_group.find((b) => b.group === p.group)?.precision;
  const status = d.status === "watch" ? "on the list" : d.status === "no_signal" ? "not on the list" : "not scored";
  return (
    <details open={open} className="group border-t border-slate-100 pt-4">
      <summary className="cursor-pointer text-sm text-slate-700 select-none">
        <span className="font-semibold">Next year's watch list ({meta.watch_year}, experimental):</span>{" "}
        <span className={d.status === "watch" ? "text-red-700 font-semibold" : ""}>{status}</span>
      </summary>
      <div className="text-sm text-slate-600 mt-2 space-y-2">
        {p ? (
          <p>
            Compared with the {p.size} districts of similar size ({p.min_cases.toLocaleString("en-IN")} to{" "}
            {p.max_cases.toLocaleString("en-IN")} cases in {meta.data_years[1]}
            {d.unit_members.length > 0 && <>; assessed with {listText(d.unit_members.map((m) => m.name))}, {num(p.cases)} cases together</>}
            ), it ranks <strong>{p.rank}</strong> for how likely it is to be among the fastest-growing in {meta.watch_year}; the top{" "}
            {Math.max(1, Math.ceil(p.size * meta.watch_share))} are on the list.
            {hit != null && <> In past tests, about {Math.round(hit * 100)} in 100 flags for districts this size were right (by guessing: 10).</>}
          </p>
        ) : (
          <p>Not scored: it lacks six recent years of comparable numbers.</p>
        )}
        <Link to="/watch-list" className="text-blue-700 hover:underline">What the watch list is, and how well it works</Link>
      </div>
    </details>
  );
}

export default function DistrictCard({ district: d, meta, watchOpen = false }) {
  const [copied, setCopied] = useState(false);
  const s = d.summary;
  const cats = meta.categories;
  const max = d.latest ? Math.max(1, ...Object.values(d.latest)) : 1;
  const t = s.trend;

  const copyLink = () => {
    navigator.clipboard?.writeText(window.location.href).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };
  const download = () =>
    downloadCsv(`${d.name.toLowerCase().replace(/[^a-z0-9]+/g, "_")}_crimes_against_women.csv`,
      ["state", "district", "district_code", "year", "total", "how_counted"],
      d.series.map(([year, total, kind]) => [d.state, d.name, d.code, year, total, HOW[kind]]));

  return (
    <article className="bg-white border border-slate-200 rounded-2xl p-5 sm:p-6 space-y-5 shadow-sm">
      <header className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-2xl font-display font-bold text-slate-900 leading-tight">{d.name}</h2>
          <p className="text-slate-500">{d.state}</p>
        </div>
        <button onClick={copyLink} className="shrink-0 text-sm text-slate-600 border border-slate-200 rounded-lg px-3 py-1.5 hover:bg-slate-50">
          {copied ? "Link copied" : "Copy link"}
        </button>
      </header>

      {d.latest_total != null ? (
        <>
          <p className="text-slate-700">
            <span className="text-3xl font-display font-bold text-slate-900">{num(d.latest_total)}</span>{" "}
            crimes against women reported in {d.latest_year}
          </p>
          <Trend d={d} t={t} firstYear={meta.trend.from_year} />
          {(s.rank || s.top) && (
            <ul className="text-sm text-slate-600 space-y-1.5">
              {s.rank && (
                <li>
                  {ordinal(s.rank.rank)} of {s.rank.of} districts in {d.state} by cases reported in {s.rank.year} (counts, not
                  adjusted for population, so larger districts rank higher).
                </li>
              )}
              {s.top && (
                <li>
                  Most common: {cats[s.top.category].toLowerCase()}, {pct(s.top.share)} of its cases
                  {s.top.state_share != null && <> ({d.state}: {pct(s.top.state_share)})</>}.
                </li>
              )}
            </ul>
          )}
          <div>
            <p className="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-2">Every year, {d.series[0][0]} to {d.latest_year}</p>
            <Sparkline series={d.series} compared={t && t.band !== "few_cases" ? [t.from_year, t.to_year] : null} />
          </div>
          <div>
            <p className="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-2">By offence, {d.latest_year}</p>
            <ul className="space-y-2">
              {Object.entries(cats).map(([key, label]) => (
                <li key={key} className="text-sm">
                  <div className="flex justify-between text-slate-600"><span>{label}</span><span className="font-mono">{Math.round(d.latest[key])}</span></div>
                  <div className="h-1.5 bg-slate-100 rounded-full"><div className="h-1.5 bg-blue-500 rounded-full" style={{ width: `${(d.latest[key] / max) * 100}%` }} /></div>
                </li>
              ))}
            </ul>
          </div>
        </>
      ) : (
        <p className="text-slate-600">No numbers of its own in the NCRB files.</p>
      )}

      {d.notes.length > 0 && (
        <ul className="text-sm text-slate-600 space-y-1.5 border-t border-slate-100 pt-4 list-disc pl-5">
          {d.notes.map((n) => <li key={n}>{n}</li>)}
        </ul>
      )}

      <WatchStatus d={d} meta={meta} open={watchOpen} />

      <div className="flex flex-wrap gap-2 border-t border-slate-100 pt-4 text-sm">
        {d.series.length > 0 && (
          <button onClick={download} className="bg-slate-900 text-white rounded-lg px-3 py-2 hover:bg-slate-700">
            Download {d.name}'s numbers (CSV)
          </button>
        )}
        <Link to="/data" className="border border-slate-200 rounded-lg px-3 py-2 text-slate-700 hover:bg-slate-50">
          Every district, every offence
        </Link>
      </div>
    </article>
  );
}
