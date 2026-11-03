import { useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import DistrictCard from "../components/DistrictCard";
import DistrictMap from "../components/DistrictMap";
import DistrictSearch from "../components/DistrictSearch";
import Sparkline from "../components/Sparkline";
import { NO_COMPARISON, bandColors, bandLegend } from "../lib/bands";
import { changeText, num } from "../lib/format";
import { districtList, useMap, useSnapshot, useTitle } from "../lib/useData";

const EXAMPLES = [162, 490, 315]; // Lucknow, Pune, Kolkata
const BIG_STATE = 1000; // states with fewer cases in the first year swing on a handful of cases

// What each view of the map compares, in plain words
const VIEWS = {
  long: {
    title: (w) => `Change in reported cases over ${w.to_year - w.from_year + 1} years, ${w.from_year} to ${w.to_year}`,
    note: (w) =>
      `Each district against its own ${w.from_year} numbers, or, where its borders changed, the area it was part of. India rose ` +
      `${Math.round(w.india.change * 100)}% in these years, partly because the 2013 law widened what counts and more women ` +
      "reported, so compare a district with India, not only with zero.",
  },
  recent: {
    title: (w) => `Recent change in reported cases, ${w.from_year} to ${w.to_year}`,
    note: (w) =>
      `From ${w.from_year} almost every district was counted with today's borders, so this compares real numbers, not ` +
      "estimates. A rise can mean more crime, or more women reporting it.",
  },
};

function Overview({ snap, view, onSelect }) {
  const m = snap.model_info;
  const series = m.india_series;
  const [long, recent, w] = [m.trends.long, m.trends.recent, m.trends[view]];
  const names = Object.fromEntries(snap.districts.states.map((s) => [String(s.code), s.name]));
  const big = Object.entries(w.states)
    .filter(([, v]) => v.from_total >= BIG_STATE && v.change != null)
    .map(([c, v]) => ({ name: names[c] ?? c, ...v }))
    .sort((a, b) => b.change - a.change);
  const states = (rows) => rows.map((s) => `${s.name} ${changeText(s.change)}`).join(", ");
  const list = districtList(snap);
  const examples = EXAMPLES.map((c) => list.find((d) => d.code === c)).filter(Boolean);

  return (
    <div className="bg-white border border-slate-200 rounded-2xl p-5 sm:p-6 space-y-5 shadow-sm">
      <div>
        <p className="text-sm font-semibold text-slate-500 uppercase tracking-wide">India, every year {series[0][0]} to {series.at(-1)[0]}</p>
        <p className="text-slate-700 mt-1 mb-3">
          <span className="text-3xl font-display font-bold text-slate-900">{changeText(long.india.change)}</span> over{" "}
          {long.to_year - long.from_year + 1} years ({num(long.india.from_total)} cases in {long.from_year}, {num(long.india.to_total)} in{" "}
          {long.to_year})
        </p>
        <Sparkline series={series} compared={[recent.from_year, recent.to_year]} />
      </div>
      <p className="text-slate-700 border-t border-slate-100 pt-4">
        Recent change, {recent.from_year} to {recent.to_year}: <strong>{changeText(recent.india.change)}</strong> ({num(recent.india.from_total)} to{" "}
        {num(recent.india.to_total)}).
      </p>
      {big.length >= 6 && (
        <p className="text-sm text-slate-600">
          Among states with at least {num(BIG_STATE)} cases in {w.from_year}, from {w.from_year} to {w.to_year} the biggest rises:{" "}
          {states(big.slice(0, 3))}. The smallest: {states(big.slice(-3).reverse())}.
        </p>
      )}
      <div>
        <p className="text-sm text-slate-500 mb-2">Try a district:</p>
        <div className="flex flex-wrap gap-2">
          {examples.map((d) => (
            <button key={d.code} onClick={() => onSelect(d.code)}
                    className="border border-slate-200 rounded-full px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-50">
              {d.name}, {d.state}
            </button>
          ))}
        </div>
      </div>
      <div className="flex flex-wrap gap-2 border-t border-slate-100 pt-4 text-sm">
        <a href={m.dataset.path} download className="bg-slate-900 text-white rounded-lg px-3 py-2 hover:bg-slate-700">
          Download all {long.to_year - long.from_year + 1} years (CSV)
        </a>
        <Link to="/data" className="border border-slate-200 rounded-lg px-3 py-2 text-slate-700 hover:bg-slate-50">About the data and the API</Link>
      </div>
    </div>
  );
}

export default function Explore() {
  const { data: snap, error, retry } = useSnapshot();
  const geo = useMap();
  const [params, setParams] = useSearchParams();
  const [view, setView] = useState("long");
  const code = Number(params.get("district")) || null;
  const cardRef = useRef(null);
  const list = useMemo(() => districtList(snap), [snap]);
  const byCode = useMemo(() => Object.fromEntries(list.map((d) => [d.code, d])), [list]);
  const district = code ? snap?.district[String(code)] : null;
  const w = snap?.model_info.trends[view];
  const [y0, y1] = snap?.model_info.data_years ?? [2001, 2022];
  const span = y1 - y0 + 1;
  useTitle(district ? `${district.name}, ${district.state}: crimes against women, ${y0}–${y1}` : null);

  // Colours and legend for the view on show, with how many districts fall in each band
  const colors = useMemo(() => (w ? bandColors(w.bands) : {}), [w]);
  const legend = useMemo(() => {
    if (!w) return [];
    const counts = { none: 0 };
    list.forEach((d) => {
      const b = d[`${view}_band`];
      if (colors[b]) counts[b] = (counts[b] ?? 0) + 1;
      else counts.none += 1;
    });
    return bandLegend(w.bands, counts);
  }, [w, list, view, colors]);

  const select = (c) => {
    setParams(c ? { district: String(c) } : {});
    if (c && window.innerWidth < 1024) {
      setTimeout(() => cardRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 60);
    }
  };

  const describe = (c) => {
    const t = snap?.district[String(c)]?.summary[view];
    if (!t || t.from_year !== w.from_year) return "no comparable numbers for these years";
    if (t.band === w.few_cases) return "too few cases to compare";
    return `${changeText(t.change)} since ${t.from_year}`;
  };

  const tiles = [
    [`${span} years`, `every year, ${y0} to ${y1}`],
    [snap ? num(list.length) : "...", "districts, on today's map"],
    [snap ? num(snap.model_info.dataset.rows) : "...", "rows, free to download"],
  ];

  return (
    <div className="min-h-screen bg-slate-50 pt-20 pb-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <header className="pt-4 pb-6 sm:pt-8 sm:pb-8">
          <p className="text-sm font-semibold text-blue-700 uppercase tracking-wide mb-2">Crimes against women in India, district by district</p>
          <h1 className="text-3xl sm:text-5xl font-display font-bold text-slate-900 tracking-tight max-w-4xl">
            How has your district changed in <span className="text-blue-700 whitespace-nowrap">{span} years</span>?
          </h1>
          <p className="mt-4 text-base sm:text-lg text-slate-600 leading-relaxed max-w-3xl">
            Reported crimes against women for every district of India, every year from {y0} to {y1}, on today's map. Districts keep
            splitting and being renamed, so the official yearly tables cannot be lined up year by year as published; this project
            rebuilt all {span} years onto today's districts, so they can.
          </p>
          <div className="mt-5 grid grid-cols-3 gap-3 max-w-2xl">
            {tiles.map(([value, label]) => (
              <div key={label} className="bg-white border border-slate-200 rounded-xl px-3 py-3 sm:px-4">
                <p className="text-xl sm:text-3xl font-display font-bold text-slate-900">{value}</p>
                <p className="text-xs sm:text-sm text-slate-500 leading-snug">{label}</p>
              </div>
            ))}
          </div>
          <div className="mt-5 max-w-xl">
            <DistrictSearch items={list} onSelect={select} />
          </div>
        </header>

        {error ? (
          <div className="p-4 bg-red-50 text-red-700 rounded-xl border border-red-100">
            Could not load the data. <button onClick={retry} className="underline font-semibold">Try again</button>
          </div>
        ) : !snap ? (
          <div className="text-slate-400 py-20 text-center">Loading the data...</div>
        ) : (
          <div className="grid lg:grid-cols-12 gap-6 items-start">
            <section className="lg:col-span-7 bg-white border border-slate-200 rounded-2xl p-4 sm:p-6 shadow-sm">
              <div className="inline-flex rounded-lg border border-slate-200 bg-slate-50 p-0.5 text-sm mb-4" role="tablist" aria-label="What the map compares">
                {Object.keys(VIEWS).map((k) => {
                  const kw = snap.model_info.trends[k];
                  return (
                    <button key={k} role="tab" aria-selected={view === k} onClick={() => setView(k)}
                            className={`px-3 py-1.5 rounded-md ${view === k ? "bg-white shadow-sm text-slate-900 font-semibold" : "text-slate-600 hover:text-slate-900"}`}>
                      {k === "long" ? `${kw.to_year - kw.from_year + 1} years, ${kw.from_year}–${kw.to_year}` : `Recent, ${kw.from_year}–${kw.to_year}`}
                    </button>
                  );
                })}
              </div>
              <h2 className="text-lg font-display font-semibold text-slate-900">{VIEWS[view].title(w)}</h2>
              <p className="text-sm text-slate-500 mb-3">{VIEWS[view].note(w)} Click a district.</p>
              <DistrictMap
                geo={geo}
                colorOf={(c) => colors[byCode[c]?.[`${view}_band`]] ?? NO_COMPARISON}
                legend={legend}
                selected={code}
                onSelect={select}
                describe={describe}
                label={`Map of India's districts coloured by the change in reported crimes against women, ${w.from_year} to ${w.to_year}`}
              />
            </section>
            <aside ref={cardRef} className={`lg:col-span-5 scroll-mt-20 space-y-3 ${district ? "order-first lg:order-none" : ""}`}>
              {district ? (
                <>
                  <button onClick={() => select(null)} className="text-sm text-blue-700 hover:underline">&larr; India overview</button>
                  <DistrictCard key={district.code} district={district} meta={snap.model_info} />
                </>
              ) : code ? (
                <div className="p-4 bg-amber-50 text-amber-800 rounded-xl border border-amber-100">No district has the code {code}.</div>
              ) : (
                <Overview snap={snap} view={view} onSelect={select} />
              )}
            </aside>
          </div>
        )}
      </div>
    </div>
  );
}
