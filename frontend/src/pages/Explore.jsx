import { useMemo, useRef } from "react";
import { Link, useSearchParams } from "react-router-dom";
import DistrictCard from "../components/DistrictCard";
import DistrictMap from "../components/DistrictMap";
import DistrictSearch from "../components/DistrictSearch";
import { bandColor, bandLegend } from "../lib/bands";
import { changeText, num } from "../lib/format";
import { districtList, useMap, useSnapshot, useTitle } from "../lib/useData";

const EXAMPLES = [162, 490, 315]; // Lucknow (fell), Pune (rose sharply), Kolkata (about the same)
const BIG_STATE = 1000; // states with fewer cases in the first year swing on a handful of cases

function Overview({ snap, list, onSelect }) {
  const tr = snap.model_info.trend;
  const count = (band) => list.filter((d) => d.trend === band).length;
  const names = Object.fromEntries(snap.districts.states.map((s) => [String(s.code), s.name]));
  const big = Object.entries(tr.states)
    .filter(([, v]) => v.from_total >= BIG_STATE && v.change != null)
    .map(([c, v]) => ({ name: names[c] ?? c, ...v }));
  const rises = [...big].sort((a, b) => b.change - a.change).slice(0, 3);
  const falls = [...big].sort((a, b) => a.change - b.change).slice(0, 3);
  const examples = EXAMPLES.map((c) => list.find((d) => d.code === c)).filter(Boolean);
  const states = (rows) => rows.map((s) => `${s.name} ${changeText(s.change)}`).join(", ");

  return (
    <div className="bg-white border border-slate-200 rounded-2xl p-5 sm:p-6 space-y-5 shadow-sm">
      <div>
        <p className="text-sm font-semibold text-slate-500 uppercase tracking-wide">India, {tr.from_year} to {tr.to_year}</p>
        <p className="text-slate-700 mt-1">
          <span className="text-3xl font-display font-bold text-slate-900">{changeText(tr.india.change)}</span>{" "}
          ({num(tr.india.from_total)} to {num(tr.india.to_total)} reported cases)
        </p>
      </div>
      <p className="text-slate-700">
        Of today's districts, <strong>{count("fell")}</strong> fell by more than 10%, <strong>{count("steady")}</strong> stayed
        within 10%, <strong>{count("rose")}</strong> rose by 10% to 50% and <strong>{count("rose_sharply")}</strong> rose by half
        or more. The rest had too few cases, or no two years with today's borders, to compare.
      </p>
      <p className="text-sm text-slate-600">
        Among states with at least {num(BIG_STATE)} cases in {tr.from_year}, the biggest rises: {states(rises)}. The biggest
        falls: {states(falls)}.
      </p>
      <div>
        <p className="text-sm text-slate-500 mb-2">Try a district:</p>
        <div className="flex flex-wrap gap-2">
          {examples.map((d) => (
            <button key={d.code} onClick={() => onSelect(d.code)}
                    className="flex items-center gap-2 border border-slate-200 rounded-full px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-50">
              <span className="w-2.5 h-2.5 rounded-sm" style={{ background: bandColor(d.trend) }} />
              {d.name}, {d.state}
            </button>
          ))}
        </div>
      </div>
      <div className="flex flex-wrap gap-2 border-t border-slate-100 pt-4 text-sm">
        <a href={snap.model_info.dataset.path} download className="bg-slate-900 text-white rounded-lg px-3 py-2 hover:bg-slate-700">
          Download every district (CSV)
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
  const code = Number(params.get("district")) || null;
  const cardRef = useRef(null);
  const list = useMemo(() => districtList(snap), [snap]);
  const byCode = useMemo(() => Object.fromEntries(list.map((d) => [d.code, d])), [list]);
  const district = code ? snap?.district[String(code)] : null;
  const tr = snap?.model_info.trend;
  useTitle(district ? `${district.name}, ${district.state}: crimes against women, 2001–2022` : null);

  const select = (c) => {
    setParams(c ? { district: String(c) } : {});
    if (c && window.innerWidth < 1024) {
      setTimeout(() => cardRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 60);
    }
  };

  const describe = (c) => {
    const t = snap?.district[String(c)]?.summary.trend;
    if (!t) return "no two comparable years";
    if (t.band === "few_cases") return "too few cases to compare";
    return `${changeText(t.change)} since ${t.from_year}`;
  };

  return (
    <div className="min-h-screen bg-slate-50 pt-20 pb-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <header className="max-w-3xl pt-4 pb-6 sm:pt-8 sm:pb-8">
          <h1 className="text-3xl sm:text-4xl font-display font-bold text-slate-900 tracking-tight">How has your district changed?</h1>
          <p className="mt-3 text-base sm:text-lg text-slate-600 leading-relaxed">
            Crimes against women reported to the police in every district of India, 2001 to 2022, on today's map. Districts keep
            splitting and being renamed, so the official yearly tables cannot be lined up year by year as published; this project
            rebuilt all 22 years onto today's districts, so they can.
          </p>
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
              <h2 className="text-lg font-display font-semibold text-slate-900">
                Change in reported cases, {tr.from_year} to {tr.to_year}
              </h2>
              <p className="text-sm text-slate-500 mb-3">
                Each district is compared with itself. A rise can mean more crime, or more women reporting it. Click a district.
              </p>
              <DistrictMap
                geo={geo}
                colorOf={(c) => bandColor(byCode[c]?.trend)}
                legend={bandLegend(tr.bands)}
                selected={code}
                onSelect={select}
                describe={describe}
                label={`Map of India's districts coloured by the change in reported crimes against women, ${tr.from_year} to ${tr.to_year}`}
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
                <Overview snap={snap} list={list} onSelect={select} />
              )}
            </aside>
          </div>
        )}
      </div>
    </div>
  );
}
