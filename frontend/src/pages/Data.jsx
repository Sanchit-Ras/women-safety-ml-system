import ApiStatus from "../components/ApiStatus";
import { API, REPO } from "../lib/api";
import { num } from "../lib/format";
import { useSnapshot, useTitle } from "../lib/useData";

const COLUMNS = {
  state_code: "The state's code in India's Local Government Directory (LGD)",
  state: "The state or union territory the district belongs to today",
  district_code: "The district's LGD code: one code per district on today's map",
  district: "The district's name",
  year: "The year NCRB counted the cases for",
  total: "The sum of the seven offence columns",
  how_counted: "filed: as NCRB published it. adjusted: the filed number minus the part that is today another district. estimated: a share of the district it was part of then, by its share of that district's crimes",
};

const ENDPOINTS = [
  ["/districts", "Every district by state, with its map colour (how its cases changed) and its watch-list status"],
  ["/district/671", "One district (here Shahdara, Delhi): every year's total and how it was counted, the latest year by offence, the plain summary and notes"],
  ["/watchlist", "The experimental watch list for the year after the data"],
  ["/model-info", "The comparison's settings, India's and each state's change, and the watch list's evidence"],
  ["/deploy-check", "Which version of the API and which data are live"],
];

function Section({ title, children }) {
  return (
    <section className="bg-white border border-slate-200 rounded-2xl p-5 sm:p-7 shadow-sm space-y-4">
      <h2 className="text-xl font-display font-semibold text-slate-900">{title}</h2>
      {children}
    </section>
  );
}

export default function Data() {
  const { data: snap } = useSnapshot();
  const m = snap?.model_info;
  const [first, last] = m ? m.data_years : [2001, 2022];
  const years = `${first}–${last}`;
  const example = snap?.district["671"]?.summary.trend;
  useTitle("The data: crimes against women in India, by district, 2001–2022");

  return (
    <div className="min-h-screen bg-slate-50 pt-20 pb-12">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 space-y-6">
        <header className="pt-4 sm:pt-8">
          <h1 className="text-3xl sm:text-4xl font-display font-bold text-slate-900 tracking-tight">The data, free to download</h1>
          <p className="mt-3 text-base sm:text-lg text-slate-600 leading-relaxed">
            Every district on today's map of India, every year from {first} to {last}: crimes against women reported to the police, in the
            seven offence categories that exist in every year's files, re-based so that a district means the same area in every
            year. NCRB publishes these numbers each year, but districts split and are renamed, so the published tables cannot be
            lined up year by year; this table can.
          </p>
        </header>

        <Section title="Download">
          {m ? (
            <>
              <div className="flex flex-wrap items-center gap-4">
                <a href={m.dataset.path} download className="bg-slate-900 text-white rounded-lg px-4 py-2.5 font-medium hover:bg-slate-700">
                  Download the CSV
                </a>
                <p className="text-sm text-slate-600">
                  {num(m.dataset.rows)} rows: one per district and year · {num(m.dataset.districts)} districts · {years} · about 1 MB
                </p>
              </div>
              <p className="text-sm text-slate-600">
                Opens in Excel, Google Sheets, pandas or R. Each district's page on <a href="/" className="text-blue-700 hover:underline">Explore</a> also
                has a download of its own numbers.
              </p>
            </>
          ) : (
            <p className="text-slate-400">Loading...</p>
          )}
        </Section>

        <Section title="Columns">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <tbody className="divide-y divide-slate-100">
                {(m?.dataset.columns ?? Object.keys(COLUMNS)).map((c) => (
                  <tr key={c}>
                    <td className="py-2 pr-4 font-mono text-slate-800 whitespace-nowrap align-top">{c}</td>
                    <td className="py-2 text-slate-600">{COLUMNS[c] ?? `Cases registered: ${m?.categories[c] ?? c}`}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>

        <Section title="How the years were made comparable">
          <ul className="list-disc pl-5 space-y-2 text-slate-700">
            <li>NCRB's files come in four formats with different offence lists; a reviewed crosswalk maps all of them onto the seven
              categories that exist in every year, so the total measures the same thing in 2001 and in 2022.</li>
            <li>Old place names ("Cuddapah", "ST.RAVIDASNAGAR", "GUNTAKAL RLY.") were matched to today's districts by rules,
              fuzzy matching and a language model with guards and human review: 98.5% of 989 old names placed, measured
              against an answer key.</li>
            <li>When a district split, the new district's earlier years are estimated from its share of its parent's crimes, and
              marked "estimated"; the parent's are marked "adjusted". Totals for a state never change.</li>
          </ul>
        </Section>

        <Section title="Read it with care">
          <ul className="list-disc pl-5 space-y-2 text-slate-700">
            <li>These are cases reported to the police. A rise can mean more crime, or more women able to report it.</li>
            <li>They are counts, not rates: a large district has more cases because it has more people.</li>
            <li>In 2013 the law widened what counts as rape and sexual harassment, so part of the rise after 2013 is a change in
              what is counted.</li>
            <li>Estimated years are estimates. For comparisons, prefer years marked "filed": from 2017 almost every district
              has them.</li>
            <li>2015 comes from NCRB's own workbook; it adds up to slightly less than NCRB's national figures, because the
              district tables leave out a few state-level units. Those units, and a few others before 2016 that belong to no
              district, are not in this table.</li>
          </ul>
        </Section>

        <Section title="API">
          <p className="text-slate-700">
            The same numbers, as JSON, for programs. Base address: <code className="font-mono text-sm bg-slate-100 rounded px-1.5 py-0.5 break-all">{API}</code>
          </p>
          <p className="text-sm text-slate-600 flex flex-wrap items-center gap-2">
            Status: <ApiStatus /> It runs on a free server that sleeps when unused, so the first request can take half a minute.
          </p>
          <ul className="divide-y divide-slate-100 text-sm">
            {ENDPOINTS.map(([path, what]) => (
              <li key={path} className="py-2.5 flex flex-col sm:flex-row sm:gap-4">
                <a href={`${API}${path}`} target="_blank" rel="noopener noreferrer" className="font-mono text-blue-700 hover:underline sm:w-44 shrink-0">
                  GET {path}
                </a>
                <span className="text-slate-600">{what}</span>
              </li>
            ))}
          </ul>
          {example && (
            <div>
              <p className="text-sm text-slate-600 mb-2">For example, part of the answer for Shahdara:</p>
              <pre className="bg-slate-900 text-slate-100 text-xs rounded-xl p-4 overflow-x-auto">{`curl ${API}/district/671

"summary": { "trend": ${JSON.stringify(example, null, 2).replace(/\n/g, "\n  ")} }`}</pre>
            </div>
          )}
          <p className="text-sm text-slate-600">
            Full, interactive documentation: <a href={`${API}/docs`} target="_blank" rel="noopener noreferrer" className="text-blue-700 hover:underline">{API}/docs</a>
          </p>
        </Section>

        <Section title="Source and citation">
          <p className="text-slate-700">
            Source: National Crime Records Bureau (NCRB), <i>Crime in India</i>, district tables for {years}, as published by NCRB
            and redistributed by the India Data Portal. Re-based onto today's districts by Sanchit Rastogi, Women's Safety ML (
            <a href={REPO} className="text-blue-700 hover:underline break-all" target="_blank" rel="noopener noreferrer">{REPO}</a>). Every
            step, from the raw files to this table, is in the repository and rebuilt on every change.
          </p>
        </Section>
      </div>
    </div>
  );
}
