import { Link } from "react-router-dom";
import { API, REPO } from "../lib/api";
import { LIMITS, RESULTS, STEPS, STORIES } from "../content/story";
import { useTitle } from "../lib/useData";

const file = (path) => `${REPO}/blob/main/${path}`;

export default function About() {
  useTitle("How it was built: an evaluated LLM data pipeline for India's district crime data");
  return (
    <div className="min-h-screen bg-slate-50 pt-20 pb-12">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 space-y-10">
        <header className="pt-4 sm:pt-8 max-w-3xl">
          <h1 className="text-3xl sm:text-4xl font-display font-bold text-slate-900 tracking-tight">How it was built</h1>
          <p className="mt-3 text-base sm:text-lg text-slate-600 leading-relaxed">
            The <Link to="/" className="text-blue-700 hover:underline">explorer</Link> rests on a data pipeline that does the hard
            part: putting 22 years of messy government files onto one map. A language model fixes the parts rules cannot, and
            it is measured before it is trusted. Everything below is checked automatically on every change.
          </p>
        </header>

        <section className="space-y-4">
          <h2 className="text-xl font-display font-semibold text-slate-900">The problem</h2>
          <p className="text-slate-700 leading-relaxed max-w-3xl">
            NCRB publishes district-level crimes against women every year, but the 2001–2014 file names police units the way each
            year's clerk spelled them ("Cuddapah", "ST.RAVIDASNAGAR", "GUNTAKAL RLY."), later files use other formats, and many
            districts have split since. A district cannot be compared across years until every row is on the same map, and simple
            name matching places only 56.6% of the old names.
          </p>
        </section>

        <section className="space-y-4">
          <h2 className="text-xl font-display font-semibold text-slate-900">Results</h2>
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {RESULTS.map((r) => (
              <a key={r.value} href={file(r.report)} target="_blank" rel="noopener noreferrer"
                 className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm hover:border-blue-300 block">
                <p className="text-3xl font-display font-bold text-slate-900">{r.value}</p>
                <p className="text-slate-700 mt-1">{r.what}</p>
                <p className="text-sm text-slate-500 mt-2">{r.vs}</p>
              </a>
            ))}
          </div>
        </section>

        <section className="space-y-4">
          <h2 className="text-xl font-display font-semibold text-slate-900">How it works</h2>
          <ol className="grid md:grid-cols-2 gap-4">
            {STEPS.map((s, i) => (
              <li key={s.title} className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
                <p className="font-semibold text-slate-900"><span className="text-blue-600 mr-2">{i + 1}</span>{s.title}</p>
                <p className="text-slate-600 mt-1 text-sm leading-relaxed">{s.text}</p>
              </li>
            ))}
          </ol>
          <p className="text-slate-700 bg-blue-50 border border-blue-100 rounded-2xl p-5 leading-relaxed">
            <strong>Evals on every step.</strong> Answer keys reviewed by a person, precision with confidence intervals, a check for
            invented answers, cost and latency per call, and a regression gate in CI that rebuilds the whole dataset from the
            committed files on every push and fails if any number drifts.
          </p>
        </section>

        <section className="space-y-4">
          <h2 className="text-xl font-display font-semibold text-slate-900">Four decisions worth asking about</h2>
          <div className="grid md:grid-cols-2 gap-4">
            {STORIES.map((s) => (
              <div key={s.title} className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
                <p className="font-semibold text-slate-900">{s.title}</p>
                <p className="text-slate-600 mt-2 text-sm leading-relaxed">{s.text}</p>
                <a href={file(s.link)} target="_blank" rel="noopener noreferrer" className="text-sm text-blue-700 hover:underline mt-2 inline-block">
                  The report
                </a>
              </div>
            ))}
          </div>
        </section>

        <section className="space-y-4">
          <h2 className="text-xl font-display font-semibold text-slate-900">Limits, stated up front</h2>
          <ul className="list-disc pl-5 space-y-1.5 text-slate-700 max-w-3xl">
            {LIMITS.map((l) => <li key={l}>{l}</li>)}
          </ul>
        </section>

        <section className="bg-slate-900 text-slate-100 rounded-2xl p-6 sm:p-8 space-y-3">
          <p className="text-lg font-display font-semibold">Built by Sanchit Rastogi</p>
          <p className="text-slate-300 text-sm leading-relaxed">
            Python, Gemini (structured outputs), LangGraph, pandas, SQLite, XGBoost, FastAPI, Docker, GitHub Actions, React and
            d3.js.
          </p>
          <div className="flex flex-wrap gap-3 text-sm pt-1">
            <a href={REPO} target="_blank" rel="noopener noreferrer" className="bg-white text-slate-900 rounded-lg px-3 py-2 font-medium">Source code and README</a>
            <a href={`${API}/docs`} target="_blank" rel="noopener noreferrer" className="border border-slate-600 rounded-lg px-3 py-2">API documentation</a>
            <a href="https://www.linkedin.com/in/sanchit-rastogi-95964b314/" target="_blank" rel="noopener noreferrer" className="border border-slate-600 rounded-lg px-3 py-2">LinkedIn</a>
          </div>
        </section>
      </div>
    </div>
  );
}
