const STAGES = [
  { title: "NCRB files", desc: "4 formats, 2001–2022" },
  { title: "Crosswalk", desc: "columns → 7 offence categories" },
  { title: "Name resolver", desc: "rules, map overlap, Gemini, calibrated cut" },
  { title: "Lineage & re-basing", desc: "every year on today's 755 districts" },
  { title: "Bake-off", desc: "11 methods, rolling-origin test" },
  { title: "Batch scoring", desc: "the watch list, once a year" },
  { title: "FastAPI", desc: "serves a file, no model at request time" },
  { title: "React app", desc: "map, watch list, district numbers" },
];

export default function Architecture() {
  return (
    <div className="py-24 bg-slate-50 border-y border-slate-200">
      <div className="max-w-7xl mx-auto px-6 text-center">
        <span className="text-blue-600 font-bold tracking-wider uppercase text-sm">
          System Design
        </span>
        <h2 className="text-4xl font-display font-bold text-slate-900 mt-2 mb-6">
          Architecture
        </h2>

        <p className="text-lg text-slate-600 max-w-2xl mx-auto mb-12">
          From raw government tables to a live app, with human decisions recorded as data and evals guarding every step.
        </p>

        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4 max-w-5xl mx-auto text-left">
          {STAGES.map((s, i) => (
            <div key={s.title} className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm relative">
              <span className="text-xs font-bold text-blue-600">{String(i + 1).padStart(2, "0")}</span>
              <p className="font-bold text-slate-900 mt-1">{s.title}</p>
              <p className="text-sm text-slate-500 mt-1">{s.desc}</p>
            </div>
          ))}
        </div>

        <div className="max-w-5xl mx-auto mt-4 bg-blue-600 text-white rounded-2xl p-5 text-left sm:text-center">
          <p className="font-bold">Evals across every stage</p>
          <p className="text-sm text-blue-100 mt-1">
            Frozen answer keys, dated reports, versioned prompts, and a gate on every push that rebuilds the panel from the
            committed files and checks the metrics against their floors.
          </p>
        </div>
      </div>
    </div>
  );
}
