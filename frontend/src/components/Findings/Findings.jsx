import { FINDINGS } from "./constants";

export default function Findings() {
  return (
    <div className="bg-slate-50 py-24 px-6 border-y border-slate-200">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-display font-bold text-slate-900 mb-4">
            Key Research Insights
          </h2>
          <p className="text-slate-500 max-w-2xl mx-auto text-lg">
            This project focuses on forecasting crime risk rather than reacting
            to past data — enabling proactive safety planning.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {FINDINGS.map((f, i) => (
            <div
              key={i}
              className="bg-white p-8 rounded-2xl border border-slate-100 shadow-sm hover:shadow-xl transition-all duration-300 hover:-translate-y-1 group"
            >
              <div
                className={`w-14 h-14 ${f.color} rounded-xl flex items-center justify-center text-3xl mb-6 shadow-sm group-hover:scale-110 transition-transform`}
              >
                {f.icon}
              </div>

              <h3 className="text-xl font-bold text-slate-800 mb-3 group-hover:text-blue-700 transition-colors">
                {f.title}
              </h3>

              <p className="text-slate-600 leading-relaxed text-sm">{f.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
