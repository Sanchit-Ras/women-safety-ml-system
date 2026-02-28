import { DECISIONS } from "./constants";

export default function Decisions() {
  return (
    <div className="bg-white py-24">
      <div className="max-w-7xl mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-display font-bold text-slate-900 mb-6">
            Key Technical Decisions
          </h2>
          <p className="text-lg text-slate-600 max-w-2xl mx-auto">
            Strategic architectural choices made to ensure robustness, accuracy,
            and real-world applicability.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          {DECISIONS.map((d, i) => (
            <div
              key={i}
              className={`p-8 rounded-2xl border ${d.bg} transition-transform hover:-translate-y-1 duration-300`}
            >
              <div className="flex items-start gap-6">
                <div className="text-4xl shadow-sm bg-white p-3 rounded-xl">
                  {d.icon}
                </div>
                <div>
                  <h3 className="text-xl font-bold text-slate-900 mb-2">
                    {d.title}
                  </h3>
                  <p className="text-slate-600 leading-relaxed text-sm">
                    {d.desc}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
