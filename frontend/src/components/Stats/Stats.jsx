import { STATS } from "./constants";

export default function Stats() {
  return (
    <div className="relative py-20 px-6 overflow-hidden">
      <div className="max-w-7xl mx-auto">
        <div className="grid md:grid-cols-4 gap-8 relative z-10">
          {STATS.map((stat, i) => (
            <div
              key={i}
              className="bg-white/80 backdrop-blur-sm rounded-2xl p-8 text-center shadow-lg border border-slate-100 hover:shadow-2xl hover:-translate-y-2 transition-all duration-300 group"
            >
              <h2
                className={`text-5xl font-display font-bold mb-3 ${stat.color} drop-shadow-sm`}
              >
                {stat.value}
              </h2>

              <p className="text-slate-600 font-medium leading-tight group-hover:text-slate-900 transition-colors">
                {stat.label}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
