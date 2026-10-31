import { Link } from "react-router-dom";

export default function MapSection() {
  return (
    <div className="max-w-7xl mx-auto px-6 pb-24">
      <div className="bg-slate-900 rounded-3xl shadow-2xl overflow-hidden relative group">
        <div className="absolute top-0 right-0 w-96 h-96 bg-blue-600 rounded-full blur-3xl opacity-20 -translate-y-1/2 translate-x-1/3 group-hover:opacity-30 transition-opacity duration-500"></div>

        <div className="grid lg:grid-cols-2 gap-0">
          <div className="p-12 lg:p-16 flex flex-col justify-center relative z-10">
            <span className="text-blue-400 font-bold tracking-wider uppercase text-sm mb-2">
              Live App
            </span>
            <h2 className="text-4xl font-display font-bold text-white mb-6">
              The District Watch List
            </h2>
            <p className="text-slate-300 text-lg leading-relaxed mb-8">
              A map of every district on today's map of India with its reported crimes against women since 2001, and the
              districts most likely to be among the fastest-growing next year.
            </p>

            <Link
              to="/predictor"
              className="inline-flex items-center gap-3 px-8 py-4 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded-xl shadow-lg shadow-blue-900/30 transition-all transform hover:-translate-y-1 w-fit"
            >
              Open the Watch List
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
              </svg>
            </Link>
          </div>

          <div className="relative p-12 lg:p-16 flex flex-col justify-center gap-6 border-t lg:border-t-0 lg:border-l border-slate-800">
            <div>
              <p className="text-5xl font-display font-bold text-red-400">755</p>
              <p className="text-slate-400">districts, each with 22 years of numbers</p>
            </div>
            <div>
              <p className="text-5xl font-display font-bold text-blue-400">37%</p>
              <p className="text-slate-400">of flagged districts were truly among the fastest-growing tenth (chance: 10%)</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
