export default function Architecture() {
  return (
    <div className="py-24 bg-slate-50 border-y border-slate-200">
      <div className="max-w-7xl mx-auto px-6 text-center">
        <span className="text-blue-600 font-bold tracking-wider uppercase text-sm">
          System Design
        </span>
        <h2 className="text-4xl font-display font-bold text-slate-900 mt-2 mb-6">
          Model Architecture
        </h2>

        <p className="text-lg text-slate-600 max-w-2xl mx-auto mb-12">
          An end-to-end pipeline designed for robust district-level crime
          forecasting.
        </p>

        <div className="bg-white p-4 rounded-3xl shadow-xl border border-slate-100 inline-block overflow-hidden hover:shadow-2xl transition-shadow duration-500">
          <img
            src="/pipeline.png"
            className="rounded-2xl w-full max-w-5xl h-auto"
            alt="Model Pipeline Diagram"
          />
        </div>
      </div>
    </div>
  );
}
