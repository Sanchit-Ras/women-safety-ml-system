export default function ResearchImpact() {
  return (
    <div className="bg-slate-900 text-white py-24 border-t border-slate-800">
      <div className="max-w-4xl mx-auto px-6 text-center">
        <div className="w-16 h-1 bg-blue-600 mx-auto mb-8 rounded-full"></div>

        <h2 className="text-4xl md:text-5xl font-display font-bold mb-8 leading-tight">
          Research Impact & Future Scope
        </h2>

        <p className="text-xl text-slate-300 leading-relaxed mb-8">
          This project demonstrates how machine learning can be used for{" "}
          <span className="text-blue-400 font-semibold">
            proactive public safety planning
          </span>{" "}
          by identifying high-risk districts before crime surges occur.
        </p>

        <p className="text-slate-500 max-w-2xl mx-auto">
          Future work includes integrating real-time crime data, socioeconomic
          indicators, and deep learning models for even stronger forecasting
          capabilities.
        </p>

        <div className="mt-12 flex justify-center gap-6 opacity-50 hover:opacity-100 transition-opacity">
          {/* Placeholder for logos or additional footer links if needed */}
        </div>
      </div>
    </div>
  );
}
