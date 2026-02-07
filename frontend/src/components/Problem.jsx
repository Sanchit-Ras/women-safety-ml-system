export default function Problem() {
  return (
    <div className="bg-white py-24">
      <div className="max-w-7xl mx-auto px-6 grid md:grid-cols-2 gap-12 items-center">
        <div className="relative">
          <div className="absolute -top-4 -left-4 w-24 h-24 bg-red-100 rounded-full blur-2xl opacity-60"></div>
          <h2 className="relative text-lg font-bold text-red-600 uppercase tracking-widest mb-2">
            The Challenge
          </h2>
          <h3 className="text-4xl font-display font-bold text-slate-900 mb-6 leading-tight">
            Moving from{" "}
            <span className="text-red-600 underline decoration-red-200 decoration-4 underline-offset-4">
              Reactive
            </span>{" "}
            to <br />
            <span className="text-blue-600 underline decoration-blue-200 decoration-4 underline-offset-4">
              Proactive
            </span>{" "}
            Safety
          </h3>
          <p className="text-lg text-slate-600 leading-relaxed mb-6">
            Traditional crime analysis is typically reactive — resources are
            deployed only after crime patterns emerge and incidents occur. This
            leaves a critical gap in prevention.
          </p>
          <p className="text-lg text-slate-600 leading-relaxed border-l-4 border-blue-500 pl-6 italic">
            "Does machine learning have the capability to identifying
            district-level crime risk before escalation occurs?"
          </p>
        </div>

        <div className="grid grid-cols-2 gap-6">
          <div className="bg-slate-50 p-6 rounded-2xl border border-slate-100 shadow-sm hover:shadow-md transition-shadow">
            <div className="w-12 h-12 bg-red-100 rounded-lg flex items-center justify-center mb-4 text-2xl">
              🚨
            </div>
            <h4 className="font-bold text-slate-800 mb-2">Reactive Policing</h4>
            <p className="text-sm text-slate-500">
              Responding to events after they happen, limited prevention
              capability.
            </p>
          </div>

          <div className="bg-blue-50 p-6 rounded-2xl border border-blue-100 shadow-sm hover:shadow-md transition-shadow mt-8">
            <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4 text-2xl">
              🛡️
            </div>
            <h4 className="font-bold text-blue-900 mb-2">Predictive Safety</h4>
            <p className="text-sm text-blue-700">
              Forecasting high-risk zones to enable preventative resource
              allocation.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
