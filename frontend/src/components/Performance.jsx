export default function Performance() {
  return (
    <div className="bg-white/50 py-24 border-t border-slate-100">
      <div className="max-w-7xl mx-auto px-6">
        <h2 className="text-4xl font-display font-bold text-center text-slate-900 mb-16">
          Model Performance Benchmarks
        </h2>

        <div className="grid md:grid-cols-2 gap-8 lg:gap-12 max-w-5xl mx-auto">
          {/* Random Forest Card */}
          <div className="bg-slate-50 p-8 rounded-3xl border border-slate-200">
            <h3 className="text-2xl font-bold text-slate-800 mb-6 flex items-center gap-3">
              <span className="w-3 h-3 rounded-full bg-slate-400"></span>
              Random Forest
            </h3>

            <div className="space-y-6">
              <div>
                <div className="flex justify-between text-sm font-semibold mb-2">
                  <span className="text-slate-500">R² Score</span>
                  <span className="text-slate-900">0.87</span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-2">
                  <div
                    className="bg-slate-400 h-2 rounded-full"
                    style={{ width: "87%" }}
                  ></div>
                </div>
              </div>

              <ul className="space-y-3 pt-4 border-t border-slate-200 text-slate-600 text-sm">
                <li className="flex justify-between">
                  <span>MAE</span>
                  <span className="font-mono font-bold">6.7%</span>
                </li>
                <li className="flex justify-between">
                  <span>Validation Strategy</span>
                  <span>District Holdout</span>
                </li>
              </ul>
            </div>
          </div>

          {/* XGBoost Card */}
          <div className="bg-white p-8 rounded-3xl border border-blue-100 shadow-xl relative overflow-hidden group">
            <div className="absolute top-0 right-0 bg-blue-600 text-white text-xs font-bold px-3 py-1 rounded-bl-xl">
              SELECTED MODEL
            </div>

            <h3 className="text-2xl font-bold text-blue-900 mb-6 flex items-center gap-3">
              <span className="w-3 h-3 rounded-full bg-blue-500 animate-pulse"></span>
              XGBoost
            </h3>

            <div className="space-y-6">
              <div>
                <div className="flex justify-between text-sm font-semibold mb-2">
                  <span className="text-slate-500">R² Score</span>
                  <span className="text-blue-600">0.88</span>
                </div>
                <div className="w-full bg-slate-100 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full shadow-lg shadow-blue-500/30"
                    style={{ width: "88%" }}
                  ></div>
                </div>
              </div>

              <ul className="space-y-3 pt-4 border-t border-slate-100 text-slate-600 text-sm">
                <li className="flex justify-between">
                  <span>MAE</span>
                  <span className="font-mono font-bold text-blue-700">
                    6.4%
                  </span>
                </li>
                <li className="flex justify-between">
                  <span>Optimization</span>
                  <span>Hyperparameter Tuned</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
