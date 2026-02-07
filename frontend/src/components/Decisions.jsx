export default function Decisions() {
  const decisions = [
    {
      title: "District Holdout Validation",
      desc: "Instead of random splitting, the model was tested on completely unseen districts to simulate real-world deployment.",
      icon: "🏙️",
      bg: "bg-blue-50 border-blue-100",
    },
    {
      title: "Sliding Window Forecasting",
      desc: "Converted time-series crime data into supervised learning format using lag features.",
      icon: "📅",
      bg: "bg-purple-50 border-purple-100",
    },
    {
      title: "Leakage Prevention",
      desc: "Applied strict time-based splits to ensure the model never trained on future data.",
      icon: "🔒",
      bg: "bg-green-50 border-green-100",
    },
    {
      title: "Dual Model Strategy",
      desc: "Used Random Forest for robustness and XGBoost for high predictive power.",
      icon: "🤖",
      bg: "bg-orange-50 border-orange-100",
    },
  ];

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
          {decisions.map((d, i) => (
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
