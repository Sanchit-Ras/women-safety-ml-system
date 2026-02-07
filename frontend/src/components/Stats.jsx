export default function Stats() {
  const stats = [
    {
      value: "600+",
      label: "Districts Modeled Across India",
      color: "text-blue-600",
    },
    {
      value: "14 Years",
      label: "Historical Crime Data Used",
      color: "text-purple-600",
    },
    {
      value: "0.88 R²",
      label: "Future Crime Prediction Score",
      color: "text-green-600",
    },
    {
      value: "District Holdout",
      label: "Validation Strategy (Prevents Memorization)",
      color: "text-red-600",
    },
  ];

  return (
    <div className="relative py-20 px-6 overflow-hidden">
      <div className="max-w-7xl mx-auto">
        <div className="grid md:grid-cols-4 gap-8 relative z-10">
          {stats.map((stat, i) => (
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
