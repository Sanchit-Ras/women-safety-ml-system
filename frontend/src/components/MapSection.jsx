export default function MapSection() {
  return (
    <div className="max-w-7xl mx-auto px-6 pb-24">
      <div className="bg-slate-900 rounded-3xl shadow-2xl overflow-hidden relative group">
        {/* Abstract Background */}
        <div className="absolute top-0 right-0 w-96 h-96 bg-blue-600 rounded-full blur-3xl opacity-20 -translate-y-1/2 translate-x-1/3 group-hover:opacity-30 transition-opacity duration-500"></div>

        <div className="grid lg:grid-cols-2 gap-0">
          {/* Text Content */}
          <div className="p-12 lg:p-16 flex flex-col justify-center relative z-10">
            <span className="text-blue-400 font-bold tracking-wider uppercase text-sm mb-2">
              Live Demo
            </span>
            <h2 className="text-4xl font-display font-bold text-white mb-6">
              Interactive Crime Risk Forecast
            </h2>
            <p className="text-slate-300 text-lg leading-relaxed mb-8">
              Explore the visualization of predicted crime risk levels across
              Indian districts. The model analyzes historical crime trends to
              forecast future risk zones, enabling early identification of
              vulnerable regions.
            </p>

            <a
              href="https://Sanchit-Ras.github.io/crime-risk-map/"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-3 px-8 py-4 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded-xl shadow-lg shadow-blue-900/30 transition-all transform hover:-translate-y-1 w-fit"
            >
              View Interactive Map
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-5 w-5"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M17 8l4 4m0 0l-4 4m4-4H3"
                />
              </svg>
            </a>
          </div>

          {/* Map Image */}
          <div className="relative h-full min-h-[400px] lg:min-h-0">
            <a
              href="https://Sanchit-Ras.github.io/crime-risk-map/"
              target="_blank"
              rel="noopener noreferrer"
              className="absolute inset-0 block overflow-hidden"
            >
              <div className="absolute inset-0 bg-linear-to-l from-slate-900 via-transparent to-transparent z-10 lg:hidden"></div>
              <img
                src="/map-preview.png"
                alt="Crime Risk Forecast Map"
                className="w-full h-full object-cover transform group-hover:scale-105 transition-transform duration-700"
              />
              <div className="absolute inset-0 bg-blue-900/0 group-hover:bg-blue-900/10 transition-colors duration-300"></div>
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}
