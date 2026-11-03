const W = 420;
const H = 200;
const PAD = { right: 10, top: 28, bottom: 26 };
const LAW = 2013; // the Criminal Law (Amendment) Act, 2013 widened what counts as rape and sexual harassment

// series: [year, total, kind]; kind 0 = as filed, 1 = adjusted (filed minus a later district's part), 2 = estimated.
// compared: [from, to], the years the summary compares, shaded.
export default function Sparkline({ series, compared }) {
  if (!series?.length) return null;
  const years = series.map((p) => p[0]);
  const max = Math.max(1, ...series.map((p) => p[1]));
  const top = Math.round(max).toLocaleString("en-IN");
  const pad = { ...PAD, left: Math.max(36, 12 + 7 * top.length) }; // room for the largest number: 4,517 or 3,54,896
  const x = (y) => pad.left + ((y - years[0]) / Math.max(1, years.at(-1) - years[0])) * (W - pad.left - pad.right);
  const yv = (v) => H - pad.bottom - (v / max) * (H - pad.top - pad.bottom);
  const line = series.map((p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(1)},${yv(p[1]).toFixed(1)}`).join(" ");
  const showLaw = years[0] < LAW && years.at(-1) > LAW;
  const ticks = years.filter((y) => y % 5 === 0 && y - years[0] >= 3 && years.at(-1) - y >= 3);

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" role="img" aria-label="Reported crimes against women by year">
        {compared && (
          <rect x={x(compared[0])} y={pad.top} width={x(compared[1]) - x(compared[0])} height={H - pad.top - pad.bottom} fill="#fff7ed" />
        )}
        <line x1={pad.left} x2={W - pad.right} y1={H - pad.bottom} y2={H - pad.bottom} stroke="#e2e8f0" />
        <line x1={pad.left} x2={W - pad.right} y1={yv(max)} y2={yv(max)} stroke="#f1f5f9" />
        <text x={pad.left - 6} y={yv(max) + 4} textAnchor="end" className="fill-slate-400 text-[12px]">{top}</text>
        <text x={pad.left - 6} y={H - pad.bottom + 4} textAnchor="end" className="fill-slate-400 text-[12px]">0</text>
        {[years[0], ...ticks, years.at(-1)].map((y, i, all) => (
          <text key={y} x={x(y)} y={H - 7} textAnchor={i === 0 ? "start" : i === all.length - 1 ? "end" : "middle"} className="fill-slate-400 text-[12px]">{y}</text>
        ))}
        {showLaw && (
          <g>
            <line x1={x(LAW)} x2={x(LAW)} y1={pad.top - 8} y2={H - pad.bottom} stroke="#94a3b8" strokeDasharray="3 3" />
            <text x={x(LAW) + 4} y={pad.top - 10} className="fill-slate-500 text-[12px]">2013 law change</text>
          </g>
        )}
        <path d={line} fill="none" stroke="#2563eb" strokeWidth="2" />
        {series.map(([year, total, kind]) => (
          <circle key={year} cx={x(year)} cy={yv(total)} r="3.5"
                  fill={kind === 0 ? "#2563eb" : kind === 1 ? "#93c5fd" : "#ffffff"} stroke="#2563eb" strokeWidth="1.5">
            <title>{`${year}: ${Math.round(total).toLocaleString("en-IN")}${kind === 1 ? " (adjusted)" : kind === 2 ? " (estimated)" : ""}`}</title>
          </circle>
        ))}
      </svg>
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-500 mt-1">
        <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-blue-600" /> As filed</span>
        <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-blue-300 border border-blue-600" /> Adjusted</span>
        <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-white border border-blue-600" /> Estimated</span>
        {compared && <span className="flex items-center gap-1.5"><span className="w-3 h-2.5 bg-orange-50 border border-orange-200" /> Recent change: {compared[0]}–{compared[1]}</span>}
      </div>
      {showLaw && (
        <p className="text-xs text-slate-500 mt-2">
          In 2013, after the Delhi gang-rape case, the law widened what counts as rape and sexual harassment, so part of the
          rise after 2013 is a change in what is counted.
        </p>
      )}
    </div>
  );
}
