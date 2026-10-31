const W = 600;
const H = 170;
const PAD = { left: 44, right: 12, top: 14, bottom: 26 };

// series: [year, total, kind]; kind 0 = as filed, 1 = adjusted (filed minus a later district's part), 2 = estimated
export default function Sparkline({ series }) {
  if (!series?.length) return null;
  const years = series.map((p) => p[0]);
  const max = Math.max(1, ...series.map((p) => p[1]));
  const x = (y) => PAD.left + ((y - years[0]) / Math.max(1, years.at(-1) - years[0])) * (W - PAD.left - PAD.right);
  const yv = (v) => H - PAD.bottom - (v / max) * (H - PAD.top - PAD.bottom);
  const line = series.map((p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(1)},${yv(p[1]).toFixed(1)}`).join(" ");

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" role="img" aria-label="Total crimes against women by year">
        <line x1={PAD.left} x2={W - PAD.right} y1={H - PAD.bottom} y2={H - PAD.bottom} stroke="#e2e8f0" />
        <text x={PAD.left - 6} y={yv(max) + 4} textAnchor="end" className="fill-slate-400 text-[11px]">{Math.round(max)}</text>
        <text x={PAD.left - 6} y={H - PAD.bottom + 4} textAnchor="end" className="fill-slate-400 text-[11px]">0</text>
        <text x={x(years[0])} y={H - 6} textAnchor="middle" className="fill-slate-400 text-[11px]">{years[0]}</text>
        <text x={x(years.at(-1))} y={H - 6} textAnchor="end" className="fill-slate-400 text-[11px]">{years.at(-1)}</text>
        <path d={line} fill="none" stroke="#2563eb" strokeWidth="2" />
        {series.map(([year, total, kind]) => (
          <circle key={year} cx={x(year)} cy={yv(total)} r="3.5"
                  fill={kind === 0 ? "#2563eb" : kind === 1 ? "#93c5fd" : "#ffffff"} stroke="#2563eb" strokeWidth="1.5">
            <title>{`${year}: ${Math.round(total)}${kind === 1 ? " (adjusted)" : kind === 2 ? " (estimated)" : ""}`}</title>
          </circle>
        ))}
      </svg>
      <div className="flex flex-wrap gap-4 text-xs text-slate-500 mt-1">
        <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-blue-600" /> As filed</span>
        <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-blue-300 border border-blue-600" /> Adjusted</span>
        <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-white border border-blue-600" /> Estimated</span>
      </div>
    </div>
  );
}
