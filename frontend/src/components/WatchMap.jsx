import { useMemo, useState } from "react";
import { geoMercator, geoPath } from "d3-geo";

const W = 600;
const H = 680;
const FILL = { watch: "#dc2626", no_signal: "#cbd5e1", not_scored: "#f1f5f9" };

export default function WatchMap({ geo, statusByCode, selected, onSelect }) {
  const [hover, setHover] = useState(null);

  const shapes = useMemo(() => {
    if (!geo) return [];
    const path = geoPath(geoMercator().fitSize([W, H], geo));
    return geo.features.map((f) => ({ code: f.properties.code, name: f.properties.name, d: path(f) }));
  }, [geo]);

  if (!geo) {
    return <div className="h-96 flex items-center justify-center text-slate-400">Loading the map...</div>;
  }

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" role="img" aria-label="Map of India's districts with the watch list in red">
        {shapes.map((s) => (
          <path
            key={s.code}
            d={s.d}
            fill={FILL[statusByCode[s.code]] ?? FILL.not_scored}
            stroke={s.code === selected ? "#0f172a" : "#ffffff"}
            strokeWidth={s.code === selected ? 1.5 : 0.3}
            className="cursor-pointer hover:opacity-80"
            onClick={() => onSelect(s.code)}
            onMouseEnter={() => setHover(s)}
            onMouseLeave={() => setHover(null)}
          />
        ))}
      </svg>
      {hover && (
        <div className="absolute top-2 left-2 bg-white/95 border border-slate-200 rounded-lg px-3 py-1.5 text-sm text-slate-700 shadow pointer-events-none">
          {hover.name ?? hover.code}
        </div>
      )}
      <div className="flex flex-wrap gap-4 text-xs text-slate-600 mt-2">
        <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm" style={{ background: FILL.watch }} /> On the watch list</span>
        <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm" style={{ background: FILL.no_signal }} /> Not on it</span>
        <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm border border-slate-300" style={{ background: FILL.not_scored }} /> Not scored</span>
      </div>
    </div>
  );
}
