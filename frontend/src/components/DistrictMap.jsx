import { useMemo, useState } from "react";
import { geoMercator, geoPath } from "d3-geo";

const W = 600;
const H = 680;

// Today's districts, coloured by colorOf(code). describe(code) adds a line to the hover label.
export default function DistrictMap({ geo, colorOf, legend, selected, onSelect, describe, label }) {
  const [hover, setHover] = useState(null);

  const shapes = useMemo(() => {
    if (!geo) return [];
    const path = geoPath(geoMercator().fitSize([W, H], geo));
    return geo.features.map((f) => ({ code: f.properties.code, name: f.properties.name, d: path(f) }));
  }, [geo]);

  if (!geo) {
    return <div className="aspect-[600/680] flex items-center justify-center text-slate-400">Loading the map...</div>;
  }

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" role="img" aria-label={label}>
        {shapes.map((s) => (
          <path
            key={s.code}
            d={s.d}
            fill={colorOf(s.code)}
            stroke={s.code === selected ? "#0f172a" : "#ffffff"}
            strokeWidth={s.code === selected ? 1.6 : 0.3}
            className="cursor-pointer hover:opacity-75"
            onClick={() => onSelect(s.code)}
            onMouseEnter={() => setHover(s)}
            onMouseLeave={() => setHover(null)}
          />
        ))}
      </svg>
      {hover && (
        <div className="absolute top-2 left-2 bg-white/95 border border-slate-200 rounded-lg px-3 py-1.5 text-sm text-slate-700 shadow pointer-events-none">
          <span className="font-semibold">{hover.name ?? hover.code}</span>
          {describe?.(hover.code) && <span className="block text-xs text-slate-500">{describe(hover.code)}</span>}
        </div>
      )}
      <div className="flex flex-wrap gap-x-4 gap-y-1.5 text-xs text-slate-600 mt-3">
        {legend.map((l) => (
          <span key={l.label} className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-sm border border-slate-300/60" style={{ background: l.color }} /> {l.label}
          </span>
        ))}
      </div>
    </div>
  );
}
