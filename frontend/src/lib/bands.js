// The map's colours for how reported cases changed: blue fell, grey changed little, oranges rose (darker: more). The
// bands, and their names, come from the data (model_info.trends[window].bands, lowest first), so the legend always
// says what the export computed. The 22-year view has one more rising band than the recent one.
const BLUE = "#2563eb";
const GREY = "#94a3b8";
const ORANGES = { 2: ["#fdba74", "#ea580c"], 3: ["#fdba74", "#f97316", "#c2410c"] };
export const NO_COMPARISON = "#eef2f6";

// band name -> colour, for one window's bands
export function bandColors(bands) {
  const palette = [BLUE, GREY, ...ORANGES[bands.length - 2]];
  return Object.fromEntries(bands.map((b, i) => [b.label, palette[i]]));
}

const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);

// The legend, with how many districts the map shows in each band
export function bandLegend(bands, counts = {}) {
  const colors = bandColors(bands);
  const n = (label) => (counts[label] != null ? ` (${counts[label]})` : "");
  return [
    ...bands.map((b) => ({ color: colors[b.label], label: `${cap(b.label)}${n(b.label)}` })),
    { color: NO_COMPARISON, label: `Too few cases, or no comparable years${counts.none != null ? ` (${counts.none})` : ""}` },
  ];
}

export const STATUS_COLOR = { watch: "#dc2626", no_signal: "#cbd5e1", not_scored: "#f1f5f9" };
