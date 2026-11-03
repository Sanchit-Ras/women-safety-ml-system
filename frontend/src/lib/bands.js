// The map's colours for how reported cases changed. The bounds come from the data (model_info.trend.bands), so the
// legend always says what the export computed: blue fell, grey held steady, orange rose.
const COLOR = { fell: "#2563eb", steady: "#94a3b8", rose: "#fdba74", rose_sharply: "#ea580c" };
export const NO_COMPARISON = "#eef2f6";

export const bandColor = (band) => COLOR[band] ?? NO_COMPARISON;

const signed = (x) => `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(Math.round(x * 100))}%`;

export function bandLegend(bands) {
  return [
    ...bands.map((b, i) => {
      const prev = bands[i - 1]?.below;
      const label =
        i === 0 ? `Down more than ${Math.abs(Math.round(b.below * 100))}%`
          : b.below == null ? `Up ${Math.round(prev * 100)}% or more`
            : `${signed(prev)} to ${signed(b.below)}`;
      return { color: bandColor(b.band), label };
    }),
    { color: NO_COMPARISON, label: "Too few cases, or no comparable years" },
  ];
}

export const STATUS_COLOR = { watch: "#dc2626", no_signal: "#cbd5e1", not_scored: "#f1f5f9" };
