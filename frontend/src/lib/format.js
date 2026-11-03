export const num = (n) => Math.round(n).toLocaleString("en-IN");
export const pct = (x) => `${Math.round(x * 100)}%`;

// "up 31%", "down 4%", "about the same"
export function changeText(change) {
  const p = Math.round(change * 100);
  return p > 0 ? `up ${p}%` : p < 0 ? `down ${-p}%` : "about the same";
}

export function ordinal(n) {
  const s = ["th", "st", "nd", "rd"];
  const v = n % 100;
  return `${n}${s[(v - 20) % 10] || s[v] || s[0]}`;
}

export const listText = (names) =>
  names.length <= 1 ? names.join("") : `${names.slice(0, -1).join(", ")} and ${names.at(-1)}`;

// A small CSV file the browser saves, made from rows already on the page
export function downloadCsv(filename, header, rows) {
  const cell = (v) => (typeof v === "string" && /[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : String(v ?? ""));
  const text = [header, ...rows].map((r) => r.map(cell).join(",")).join("\n");
  const url = URL.createObjectURL(new Blob([text + "\n"], { type: "text/csv;charset=utf-8" }));
  const a = Object.assign(document.createElement("a"), { href: url, download: filename });
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
