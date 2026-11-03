import { useId, useMemo, useState } from "react";

const norm = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();

// Type a district's name, pick it from the list (arrow keys and Enter work too).
export default function DistrictSearch({ items, onSelect, autoFocus = false }) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const listId = useId();

  const matches = useMemo(() => {
    const q = norm(query);
    if (!q) return [];
    const hits = items.filter((d) => norm(d.name).includes(q) || norm(`${d.name} ${d.state}`).includes(q));
    const starts = (d) => (norm(d.name).startsWith(q) ? 0 : 1);
    return hits.sort((a, b) => starts(a) - starts(b) || a.name.localeCompare(b.name)).slice(0, 8);
  }, [items, query]);

  const pick = (d) => {
    onSelect(d.code);
    setQuery("");
    setOpen(false);
  };

  const onKey = (e) => {
    if (!matches.length) return;
    if (e.key === "ArrowDown") { e.preventDefault(); setActive((i) => (i + 1) % matches.length); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setActive((i) => (i - 1 + matches.length) % matches.length); }
    else if (e.key === "Enter") { e.preventDefault(); pick(matches[Math.min(active, matches.length - 1)]); }
    else if (e.key === "Escape") setOpen(false);
  };

  return (
    <div className="relative">
      <input
        type="search"
        role="combobox"
        aria-expanded={open && matches.length > 0}
        aria-controls={listId}
        aria-label="Search for a district"
        autoFocus={autoFocus}
        value={query}
        onChange={(e) => { setQuery(e.target.value); setOpen(true); setActive(0); }}
        onFocus={() => setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        onKeyDown={onKey}
        placeholder="Search your district, e.g. Lucknow"
        className="w-full bg-white border border-slate-300 rounded-xl pl-11 pr-4 py-3 text-base text-slate-800 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
      />
      <svg className="w-5 h-5 text-slate-400 absolute left-4 top-1/2 -translate-y-1/2 pointer-events-none" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
        <circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" />
      </svg>
      {open && query && (
        <ul id={listId} role="listbox" className="absolute z-20 mt-1 w-full bg-white border border-slate-200 rounded-xl shadow-lg overflow-hidden">
          {matches.length === 0 && <li className="px-4 py-2.5 text-sm text-slate-500">No district matches "{query}"</li>}
          {matches.map((d, i) => (
            <li
              key={d.code}
              role="option"
              aria-selected={i === active}
              onMouseDown={(e) => { e.preventDefault(); pick(d); }}
              onMouseEnter={() => setActive(i)}
              className={`px-4 py-2.5 text-sm cursor-pointer ${i === active ? "bg-blue-50 text-blue-900" : "text-slate-700"}`}
            >
              {d.name}<span className="text-slate-400">, {d.state}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
