import { useEffect, useState } from "react";
import { API, checkApi } from "../lib/api";

const LOOK = {
  checking: { dot: "bg-slate-300", text: "API: checking" },
  waking: { dot: "bg-amber-400 animate-pulse", text: "API: waking up (free server, about 30 s)" },
  live: { dot: "bg-emerald-500", text: "API: live" },
  down: { dot: "bg-slate-400", text: "API: asleep or unreachable (the site works without it)" },
};

// A light showing whether the live API answers. The site never waits for it: its numbers come with the page.
export default function ApiStatus() {
  const [state, setState] = useState("checking");
  useEffect(() => {
    let active = true;
    const slow = setTimeout(() => active && setState((s) => (s === "checking" ? "waking" : s)), 2500);
    checkApi()
      .then(() => active && setState("live"))
      .catch(() => active && setState("down"))
      .finally(() => clearTimeout(slow));
    return () => {
      active = false;
      clearTimeout(slow);
    };
  }, []);
  const look = LOOK[state];
  return (
    <a href={`${API}/docs`} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-2 hover:underline">
      <span className={`w-2 h-2 rounded-full ${look.dot}`} />
      {look.text}
      {state === "live" && <span className="text-slate-400">· docs</span>}
    </a>
  );
}
