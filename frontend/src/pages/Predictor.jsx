import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import BootSequence from "../components/BootSequence";
import DistrictCard from "../components/DistrictCard";
import WatchExplanation from "../components/WatchExplanation";
import WatchMap from "../components/WatchMap";
import { REPO, getJson, getMap } from "../lib/api";

const MAX_RETRIES = 12; // about a minute (12 x 5 s): the free server may be waking up

export default function Predictor() {
  const [base, setBase] = useState(null); // { districts, meta, watch } from the API
  const [geo, setGeo] = useState(null);
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState("");
  const [retryCount, setRetryCount] = useState(0);
  const [params, setParams] = useSearchParams();
  const code = params.get("district") ? Number(params.get("district")) : null;

  // The API: districts, the model card and the watch list, retried while the server wakes up
  useEffect(() => {
    let cancelled = false;
    Promise.all([getJson("/districts"), getJson("/model-info"), getJson("/watchlist")])
      .then(([districts, meta, watch]) => !cancelled && setBase({ districts, meta, watch }))
      .catch(() => {
        if (cancelled) return;
        if (retryCount < MAX_RETRIES) setTimeout(() => setRetryCount((n) => n + 1), 5000);
        else setError("Could not reach the server. Please try again in a minute.");
      });
    return () => { cancelled = true; };
  }, [retryCount]);

  // The map: a static file served with the page
  useEffect(() => {
    getMap().then(setGeo).catch(() => setGeo(null));
  }, []);

  // The selected district's numbers
  useEffect(() => {
    if (!code || !base) return;
    let cancelled = false;
    getJson(`/district/${code}`)
      .then((d) => !cancelled && setDetail(d))
      .catch(() => !cancelled && setDetail({ error: true, code }));
    return () => { cancelled = true; };
  }, [code, base]);

  const statusByCode = useMemo(() => {
    const out = {};
    base?.districts.states.forEach((s) => s.districts.forEach((d) => { out[d.code] = d.status; }));
    return out;
  }, [base]);

  const stateOf = useMemo(() => {
    const out = {};
    base?.districts.states.forEach((s) => s.districts.forEach((d) => { out[d.code] = s.code; }));
    return out;
  }, [base]);

  const [pickedState, setPickedState] = useState(null);
  const stateCode = pickedState ?? (code ? stateOf[code] : null);
  const state = base?.districts.states.find((s) => s.code === stateCode);
  const select = (c) => setParams(c ? { district: String(c) } : {});
  const shown = detail && detail.code === code && !detail.error ? detail : null;

  return (
    <div className="min-h-screen bg-linear-to-br from-slate-50 to-blue-50 pt-28 pb-12 px-4 sm:px-6 lg:px-8 font-sans">
      <div className="max-w-6xl mx-auto">
        <div className="bg-white rounded-3xl shadow-xl overflow-hidden border border-slate-100">
          <div className="bg-slate-900 text-white p-8 sm:p-10 text-center">
            <h1 className="text-3xl sm:text-4xl font-display font-bold mb-3 tracking-tight">
              District Watch List {base?.meta.watch_year ?? ""}
            </h1>
            <p className="text-slate-300 text-lg max-w-3xl mx-auto font-light">
              Reported crimes against women for every district on today's map of India, and the districts most likely to be
              among the fastest-growing next year.
            </p>
          </div>

          <div className="p-6 sm:p-10 space-y-8">
            {error ? (
              <div className="p-4 bg-red-50 text-red-700 rounded-xl border border-red-100">{error}</div>
            ) : !base ? (
              <BootSequence />
            ) : (
              <>
                <WatchExplanation meta={base.meta} />

                <div className="grid lg:grid-cols-5 gap-8 items-start">
                  <div className="lg:col-span-3 bg-slate-50 rounded-2xl p-4 border border-slate-200">
                    <WatchMap geo={geo} statusByCode={statusByCode} selected={code} onSelect={select} />
                  </div>

                  <div className="lg:col-span-2 space-y-4">
                    <div className="bg-slate-50 p-5 rounded-2xl border border-slate-200 space-y-3">
                      <p className="font-semibold text-slate-800">Pick a district, or click one on the map</p>
                      <select
                        className="w-full bg-white border border-slate-300 text-slate-700 py-2.5 px-3 rounded-xl"
                        value={stateCode ?? ""}
                        onChange={(e) => { setPickedState(Number(e.target.value)); select(null); }}
                      >
                        <option value="" disabled>Select a state</option>
                        {base.districts.states.map((s) => <option key={s.code} value={s.code}>{s.name}</option>)}
                      </select>
                      <select
                        className="w-full bg-white border border-slate-300 text-slate-700 py-2.5 px-3 rounded-xl disabled:bg-slate-100"
                        value={code && stateOf[code] === stateCode ? code : ""}
                        disabled={!state}
                        onChange={(e) => select(Number(e.target.value))}
                      >
                        <option value="" disabled>Select a district</option>
                        {state?.districts.map((d) => <option key={d.code} value={d.code}>{d.name}</option>)}
                      </select>
                    </div>

                    <div className="bg-white rounded-2xl border border-slate-200 overflow-hidden">
                      <p className="px-5 py-3 font-semibold text-slate-800 border-b border-slate-100">
                        On the watch list ({base.watch.districts.length} districts)
                      </p>
                      <ol className="max-h-80 overflow-y-auto divide-y divide-slate-100 text-sm">
                        {base.watch.districts.map((d) => (
                          <li key={d.code}>
                            <button
                              onClick={() => { setPickedState(null); select(d.code); }}
                              className={`w-full text-left px-5 py-2 hover:bg-red-50 flex justify-between gap-3 ${d.code === code ? "bg-red-50" : ""}`}
                            >
                              <span><span className="text-slate-400 mr-2">{d.rank}.</span>{d.name}<span className="text-slate-400">, {d.state}</span></span>
                              <span className="text-slate-500 font-mono">{d.latest_total == null ? "-" : Math.round(d.latest_total)}</span>
                            </button>
                          </li>
                        ))}
                      </ol>
                      <p className="px-5 py-2 text-xs text-slate-400 border-t border-slate-100">
                        Number: crimes reported in {base.meta.data_years[1]}. Districts assessed together share a rank.
                      </p>
                    </div>
                  </div>
                </div>

                {code && !shown && !(detail?.error && detail.code === code) && (
                  <div className="text-center text-slate-500">Loading the district...</div>
                )}
                {detail?.error && detail.code === code && (
                  <div className="p-4 bg-red-50 text-red-700 rounded-xl border border-red-100">That district could not be loaded.</div>
                )}
                {shown && <DistrictCard district={shown} meta={base.meta} />}
              </>
            )}
          </div>

          <div className="bg-slate-50 border-t border-slate-100 p-4 text-center text-xs text-slate-500">
            Data: NCRB district tables 2001-2022, re-based onto today's districts. Model: {base?.meta.model.name ?? "hotspot classifier"},
            batch-scored.{" "}
            {base && (
              <a className="text-blue-600 hover:underline" href={`${REPO}/blob/main/${base.meta.evidence.report}`} target="_blank" rel="noopener noreferrer">
                The evidence report
              </a>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
