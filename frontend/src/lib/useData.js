import { useEffect, useState } from "react";
import { getMap, getSnapshot } from "./api";

function useStatic(load) {
  const [state, setState] = useState({ data: null, error: null });
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let active = true;
    load()
      .then((data) => active && setState({ data, error: null }))
      .catch((error) => active && setState({ data: null, error }));
    return () => {
      active = false;
    };
  }, [load, attempt]);
  return { ...state, retry: () => setAttempt((n) => n + 1) };
}

// { data, error, retry }: data is the API's answers, { deploy_check, districts, watchlist, model_info, district }
export const useSnapshot = () => useStatic(getSnapshot);
export const useMap = () => useStatic(getMap).data;

export const SITE_TITLE = "How has your district changed in 22 years? Crimes against women in India, 2001–2022";

// The browser tab's title for the page on show
export function useTitle(title) {
  useEffect(() => {
    document.title = title ?? SITE_TITLE;
  }, [title]);
}

// Every district with its state's name, for search and the map
export function districtList(snap) {
  return snap ? snap.districts.states.flatMap((s) => s.districts.map((d) => ({ ...d, state: s.name, stateCode: s.code }))) : [];
}
