// The site draws from static files served with the page: the API's own answers saved once (snapshot.json, written
// with the data and checked against the API by its tests) and the district map. So it never waits for the API, which
// runs on a free server that sleeps when unused. The live API is for programs, and for the status light in the footer.
export const API = import.meta.env.VITE_API_URL || "https://women-safety-ml-system-1.onrender.com";
export const REPO = "https://github.com/Sanchit-Ras/women-safety-ml-system";

export async function getJson(path, { signal } = {}) {
  const res = await fetch(`${API}${path}`, { signal });
  if (!res.ok) throw new Error(`${path} returned ${res.status}`);
  return res.json();
}

// Each static file is fetched once per visit and shared by every page; a failed fetch is forgotten so a retry refetches.
const cache = {};
function once(path) {
  cache[path] ??= fetch(path)
    .then((res) => {
      if (!res.ok) throw new Error(`${path} returned ${res.status}`);
      return res.json();
    })
    .catch((e) => {
      delete cache[path];
      throw e;
    });
  return cache[path];
}

export const getSnapshot = () => once("/snapshot.json");
export const getMap = () => once("/districts.geojson");

// One check of the live API per visit, shared by every status light on the page.
let apiCheck;
export function checkApi() {
  apiCheck ??= (async () => {
    const ctrl = new AbortController();
    const give = setTimeout(() => ctrl.abort(), 90000); // a cold start takes about half a minute
    try {
      return await getJson("/deploy-check", { signal: ctrl.signal });
    } finally {
      clearTimeout(give);
    }
  })();
  return apiCheck;
}
