// The API is on Render's free tier: the first request after a quiet spell wakes the server, which can take a minute.
const BASE = import.meta.env.VITE_API_URL;
export const REPO = "https://github.com/Sanchit-Ras/women-safety-ml-system";

export async function getJson(path) {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${path} returned ${res.status}`);
  return res.json();
}

// The district map is a static file served with the frontend, not by the API.
export async function getMap() {
  const res = await fetch("/districts.geojson");
  if (!res.ok) throw new Error(`map returned ${res.status}`);
  return res.json();
}

export const pct = (x) => `${Math.round(x * 100)}%`;
