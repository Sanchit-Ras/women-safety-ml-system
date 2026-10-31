// Every number here comes from a committed report: evals/reports/2026-10-18.md and ml/reports/2026-10-31_bakeoff.md.
export const FINDINGS = [
  {
    title: "Old Place Names, Matched",
    desc: "The 2001–2014 file names 989 police units the way clerks spelled them. Lookups, fuzzy matching, the old district map and Gemini place 98.5% of them on a district, against 56.6% for name matching alone.",
    icon: "🗺️",
    color: "bg-green-100 text-green-700",
  },
  {
    title: "The Model Is Trusted Only Where It Earned It",
    desc: "Above 0.90 confidence the model was right 49 times in 51; below it, 9 in 13. Answers above that cut are accepted without review, cutting the names a person must check from 124 to 42, at a measured cost of 2 wrong names.",
    icon: "🎯",
    color: "bg-blue-100 text-blue-700",
  },
  {
    title: "22 Years on One Map",
    desc: "Districts split and merge. Every split is recorded, and each new district's past is estimated from its share of its parent, so every year adds back to what was filed and national totals never change.",
    icon: "⏳",
    color: "bg-purple-100 text-purple-700",
  },
  {
    title: "Hotspots Can Be Flagged, Fairly",
    desc: "Percentage growth favours tiny districts (1 case to 3 is +200%), so each district is compared only with districts its size. Of the districts flagged, 25% really were among the fastest-growing tenth of their size group (95% interval 22–31%), against 10% by guessing and 14% by size alone. It works best for smaller districts.",
    icon: "📍",
    color: "bg-red-100 text-red-700",
  },
  {
    title: "The Size of Next Year's Change Cannot",
    desc: "Eleven methods were compared, from simple baselines to boosted trees and reconciled forecasts. None forecast how much crime would change better than assuming no change, so the app shows no forecast percentage.",
    icon: "🧭",
    color: "bg-yellow-100 text-yellow-700",
  },
  {
    title: "Leakage Found and Removed",
    desc: "An earlier version reported R² 0.88 by scaling each district with its own future years and splitting by district. Tested honestly, training on the past to predict the next year, that score fell to 0.06.",
    icon: "🛡️",
    color: "bg-slate-200 text-slate-700",
  },
];
