// Every number here comes from a committed report: evals/reports/2026-10-18.md and ml/reports/2026-10-31_bakeoff.md
// (tests/test_export_app.py checks the watch-list numbers against the app's data).
export const STATS = [
  {
    value: "22 Years",
    label: "Of NCRB district data, 2001–2022",
    color: "text-blue-600",
  },
  {
    value: "755",
    label: "Districts, every year re-based onto today's map",
    color: "text-purple-600",
  },
  {
    value: "98.5%",
    label: "Old place names matched to a district (name matching alone: 56.6%)",
    color: "text-green-600",
  },
  {
    value: "2.5×",
    label: "Better than guessing at flagging next year's fastest-growing districts, each compared with districts its size",
    color: "text-red-600",
  },
];
