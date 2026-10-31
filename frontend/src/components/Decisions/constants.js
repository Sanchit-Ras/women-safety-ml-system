export const DECISIONS = [
  {
    title: "An LLM Only Where It Earns Its Place",
    desc: "The model resolves the names lookups cannot, and every answer is checked: 0 of 76 answers named a district outside the allowed list, and only answers above a measured confidence skip human review.",
    icon: "🤖",
    bg: "bg-blue-50 border-blue-100",
  },
  {
    title: "Human Decisions Recorded as Data",
    desc: "Every name mapping, district lineage fact and split weight lives in a reviewable file that says who decided and why. The pipeline never overwrites a person's decision.",
    icon: "📝",
    bg: "bg-purple-50 border-purple-100",
  },
  {
    title: "Honest Test Design",
    desc: "Models see only the past, settings are chosen on years kept apart from the headline, and a result counts only if it beats the best simple baseline beyond the noise.",
    icon: "🔒",
    bg: "bg-green-50 border-green-100",
  },
  {
    title: "Batch Scoring, Not Per-Request Inference",
    desc: "The data changes once a year, so the watch list is computed once and served from a file. The API needs no ML libraries, which keeps the server small and its cold start short.",
    icon: "⚡",
    bg: "bg-orange-50 border-orange-100",
  },
];
