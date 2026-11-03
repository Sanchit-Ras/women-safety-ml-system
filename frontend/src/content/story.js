// Every number the "How it was built" page quotes, in one place. tests/test_export_app.py checks each one against the
// report it comes from, so the page cannot drift from the evidence.
const EVAL = "evals/reports/2026-10-18.md";

export const RESULTS = [
  { value: "98.5%", what: "of 989 messy old place names matched to today's districts", vs: "rule-based matching: 56.6%", report: EVAL },
  { value: "0 of 76", what: "model answers named a district outside the allowed list", vs: "a guard rejects any that would", report: EVAL },
  { value: "124 → 42", what: "names a person must check, after a measured confidence threshold", vs: "−66%, at a measured cost of 2 errors", report: EVAL },
  { value: "99.6%", what: "of 17,140 records from 4 file formats placed on a district", vs: "the whole dataset is rebuilt and checked on every push", report: EVAL },
  { value: "25%", what: "of next-year watch-list flags were right in tests on 2019–2022", vs: "10% by guessing, 14% by size alone", report: "ml/reports/2026-10-31_bakeoff.md" },
  { value: "Same 92%", what: "accuracy from LangGraph as from the hand-built loop, with identical answers on the same 66 names", vs: "but 35 lines of control code instead of 14", report: "evals/reports/2026-11-01_frameworks.md" },
];

export const STEPS = [
  { title: "Extract", text: "Four NCRB file formats (2001–2014, 2015, 2016, 2017–2022) are read into one table. A reviewed crosswalk maps each era's columns onto the seven offence categories that exist in all of them." },
  { title: "Resolve", text: "Rules, fuzzy matching and the old district map place most old place names. Gemini places the rest inside guards: an allowed-list check, one repair call, and a confidence threshold above which no person needs to check." },
  { title: "Re-base", text: "Every split and rename is recorded as reviewable data. A new district's past is estimated from its share of its parent's crimes and marked as estimated, so all 22 years sit on today's map and no total changes." },
  { title: "Serve", text: "A batch job writes every district's numbers and summary. FastAPI serves them to programs; this site loads a static copy of the same answers, so it opens at once." },
];

export const STORIES = [
  {
    title: "The model is trusted only where it earned it",
    text: "Gemini's confidence was compared with how often it was actually right before any answer skipped review. Above the measured threshold, answers go straight in; below it, a person checks. That cut the names a person must check from 124 to 42, and the 2 errors it let through are written down.",
    link: EVAL,
  },
  {
    title: "A great score that was wrong",
    text: "The original model reported R² 0.88, because each district was scaled with its own future years and the test split by district. Trained only on the past and tested on the next year, it scored 0.06. Every later result uses that honest set-up.",
    link: "ml/reports/2026-09-13_era_a_baseline.md",
  },
  {
    title: "A hidden bias in the watch list",
    text: "Ranking districts by growth mostly picked tiny ones (1 case to 3 is +200%). Now each district is compared only with districts of similar size, in size groups chosen on earlier years, and the list's accuracy is reported per group.",
    link: "ml/reports/2026-10-31_bakeoff.md",
  },
  {
    title: "A popular framework, measured instead of assumed",
    text: "The same model step was rebuilt with LangGraph and run live on the same answer key: the same answers, but 35 lines of control code instead of 14, and 21 more packages. LangChain's connector also ignored the temperature setting, with only a console warning. The hand-built loop stays.",
    link: "evals/reports/2026-11-01_frameworks.md",
  },
];

export const LIMITS = [
  "Counts, not rates: without population figures the site cannot say where crime is highest per person.",
  "These are crimes reported to the police: a rise can mean more reporting as well as more crime.",
  "A district created after 2001 has estimated numbers for the years before it existed; every chart marks them.",
  "In 2013 the law widened what counts as rape and sexual harassment, so part of the rise after it is a change in what is counted.",
  "The watch list is experimental: in tests, about one flag in four was right.",
];
