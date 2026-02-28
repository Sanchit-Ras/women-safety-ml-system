export const DECISIONS = [
  {
    title: "District Holdout Validation",
    desc: "Instead of random splitting, the model was tested on completely unseen districts to simulate real-world deployment.",
    icon: "🏙️",
    bg: "bg-blue-50 border-blue-100",
  },
  {
    title: "Sliding Window Forecasting",
    desc: "Converted time-series crime data into supervised learning format using lag features.",
    icon: "📅",
    bg: "bg-purple-50 border-purple-100",
  },
  {
    title: "Leakage Prevention",
    desc: "Applied strict time-based splits to ensure the model never trained on future data.",
    icon: "🔒",
    bg: "bg-green-50 border-green-100",
  },
  {
    title: "Dual Model Strategy",
    desc: "Used Random Forest for robustness and XGBoost for high predictive power.",
    icon: "🤖",
    bg: "bg-orange-50 border-orange-100",
  },
];
