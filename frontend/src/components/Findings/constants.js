export const FINDINGS = [
  {
    title: "Strong Future Prediction Capability",
    desc: "The model achieved an R² score close to 0.88 on future unseen districts, indicating high predictive reliability for real-world deployment.",
    icon: "📈",
    color: "bg-green-100 text-green-700",
  },
  {
    title: "District Holdout Validation",
    desc: "Instead of random splitting, entire districts were excluded during training to ensure the model learned crime patterns rather than memorizing geographic behavior.",
    icon: "🛡️",
    color: "bg-blue-100 text-blue-700",
  },
  {
    title: "Temporal Sliding Window Modeling",
    desc: "Historical crime trends were transformed into supervised learning sequences, allowing the model to capture temporal momentum in crime evolution.",
    icon: "⏳",
    color: "bg-purple-100 text-purple-700",
  },
  {
    title: "Feature Engineering Impact",
    desc: "Lag features such as prior-year crime change, rolling averages, and trend slopes were the strongest predictors of future crime escalation.",
    icon: "🧠",
    color: "bg-yellow-100 text-yellow-700",
  },
  {
    title: "Low Overfitting Observed",
    desc: "Training R² remained close to validation performance, demonstrating strong model generalization across districts.",
    icon: "🎯",
    color: "bg-red-100 text-red-700",
  },
];
