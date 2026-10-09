"""Published model prices, used to compute cost without spending. Figures are entered by hand with their date;
the report prints them next to every cost so a reader can check or update them.

USD per million tokens (input, output). Verify against https://ai.google.dev/gemini-api/docs/pricing before
quoting a number outside this repo.
"""
RATES_USD_PER_M = {
    "gemini-3.5-flash-lite": (0.30, 2.50),
}
RATES_AS_OF = "2026-10-10"
USD_INR = 88.0                 # set by hand on RATES_AS_OF; update with the rate on the day a report is quoted


def cost_usd(model: str, prompt_tokens: float, completion_tokens: float) -> float | None:
    rate = RATES_USD_PER_M.get(model)
    if rate is None:
        return None
    return (prompt_tokens * rate[0] + completion_tokens * rate[1]) / 1_000_000
