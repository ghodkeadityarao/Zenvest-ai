"""
llm_profiler.py

Uses the Groq API (free tier) with Llama 3.1 8B to parse a user's
natural-language investment goal into a structured UserProfile.

How to get a free Groq API key (no credit card required):
  1. Go to https://console.groq.com
  2. Sign up for a free account
  3. Click "API Keys" → "Create API Key"
  4. Paste your key below in GROQ_API_KEY

Free tier limits (as of 2025): 14,400 requests/day, 6,000 tokens/min
That is more than enough for this project.
"""

import json
import re
from dataclasses import dataclass, asdict
import os

from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
#  PASTE YOUR FREE GROQ API KEY HERE
#  Get one at: https://console.groq.com (free, no credit card)
# ─────────────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = "llama-3.1-8b-instant"   # fast, free, high quality
# ─────────────────────────────────────────────────────────────────────────────

try:
    from groq import Groq as GroqClient
    cl = GroqClient(api_key=GROQ_API_KEY)
    # cl.chat.completions.create(
    #     model="llama-3.1-8b-instant",
    #     messages=[{"role": "user", "content": "test"}]
    # )
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

print(f"[llm_profiler] Groq client available: {GROQ_AVAILABLE}. ")

# ── UserProfile dataclass ────────────────────────────────────────────────────
@dataclass
class UserProfile:
    risk_level:         str    # "low" | "medium" | "high"
    horizon:            str    # "short" | "medium" | "long"
    distribution:       str    # "accumulating" | "distributing"
    max_ter:            float  # annual expense ratio, e.g. 0.002 = 0.20%
    preferred_sectors:  list   # e.g. ["Technology", "Health Care"]
    preferred_regions:  list   # e.g. ["United States", "Germany"]
    min_fund_size:      float  # in millions, e.g. 500.0
    currency:           str    # "EUR" | "USD" | "any"
    raw_goal:           str    # original user text
    is_intent:          bool   # True when the input expresses an investment intent


# ── System prompt sent to Llama ───────────────────────────────────────────────
_SYSTEM_PROMPT = """You are a financial profile extraction assistant.
Given a user's investment goal in natural language, return ONLY a valid JSON object
with exactly these keys — no markdown, no explanation, no extra text:

{
  "risk_level": "low" | "medium" | "high",
  "horizon": "short" | "medium" | "long",
  "distribution": "accumulating" | "distributing",
  "max_ter": <float: annual TER limit, e.g. 0.002 means 0.20%>,
  "preferred_sectors": [list from: Technology, Consumer Staples, Industrials,
    Consumer Discretionary, Financials, Basic Materials, Real Estate, Utilities,
    Energy, Health Care, Telecommunication, Other],
  "preferred_regions": [list from: United States, United Kingdom, Germany, Japan,
    France, China, Canada, Switzerland, Australia, Other],
  "min_fund_size": <float in millions, 0 if no preference>,
  "currency": "EUR" | "USD" | "any"
}

Inference rules:
- "safe", "conservative", "low risk", "stable", "protect capital" → risk_level: "low"
- "balanced", "moderate", "medium risk" → risk_level: "medium"
- "aggressive", "growth", "high return", "maximize" → risk_level: "high"
- "short term", "1-2 year", "near future" → horizon: "short"
- "medium term", "3-7 year" → horizon: "medium"
- "long term", "retirement", "decade", "10+ year" → horizon: "long"
- "no dividends", "reinvest", "tax efficient", "accumulate" → distribution: "accumulating"
- "income", "dividend", "cash flow", "payout", "yield" → distribution: "distributing"
- "low cost", "cheap", "low fees", "minimal cost" → max_ter: 0.002
- "very cheap", "under 0.1%" → max_ter: 0.001
- default max_ter: 0.005
- "large fund", "liquid", "established" → min_fund_size: 1000
- default min_fund_size: 100
- No sector mentioned → preferred_sectors: []
- No region mentioned → preferred_regions: []

If the input is unrelated to investing or is casual text (e.g. "Hello, how are you?"),
    return { "is_intent": false } along with the keys above set to sensible defaults.
    Otherwise set "is_intent": true.

Examples (required format):

1) Investment intent example
Input: "I want safe long-term growth with low fees and no dividends; prefer large European funds"
Output:
{
    "risk_level": "low",
    "horizon": "long",
    "distribution": "accumulating",
    "max_ter": 0.002,
    "preferred_sectors": [],
    "preferred_regions": ["Germany"],
    "min_fund_size": 1000.0,
    "currency": "EUR",
    "is_intent": true
}

2) Non-investment example (casual greeting)
Input: "Hello, how are you?"
Output:
{
    "risk_level": "medium",
    "horizon": "medium",
    "distribution": "accumulating",
    "max_ter": 0.005,
    "preferred_sectors": [],
    "preferred_regions": [],
    "min_fund_size": 100.0,
    "currency": "any",
    "is_intent": false
}
"""


# ── Groq API call ─────────────────────────────────────────────────────────────
def _call_groq(goal: str) -> dict | None:
    """Call Groq API with Llama 3.1. Returns parsed dict or None on failure."""
    if not GROQ_AVAILABLE:
        return None
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        return None

    try:
        client = GroqClient(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": f"Investment goal: {goal}"},
            ],
            temperature=0.05,
            max_tokens=512,
        )
        raw = response.choices[0].message.content.strip()
        # Strip accidental markdown fences
        raw = re.sub(r"```json|```", "", raw).strip()
        # Extract first JSON object found
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"[llm_profiler] Groq call failed: {e}. Using rule-based fallback.")
    return None


# ── Public API ────────────────────────────────────────────────────────────────
def extract_profile(goal: str) -> UserProfile:
    """
    Parse a natural-language investment goal into a UserProfile.
    Tries Groq/Llama 3.1 first; falls back to rule-based parser automatically.
    No API key needs to be passed — it is read from GROQ_API_KEY above.
    """
    # Require the Groq API client and a valid API key. If either is missing we
    # treat the service as down and do NOT fall back to the rule-based parser.
    if not (GROQ_AVAILABLE and GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here"):
        raise RuntimeError("Groq API unavailable or API key not set (server down)")

    profile_dict = _call_groq(goal)
    if profile_dict is None:
        # LLM call failed at runtime (network/error). Surface as server-down.
        raise RuntimeError("Groq API call failed (server down)")

    # If LLM returned a dict, prefer its explicit is_intent flag when present.
    # Be conservative: if the LLM did not include an explicit is_intent field,
    # treat the input as non-intent to avoid answering casual chat with finance outputs.
    intent = bool(profile_dict.get("is_intent", False))

    return UserProfile(
        risk_level         = str(profile_dict.get("risk_level",         "medium")),
        horizon            = str(profile_dict.get("horizon",            "medium")),
        distribution       = str(profile_dict.get("distribution",       "accumulating")),
        max_ter            = float(profile_dict.get("max_ter",          0.005)),
        preferred_sectors  = list(profile_dict.get("preferred_sectors", [])),
        preferred_regions  = list(profile_dict.get("preferred_regions", [])),
        min_fund_size      = float(profile_dict.get("min_fund_size",    100.0)),
        currency           = str(profile_dict.get("currency",           "any")),
        raw_goal           = goal,
        is_intent          = intent,
    )


def profile_to_dict(profile: UserProfile) -> dict:
    return asdict(profile)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_goals = [
        "Hello, How are you?",
        "I want safe long-term growth with low fees and no dividends",
        "I need income from dividends, moderate risk, prefer tech and healthcare",
        "Aggressive growth in US tech stocks, don't care about fees",
        "Conservative European equity fund, EUR currency, accumulating",
        "What is the weather like today?",
        "My name is John Doe and I like pizza.",
    ]
    for goal in test_goals:
        print(f"\nGoal  : {goal}")
        try:
            p = extract_profile(goal)
        except RuntimeError:
            print("Server is down — the LLM profile service is unavailable. Please try again later.")
            continue

        if not getattr(p, "is_intent", False):
            print(
                "I am an AI finance advisor and can help with finance-related questions."
                " I didn't detect an investment request in your input — please ask a finance-related question or rephrase your goal."
            )
        else:
            print(f"Risk  : {p.risk_level}  |  Horizon: {p.horizon}  |  "
                  f"Dist: {p.distribution}  |  TER: {p.max_ter:.3f}")
            print(f"Sectors: {p.preferred_sectors}")
            print(f"Regions: {p.preferred_regions}  |  Currency: {p.currency}")
