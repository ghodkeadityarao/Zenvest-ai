import os
import pickle
import warnings
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
from flask import Flask, jsonify, render_template, request

warnings.filterwarnings("ignore")

# Reuse the existing backend logic from the uploaded project
from llm_profiler import extract_profile
from recommender import recommend

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = "./"
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "justetf_2.csv")
DEFAULT_CACHE_PATH = os.path.join(PROJECT_ROOT, "model_cache.pkl")

app = Flask(__name__, template_folder=os.path.join(APP_ROOT, "templates"))


def fmt_pct(v: Any, digits: int = 1, sign: bool = True) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v * 100:+.{digits}f}%" if sign else f"{v * 100:.{digits}f}%"


def fmt_ter(v: Any) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v * 100:.2f}%"


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


_system_cache: dict[str, Any] | None = None


def load_system(data_path: str = DEFAULT_DATA_PATH, cache_path: str = DEFAULT_CACHE_PATH) -> dict[str, Any]:
    global _system_cache
    if _system_cache is not None:
        return _system_cache

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                _system_cache = pickle.load(f)
                return _system_cache
        except Exception as exc:
            print(f"[load_system] Cache load failed: {exc}. Rebuilding cache...")
            try:
                os.remove(cache_path)
            except OSError:
                pass

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            "justetf_2.csv was not found. Put the dataset in /mnt/data or update DEFAULT_DATA_PATH."
        )

    from train_and_evaluate import train_and_save

    _system_cache = train_and_save(data_path, cache_path)
    return _system_cache


def build_chat_reply(user_msg: str, profile, results: list[dict], history: list[dict]) -> str:
    try:
        from groq import Groq
        from llm_profiler import GROQ_API_KEY, GROQ_MODEL

        if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
            raise RuntimeError("Groq API key not configured")

        context = ""
        if profile:
            context += (
                f"User profile — risk: {profile.risk_level}, horizon: {profile.horizon}, "
                f"distribution: {profile.distribution}, max TER: {profile.max_ter:.3f}, "
                f"sectors: {profile.preferred_sectors}, regions: {profile.preferred_regions}.\n"
            )
        if results:
            context += "Recommended ETFs:\n"
            for r in results[:3]:
                context += (
                    f"  #{r['rank']} {r['name']}: TER {fmt_ter(r['ter'])}, "
                    f"1yr {fmt_pct(r['year_return'])}, vol {fmt_pct(r['volatility'], sign=False)}, "
                    f"{r['distribution']}, score {r['final_score']:.3f}\n"
                )

        system = (
            "You are a concise, expert AI financial advisor specializing in ETFs, any finance topics and portfolio management. "
            "Answer only finance-related questions in under 100 words. "
            "If asked something unrelated, politely redirect to finance topics. "
            "Ground your answer in the supplied portfolio context when relevant.\n"
            + context
        )

        messages = [{"role": "system", "content": system}]
        for m in history[-8:]:
            role = m.get("role", "assistant")
            if role not in {"user", "assistant", "system"}:
                role = "assistant"
            messages.append({"role": role, "content": str(m.get("text", ""))})
        messages.append({"role": "user", "content": user_msg})

        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.35,
            max_tokens=220,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        return f"Sorry, I’m having trouble connecting right now. ({type(exc).__name__})"


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/api/recommend")
def api_recommend():
    payload = request.get_json(silent=True) or {}
    goal = str(payload.get("goal", "")).strip()

    if not goal:
        return jsonify({"error": "Please enter an investment goal first."}), 400

    try:
        cache = load_system()
        df = cache["df"]
        cf_scores = cache["cf_scores"]
        ranker = cache["ranker"]
    except Exception as exc:
        return jsonify({"error": f"Failed to load recommendation engine: {exc}"}), 500

    try:
        profile = extract_profile(goal)
    except RuntimeError as exc:
        return jsonify({"error": f"LLM service unavailable: {exc}"}), 503
    except Exception as exc:
        return jsonify({"error": f"Profile extraction failed: {exc}"}), 500

    if not getattr(profile, "is_intent", False):
        return jsonify(
            {
                "intent": False,
                "message": "I'm an AI finance advisor. I didn't detect an investment request — please describe your investment goal.",
                "profile": _json_safe(profile),
                "results": [],
            }
        )

    try:
        results = recommend(df, profile, cf_scores, ranker, top_n=5)
    except Exception as exc:
        return jsonify({"error": f"ETF ranking failed: {exc}"}), 500

    return jsonify(
        {
            "intent": True,
            "profile": _json_safe(profile),
            "results": _json_safe(results),
        }
    )

def is_finance_chat_query(user_msg: str) -> bool:
    """
    Use the LLM to classify whether a chat message is finance-related broadly,
    not just an investment-goal request.
    """
    try:
        from groq import Groq
        from llm_profiler import GROQ_API_KEY, GROQ_MODEL

        if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
            raise RuntimeError("Groq API key not configured")

        system = """
You are a strict classifier for a financial advisor chatbot.

Decide whether the user's message is related to finance, investing, taxes, money,
ETFs, stocks, bonds, portfolio management, savings, budgeting, inflation, interest,
or personal/business financial concepts.

Return ONLY one word:
YES -> if the message is finance-related
NO  -> if the message is unrelated to finance

Examples:
User: "What is tax?"
YES

User: "Explain inflation simply."
YES

User: "Why was this ETF ranked highly for me?"
YES

User: "How are you?"
NO

User: "What is the weather today?"
NO
""".strip()

        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            max_tokens=5,
        )

        answer = resp.choices[0].message.content.strip().upper()
        return answer.startswith("YES")

    except Exception:
        # Fail open so finance questions are still answered rather than blocked
        return True
    

@app.post("/api/chat")
def api_chat():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    history = payload.get("history", []) or []
    profile_payload = payload.get("profile")
    results = payload.get("results", []) or []

    if not message:
        return jsonify({"reply": "Please enter a question."}), 400

    if not is_finance_chat_query(message):
        return jsonify({
            "reply": "Hi! I'm your AI financial advisor. Ask me anything about your portfolio or ETFs."
        })

    profile = None
    if isinstance(profile_payload, dict):
        try:
            from llm_profiler import UserProfile
            profile = UserProfile(**profile_payload)
        except Exception:
            profile = None

    reply = build_chat_reply(message, profile, results, history)
    return jsonify({"reply": reply})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
