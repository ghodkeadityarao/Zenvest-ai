# Hybrid LLM-Driven Financial Recommendation System
**CSCE 670 — Texas A&M University**
Debmalya Chatterjee · Aditya Rao Ghodke · Durgesh Bhirud

---

## Overview

An end-to-end ETF recommendation system that:
1. **Understands** investment goals in natural language using **Llama 3.1 8B via Groq** (free)
2. **Recommends** ETFs by combining content-based filtering + collaborative filtering via **XGBoost**
3. **Explains** every recommendation using **SHAP** feature attributions in plain English

**Dataset:** `justetf_2.csv` — 2,264 ETFs × 129 features (returns, volatility, TER, sector/country exposures)

---

## File Structure

```
financial_recommender/
├── app.py                  ← Streamlit web app
├── data_pipeline.py        ← Data loading, cleaning, feature engineering
├── llm_profiler.py         ← Groq/Llama 3.1 profile extractor + rule-based fallback
├── collaborative_filter.py ← Synthetic user generator + SVD collaborative filter
├── recommender.py          ← Content scoring, XGBoost ranker, SHAP explanations
├── train_and_evaluate.py   ← Training pipeline + NDCG/Precision@K evaluation
├── requirements.txt        ← Python dependencies
├── justetf_2.csv           ← Dataset (place here)
└── README.md
```

---

## Setup

### 1. Get a free Groq API key (no credit card required)
1. Go to **https://console.groq.com**
2. Sign up (free) → click **API Keys** → **Create API Key**
3. Open `llm_profiler.py` and paste your key:
   ```python
   GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxx"
   ```
   > **Without a key:** the system still works using the built-in rule-based parser.
   > No functionality is lost — the rule-based parser covers all common investment vocabulary.

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Place the dataset
Copy `justetf_2.csv` into the `financial_recommender/` folder.

### 5. Pre-train the model (recommended)
```bash
python train_and_evaluate.py
```
This trains the XGBoost ranker and saves `model_cache.pkl`.
The Streamlit app will auto-train on first launch if the cache is missing.

### 6. Launch the app
```bash
streamlit run app.py
```
Open: **http://localhost:8501**

---

## How to Use

Type your investment goal in plain English, then click **Get Recommendations**:

| Example Goal | Profile Extracted |
|---|---|
| "Safe long-term growth, low fees, no dividends" | low risk · long · accumulating · TER < 0.20% |
| "Dividend income, moderate risk, prefer tech" | medium risk · distributing · Technology sector |
| "Aggressive US tech growth, 5-year horizon" | high risk · medium · Technology · United States |
| "Conservative EUR fund, low volatility" | low risk · accumulating · EUR currency |

---

## LLM: Groq + Llama 3.1 8B

- **Provider:** [Groq](https://groq.com) — free tier, no credit card
- **Model:** `llama-3.1-8b-instant` — fast and highly capable
- **Free limits:** 14,400 requests/day, 6,000 tokens/min
- **Fallback:** built-in rule-based parser runs automatically if key is missing or invalid

The API key is set once in `llm_profiler.py`. There is no API key input in the UI.

---

## Architecture

```
User goal (text)
      │
      ▼
┌──────────────────────┐
│   Groq / Llama 3.1   │  → UserProfile {risk, horizon, distribution,
│   llm_profiler.py    │               ter, sectors, regions, currency}
└────────┬─────────────┘
         │
    ┌────┴────┐
    ▼         ▼
Content-   Collaborative
Based      Filtering
Scorer     (SVD · synthetic
           interaction matrix)
    │         │
    └────┬────┘
         ▼
    XGBoost ranker
    (hybrid score)
         │
         ▼
    SHAP explainer
    (feature attribution)
         │
         ▼
    Streamlit UI
    (cards + waterfall charts)
```

---

## Dataset Columns Used

| Feature | Column | Purpose |
|---|---|---|
| Expense ratio | `ter` | Cost penalty |
| 1-yr return | `yearReturnCUR` | Performance |
| 3-yr return | `threeYearReturnCUR` | Medium-horizon signal |
| Volatility | `yearVolatilityCUR` | Risk matching |
| Distribution | `distributionPolicy` | Tax preference |
| Fund size | `fundSizeMillions` | Liquidity |
| Sector exposure | `exposureSector_*` | Sector preference |
| Country exposure | `exposureCountry_*` | Region preference |
| Currency | `fundCurrency` | Currency preference |

---

## Evaluation

- **NDCG@10** — ranking quality
- **Precision@5** — top-5 overlap with ground truth

Three models compared: Content-only · CF-only · Hybrid (XGBoost)
