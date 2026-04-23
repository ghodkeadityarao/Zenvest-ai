"""
recommender.py
Content-based scoring, XGBoost hybrid ranker, and SHAP-based explanations.
Falls back gracefully to a linear ranker if XGBoost is not installed.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    import shap
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

print(f"[recommender] XGBoost available: {XGB_AVAILABLE}")

from llm_profiler import UserProfile
from data_pipeline import SECTOR_COLS, COUNTRY_COLS

# ── Risk → target volatility map ─────────────────────────────────────────────
RISK_VOL = {"low": 0.10, "medium": 0.20, "high": 0.50}

# ── Human-readable feature labels for SHAP display ───────────────────────────
FEATURE_LABELS = {
    "content_score":                        "Overall profile match",
    "cf_score":                             "Similar investor preference",
    "ter":                                  "Expense ratio (TER)",
    "ter_excess":                           "TER above your limit",
    "vol_excess":                           "Volatility above risk limit",
    "volatility":                           "Fund volatility",
    "year_return":                          "1-year return",
    "three_yr_return":                      "3-year return",
    "log_size":                             "Fund size (liquidity)",
    "return_per_risk":                      "Return-to-risk ratio",
    "is_accumulating":                      "Accumulating policy",
    "dist_match":                           "Distribution policy match",
    "size_ok":                              "Fund size meets minimum",
    "exposureSector_Technology":            "Technology sector",
    "exposureSector_Health_Care":           "Health Care sector",
    "exposureSector_Financials":            "Financials sector",
    "exposureSector_Energy":                "Energy sector",
    "exposureSector_Consumer_Discretionary":"Consumer Discretionary",
    "exposureSector_Industrials":           "Industrials sector",
    "exposureSector_Real_Estate":           "Real Estate sector",
    "exposureSector_Utilities":             "Utilities sector",
    "exposureSector_Basic_Materials":       "Basic Materials sector",
    "exposureSector_Telecommunication":     "Telecommunication sector",
    "exposureSector_Consumer_Staples":      "Consumer Staples sector",
    "exposureSector_Other":                 "Other sectors",
}


# ── Content-based scorer ──────────────────────────────────────────────────────
def compute_content_scores(df: pd.DataFrame, profile: UserProfile) -> pd.Series:
    """
    Score each ETF against the user profile using weighted feature matching.
    Returns normalised Series [0, 1] indexed by df.index.
    """
    scores     = np.zeros(len(df))
    target_vol = RISK_VOL[profile.risk_level]

    # 1. Volatility fit (25%)
    vol       = df["yearVolatilityCUR"].fillna(0.5).values
    scores   += np.exp(-2 * np.maximum(vol - target_vol, 0)) * 0.25

    # 2. Return for horizon (20%)
    ret_col   = {"short": "yearReturnCUR", "medium": "threeYearReturnCUR",
                 "long": "yearReturnCUR"}[profile.horizon]
    returns   = df[ret_col].fillna(df[ret_col].median()).values
    scores   += MinMaxScaler().fit_transform(returns.reshape(-1, 1)).flatten() * 0.20

    # 3. TER fit (20%)
    ter       = df["ter"].fillna(0.01).values
    scores   += np.exp(-300 * np.maximum(ter - profile.max_ter, 0)) * 0.20

    # 4. Distribution policy match (15%)
    dist_match = (
        df["distributionPolicy"].str.lower() == profile.distribution
    ).astype(float).values
    scores   += dist_match * 0.15

    # 5. Preferred sectors (10%)
    if profile.preferred_sectors:
        sec_score = np.zeros(len(df))
        for sector in profile.preferred_sectors:
            col = f"exposureSector_{sector}"
            if col in df.columns:
                sec_score += df[col].fillna(0).values
        scores += np.clip(sec_score, 0, 1) * 0.10
    else:
        scores += 0.05

    # 6. Preferred regions (5%)
    if profile.preferred_regions:
        reg_score = np.zeros(len(df))
        for region in profile.preferred_regions:
            col = f"exposureCountry_{region}"
            if col in df.columns:
                reg_score += df[col].fillna(0).values
        scores += np.clip(reg_score, 0, 1) * 0.05

    # 7. Fund size (5%)
    size      = df["fundSizeMillions"].fillna(100).values
    scores   += (size >= profile.min_fund_size).astype(float) * 0.05

    return pd.Series(
        MinMaxScaler().fit_transform(scores.reshape(-1, 1)).flatten(),
        index=df.index,
        name="content_score",
    )


# ── Feature matrix for XGBoost ────────────────────────────────────────────────
def build_ranker_features(
    df: pd.DataFrame,
    content_scores: pd.Series,
    cf_scores: pd.Series,
    profile: UserProfile,
) -> pd.DataFrame:
    target_vol = RISK_VOL[profile.risk_level]
    feat       = pd.DataFrame(index=df.index)

    feat["content_score"]   = content_scores
    feat["cf_score"]        = df["isin"].map(cf_scores).fillna(0.0)
    feat["ter"]             = df["ter"].fillna(df["ter"].median())
    feat["year_return"]     = df["yearReturnCUR"].fillna(0)
    feat["three_yr_return"] = df["threeYearReturnCUR"].fillna(0)
    feat["volatility"]      = df["yearVolatilityCUR"].fillna(0.5)
    feat["log_size"]        = np.log1p(df["fundSizeMillions"].fillna(100))
    feat["return_per_risk"] = df["yearReturnPerRiskCUR"].fillna(0)
    feat["is_accumulating"] = (df["distributionPolicy"] == "Accumulating").astype(int)
    feat["is_physical"]     = df["replicationMethod"].apply(
        lambda x: 1 if "replication" in str(x).lower() else 0
    )
    feat["vol_excess"]      = np.maximum(feat["volatility"] - target_vol, 0)
    feat["ter_excess"]      = np.maximum(feat["ter"] - profile.max_ter, 0)
    feat["dist_match"]      = (
        df["distributionPolicy"].str.lower() == profile.distribution
    ).astype(int)
    feat["size_ok"]         = (
        df["fundSizeMillions"].fillna(0) >= profile.min_fund_size
    ).astype(int)

    for col in SECTOR_COLS:
        if col in df.columns:
            feat[col.replace(" ", "_")] = df[col].fillna(0)

    return feat.fillna(0)


def generate_pseudo_labels(
    content_scores: pd.Series, cf_scores_mapped: pd.Series
) -> pd.Series:
    combined = 0.6 * content_scores + 0.4 * cf_scores_mapped
    return pd.Series(
        MinMaxScaler().fit_transform(combined.values.reshape(-1, 1)).flatten(),
        index=content_scores.index,
    )


# ── Hybrid XGBoost ranker ─────────────────────────────────────────────────────
class HybridRanker:
    def __init__(self):
        self.model         = None
        self.feature_names = None
        self.explainer     = None

    def train(self, features: pd.DataFrame, labels: pd.Series):
        if not XGB_AVAILABLE:
            raise RuntimeError("XGBoost is required but not installed. Run: pip install xgboost")
        self.feature_names = list(features.columns)
        self.model = xgb.XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0,
        )
        self.model.fit(features, labels)
        self.explainer = shap.TreeExplainer(self.model)
        print("[recommender] XGBoost ranker trained.")

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("HybridRanker has not been trained yet.")
        return self.model.predict(features)

    def explain(self, features: pd.DataFrame, idx: int) -> dict:
        if self.explainer is None:
            raise RuntimeError("HybridRanker has not been trained yet.")
        row       = features.iloc[[idx]]
        shap_vals = self.explainer.shap_values(row)[0]
        return dict(zip(self.feature_names, shap_vals))


# ── Plain-English explanation ─────────────────────────────────────────────────
def generate_explanation(etf_row, shap_dict: dict, rank: int) -> str:
    name    = etf_row.get("name", "This ETF")
    ter_pct = etf_row.get("ter", 0) * 100
    vol_pct = etf_row.get("yearVolatilityCUR", 0) * 100
    ret_pct = etf_row.get("yearReturnCUR", 0) * 100
    dist    = etf_row.get("distributionPolicy", "N/A")
    size    = etf_row.get("fundSizeMillions", 0)

    sorted_shap  = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    top_pos  = [(k, v) for k, v in sorted_shap if v > 0][:3]
    top_neg  = [(k, v) for k, v in sorted_shap if v < 0][:2]

    reasons  = [
        f"{FEATURE_LABELS.get(k, k.replace('_',' ').title())} (+{abs(v):.3f})"
        for k, v in top_pos
    ]
    concerns = [
        f"{FEATURE_LABELS.get(k, k.replace('_',' ').title())} (−{abs(v):.3f})"
        for k, v in top_neg
    ]

    lines = [
        f"**Rank #{rank} — {name}**\n",
        f"Recommended because: {'; '.join(reasons) if reasons else 'strong overall profile match'}.\n",
        f"**TER** {ter_pct:.2f}% · **1-yr return** {ret_pct:+.1f}% · "
        f"**Volatility** {vol_pct:.1f}% · {dist} · "
        f"**Size** €{size:,.0f}M",
    ]
    if concerns:
        lines.append(f"\nMinor trade-offs: {'; '.join(concerns)}.")
    return "\n".join(lines)


# ── Full recommendation pipeline ──────────────────────────────────────────────
def recommend(
    df: pd.DataFrame,
    profile: UserProfile,
    cf_scores: pd.Series,
    ranker: HybridRanker,
    top_n: int = 5,
) -> list[dict]:
    content_scores = compute_content_scores(df, profile)
    features       = build_ranker_features(df, content_scores, cf_scores, profile)
    raw_scores     = ranker.predict(features)

    work_df           = df.copy()
    work_df["_score"] = raw_scores
    top_df            = work_df.nlargest(top_n, "_score").reset_index(drop=False)

    results = []
    for rank, (_, row) in enumerate(top_df.iterrows(), start=1):
        orig_idx = row.get("index", _)
        try:
            feat_idx   = features.index.get_loc(orig_idx)
            shap_dict  = ranker.explain(features, feat_idx)
        except Exception:
            shap_dict  = {"content_score": float(content_scores.iloc[0])}

        shap_display = {
            FEATURE_LABELS.get(k, k): round(float(v), 4)
            for k, v in sorted(shap_dict.items(),
                                key=lambda x: abs(x[1]), reverse=True)[:6]
        }

        results.append({
            "rank":          rank,
            "isin":          row.get("isin", ""),
            "name":          row.get("name", ""),
            "provider":      row.get("fundProvider", ""),
            "ter":           row.get("ter", 0),
            "year_return":   row.get("yearReturnCUR", 0),
            "volatility":    row.get("yearVolatilityCUR", 0),
            "distribution":  row.get("distributionPolicy", ""),
            "fund_size":     row.get("fundSizeMillions", 0),
            "currency":      row.get("fundCurrency", ""),
            "labels":        row.get("labels", ""),
            "final_score":   float(row["_score"]),
            "content_score": float(content_scores.loc[orig_idx])
                             if orig_idx in content_scores.index else 0.0,
            "cf_score":      float(cf_scores.get(row.get("isin", ""), 0.0)),
            "explanation":   generate_explanation(row, shap_dict, rank),
            "shap_values":   shap_display,
        })

    return results


if __name__ == "__main__":
    from data_pipeline import load_and_clean
    from llm_profiler import extract_profile
    from collaborative_filter import train_cf_model

    df = load_and_clean("justetf_2.csv")
    _, cf_scores = train_cf_model(df)
    profile      = extract_profile(
        "I want safe long-term growth, low fees, no dividends, and tax-efficient ETFs."
    )
    content_sc   = compute_content_scores(df, profile)
    cf_mapped    = pd.Series(df["isin"].map(cf_scores).fillna(0).values, index=df.index)
    features     = build_ranker_features(df, content_sc, cf_scores, profile)
    labels       = generate_pseudo_labels(content_sc, cf_mapped)
    ranker       = HybridRanker()
    ranker.train(features, labels)
    results      = recommend(df, profile, cf_scores, ranker, top_n=5)

    for r in results:
        print(f"Rank {r['rank']}: {r['name']}")
        print(f"  Score {r['final_score']:.4f} | TER {r['ter']*100:.2f}% "
              f"| Vol {r['volatility']*100:.1f}%")