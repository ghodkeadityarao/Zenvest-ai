"""
train_and_evaluate.py
Trains the hybrid ranker and evaluates it against baselines.
Run once before launching the Streamlit app:
    python train_and_evaluate.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
import warnings
warnings.filterwarnings("ignore")

from data_pipeline import load_and_clean
from collaborative_filter import train_cf_model
from recommender import (
    compute_content_scores,
    build_ranker_features,
    generate_pseudo_labels,
    HybridRanker,
)
from llm_profiler import extract_profile

TEST_GOALS = [
    "I want safe long-term growth with low fees and no dividends",
    "I need dividend income, moderate risk, prefer technology and healthcare",
    "Aggressive high-growth US tech exposure, short horizon",
    "Conservative European equities, accumulating, EUR currency, low cost",
    "Balanced global portfolio, 10-year horizon, low cost, medium risk",
    "How are you and where are you located?",  # non-intent input to test robustness
]


def precision_at_k(true_scores, pred_scores, k=5):
    top_true = set(np.argsort(true_scores)[::-1][:k])
    top_pred = set(np.argsort(pred_scores)[::-1][:k])
    return len(top_true & top_pred) / k


def evaluate_models(df, cf_scores, ranker, test_goals):
    rows = []
    for goal in test_goals:
        profile       = extract_profile(goal)
        content_sc    = compute_content_scores(df, profile)
        features      = build_ranker_features(df, content_sc, cf_scores, profile)
        cf_mapped     = df["isin"].map(cf_scores).fillna(0).values
        content_arr   = content_sc.values
        hybrid_arr    = ranker.predict(features)
        true_scores   = content_arr   # content = ground truth proxy

        rows.append({
            "goal":          goal[:55] + "…",
            "ndcg_content":  round(ndcg_score([true_scores], [content_arr], k=10), 4),
            "ndcg_cf":       round(ndcg_score([true_scores], [cf_mapped],   k=10), 4),
            "ndcg_hybrid":   round(ndcg_score([true_scores], [hybrid_arr],  k=10), 4),
            "prec5_content": round(precision_at_k(true_scores, content_arr, 5), 4),
            "prec5_cf":      round(precision_at_k(true_scores, cf_mapped,   5), 4),
            "prec5_hybrid":  round(precision_at_k(true_scores, hybrid_arr,  5), 4),
        })
    return pd.DataFrame(rows)


def train_and_save(data_path="justetf_2.csv", model_path="model_cache.pkl"):
    print("=" * 60)
    print("STEP 1/4  Loading and cleaning data…")
    df = load_and_clean(data_path)

    print("\nSTEP 2/4  Training collaborative filter (SVD)…")
    cf_model, cf_scores = train_cf_model(df)

    print("\nSTEP 3/4  Training hybrid XGBoost ranker…")
    train_profile = extract_profile(
        "Balanced global ETF, moderate risk, accumulating, low cost, long-term"
    )
    content_sc = compute_content_scores(df, train_profile)
    features   = build_ranker_features(df, content_sc, cf_scores, train_profile)
    cf_mapped  = pd.Series(
        df["isin"].map(cf_scores).fillna(0).values, index=df.index
    )
    labels     = generate_pseudo_labels(content_sc, cf_mapped)
    ranker     = HybridRanker()
    ranker.train(features, labels)

    print("\nSTEP 4/4  Evaluating models…")
    eval_df = evaluate_models(df, cf_scores, ranker, TEST_GOALS)

    print("\n── NDCG@10 ─────────────────────────────────────────────")
    print(eval_df[["goal", "ndcg_content", "ndcg_cf", "ndcg_hybrid"]]
          .to_string(index=False))
    print("\n── Precision@5 ─────────────────────────────────────────")
    print(eval_df[["goal", "prec5_content", "prec5_cf", "prec5_hybrid"]]
          .to_string(index=False))

    means = eval_df[["ndcg_content", "ndcg_cf", "ndcg_hybrid"]].mean()
    print(f"\nMean NDCG@10 — Content: {means['ndcg_content']:.4f} | "
          f"CF: {means['ndcg_cf']:.4f} | "
          f"Hybrid: {means['ndcg_hybrid']:.4f}")

    cache = {"df": df, "cf_scores": cf_scores,
             "ranker": ranker, "eval_df": eval_df}
    with open(model_path, "wb") as f:
        pickle.dump(cache, f)
    print(f"\nModel cached to {model_path}")
    print("=" * 60)
    return cache


if __name__ == "__main__":
    import sys
    data_path  = sys.argv[1] if len(sys.argv) > 1 else "justetf_2.csv"
    model_path = sys.argv[2] if len(sys.argv) > 2 else "model_cache.pkl"
    train_and_save(data_path, model_path)
