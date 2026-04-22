"""
collaborative_filter.py
Generates synthetic user-ETF interaction data and trains a collaborative
filtering model using matrix factorization (SVD via scipy).
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
import random
import warnings
warnings.filterwarnings("ignore")

RISK_PROFILES = {
    "low":    {"vol_max": 0.12, "ter_max": 0.003, "size_min": 500},
    "medium": {"vol_max": 0.22, "ter_max": 0.006, "size_min": 100},
    "high":   {"vol_max": 0.50, "ter_max": 0.010, "size_min":   0},
}
HORIZON_RETURN_COL = {
    "short":  "yearReturnCUR",
    "medium": "threeYearReturnCUR",
    "long":   "yearReturnCUR",
}


def generate_synthetic_interactions(
    df: pd.DataFrame,
    n_users: int = 500,
    interactions_per_user: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate a user-ETF interaction matrix.
    Each synthetic user has a random risk profile and rates ETFs 1-5
    based on how well the ETF matches their simulated preferences.
    Returns DataFrame with columns: [user_id, isin, rating].
    """
    random.seed(seed)
    np.random.seed(seed)

    risk_levels   = ["low", "medium", "high"]
    horizons      = ["short", "medium", "long"]
    distributions = ["accumulating", "distributing"]
    records       = []

    for uid in range(n_users):
        risk      = random.choice(risk_levels)
        horizon   = random.choice(horizons)
        dist_pref = random.choice(distributions)
        prefs     = RISK_PROFILES[risk]
        ret_col   = HORIZON_RETURN_COL[horizon]

        candidates = df[
            (df["yearVolatilityCUR"] <= prefs["vol_max"]) &
            (df["ter"]               <= prefs["ter_max"]) &
            (df["fundSizeMillions"]  >= prefs["size_min"])
        ].copy()

        if len(candidates) < 5:
            candidates = df.copy()

        scaler = MinMaxScaler()
        score  = np.zeros(len(candidates))

        if ret_col in candidates.columns:
            returns = candidates[ret_col].fillna(0).values.reshape(-1, 1)
            score  += scaler.fit_transform(returns).flatten() * 0.4

        vol    = candidates["yearVolatilityCUR"].fillna(0.5).values.reshape(-1, 1)
        score += (1 - scaler.fit_transform(vol).flatten()) * 0.3

        ter    = candidates["ter"].fillna(0.01).values.reshape(-1, 1)
        score += (1 - scaler.fit_transform(ter).flatten()) * 0.2

        dist_match = (
            candidates["distributionPolicy"].str.lower() == dist_pref
        ).astype(float).values
        score += dist_match * 0.1

        score += np.random.normal(0, 0.05, len(score))
        score  = np.clip(score, 0, 1)

        probs     = score / (score.sum() + 1e-9)
        n_sample  = min(interactions_per_user, len(candidates))
        chosen_ix = np.random.choice(len(candidates), size=n_sample,
                                     replace=False, p=probs)
        chosen    = candidates.iloc[chosen_ix]

        for i, (_, row) in enumerate(chosen.iterrows()):
            etf_score = score[candidates.index.get_loc(row.name)]
            rating    = max(1, min(5, round(1 + etf_score * 4)))
            records.append({"user_id": uid, "isin": row["isin"], "rating": rating})

    interactions = pd.DataFrame(records)
    print(f"[collaborative_filter] Generated {len(interactions)} interactions "
          f"from {n_users} synthetic users across "
          f"{interactions['isin'].nunique()} ETFs.")
    return interactions

class SVDRecommender:
    """
    Lightweight SVD-based collaborative filter using scipy.
    """
    def __init__(self, n_factors: int = 50):
        self.n_factors    = n_factors
        self.item_factors = None
        self.isin_list    = None
        self.isin_to_idx  = None

    def fit(self, interactions: pd.DataFrame):
        pivot = interactions.pivot_table(
            index="user_id", columns="isin", values="rating", fill_value=0
        )
        self.isin_list    = list(pivot.columns)
        self.isin_to_idx  = {isin: i for i, isin in enumerate(self.isin_list)}
        matrix            = csr_matrix(pivot.values, dtype=np.float32)
        k                 = min(self.n_factors, min(matrix.shape) - 1)
        _, _, Vt          = svds(matrix, k=k)
        self.item_factors = Vt.T          # (n_items, k)
        print(f"[collaborative_filter] SVD: {len(self.isin_list)} ETFs, "
              f"{k} latent factors.")

    def score_all(self) -> pd.Series:
        norms        = np.linalg.norm(self.item_factors, axis=1)
        scores_norm  = MinMaxScaler().fit_transform(
            norms.reshape(-1, 1)
        ).flatten()
        return pd.Series(scores_norm, index=self.isin_list, name="cf_score")


def train_cf_model(df: pd.DataFrame) -> tuple[SVDRecommender, pd.Series]:
    """Full pipeline: generate synthetic data → train SVD → return scores."""
    interactions = generate_synthetic_interactions(df)
    model        = SVDRecommender(n_factors=50)
    model.fit(interactions)
    cf_scores    = model.score_all()
    return model, cf_scores

if __name__ == "__main__":
    from data_pipeline import load_and_clean
    df = load_and_clean("justetf_2.csv")    
    model, cf_scores = train_cf_model(df)
    print(f"\nTop 5 ETFs by CF score:")
    for isin, score in cf_scores.nlargest(5).items():
        name = df.loc[df["isin"] == isin, "name"].values
        print(f"  {score:.4f}  {name[0] if len(name) else isin}")
