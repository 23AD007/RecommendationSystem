import pandas as pd
from src.etl.feature_engineering import CORE_FEATURES
from src.models.train_xgboost_ranker import predict_ranked_materials

# Alias for API compatibility
rank_materials = predict_ranked_materials


def rank_with_model(model, df):
    """
    Rank materials using trained XGBoost model.
    Fails explicitly if features are missing.
    """
    df = df.copy()

    missing = [c for c in CORE_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features for model inference: {missing}")

    X = df[CORE_FEATURES].copy()
    df["score"] = model.predict(X)

    return df.sort_values("score", ascending=False)


def rank_rule_based(df):
    """
    Feature-driven fallback model.
    Uses ONLY features guaranteed from UI + feature_engineering.
    """
    df = df.copy()

    required = [
        "sustainability_priority",
        "material_cost",
        "max_packaging_cost",
        "fragility_score",
        "durability_requirement",
        "innovation_level"
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features for fallback ranking: {missing}")

    cost_ratio = df["material_cost"] / (df["max_packaging_cost"] + 1e-6)

    df["score"] = (
        0.35 * df["sustainability_priority"]
        + 0.25 * (1 - cost_ratio)
        + 0.20 * (1 - df["fragility_score"])
        + 0.15 * df["durability_requirement"]
        + 0.05 * (df["innovation_level"] / 5.0)
    )

    return df.sort_values("score", ascending=False)
