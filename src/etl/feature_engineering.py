import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Core features expected by the ML models
# ------------------------------------------------------------------
CORE_FEATURES = [
    "eco_pressure",
    "cost_efficiency",
    "durability_pressure",
    "innovation_level",
    "material_cost",
    "fragility_score",
    "max_packaging_cost",
    "durability_requirement",
    "sustainability_priority",
]

# ------------------------------------------------------------------
# Feature Engineering Function
# ------------------------------------------------------------------
def derive_features(
    df: pd.DataFrame,
    impute_strategy: str = "median",
    **kwargs
) -> pd.DataFrame:
    """
    Derive and preprocess features for packaging recommendation models.

    Parameters:
    - df: Input DataFrame containing raw product features
    - impute_strategy: Strategy to handle missing values
      ("mean", "median", "zero")

    Returns:
    - DataFrame with engineered and cleaned features
    """

    df = df.copy()

    # -------------------------------
    # eco_pressure
    # -------------------------------
    if "eco_pressure" not in df.columns:
        df["eco_pressure"] = 1 - df["sustainability_priority"]

    # -------------------------------
    # cost_efficiency
    # -------------------------------
    if "cost_efficiency" not in df.columns:
        df["cost_efficiency"] = 1 - (
            df["material_cost"] / df["max_packaging_cost"]
        )

    # -------------------------------
    # durability_pressure
    # -------------------------------
    if "durability_pressure" not in df.columns:
        df["durability_pressure"] = (
            df["fragility_score"] / df["durability_requirement"]
        )

    # -------------------------------
    # Replace inf / -inf with NaN
    # -------------------------------
    df[CORE_FEATURES] = df[CORE_FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )

    # -------------------------------
    # Skip imputation - let XGBoost handle missing values
    # -------------------------------
    pass

    # -------------------------------
    # Clamp normalized features to [0, 1]
    # -------------------------------
    for col in ["eco_pressure", "cost_efficiency", "durability_pressure"]:
        df[col] = df[col].clip(0, 1)

    return df
