import numpy as np
import pandas as pd


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


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- eco_pressure ---
    if "eco_pressure" not in df.columns:
        df["eco_pressure"] = 1 - df["sustainability_priority"]

    # --- cost_efficiency ---
    if "cost_efficiency" not in df.columns:
        df["cost_efficiency"] = 1 - (
            df["material_cost"] / df["max_packaging_cost"]
        )

    # --- durability_pressure ---
    if "durability_pressure" not in df.columns:
        df["durability_pressure"] = (
            df["fragility_score"] / df["durability_requirement"]
        )

    # Replace inf / NaN
    df[CORE_FEATURES] = df[CORE_FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )

    df[CORE_FEATURES] = df[CORE_FEATURES].apply(
        lambda c: c.fillna(c.median())
    )

    # Clamp normalized features
    for col in ["eco_pressure", "cost_efficiency", "durability_pressure"]:
        df[col] = df[col].clip(0, 1)

    return df
