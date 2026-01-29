import pandas as pd
import numpy as np

CATEGORY_MAP = {
    "electronics": 0,
    "food": 1,
    "pharma": 2,
    "fashion": 3,
    "automotive": 4,
}

def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Encode category
    df["product_category_encoded"] = (
        df["product_category"]
        .map(CATEGORY_MAP)
        .fillna(-1)
        .astype(int)
    )

    # Numeric safety
    numeric_cols = [
        "fragility_score",
        "durability_requirement",
        "sustainability_priority",
        "innovation_level",
        "material_cost",
        "max_packaging_cost",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Derived features
    df["cost_efficiency"] = df["material_cost"] / (df["max_packaging_cost"] + 1e-6)
    df["eco_pressure"] = 1.0 - df["sustainability_priority"]
    df["durability_pressure"] = (
        df["fragility_score"] * df["durability_requirement"]
    )

    return df
