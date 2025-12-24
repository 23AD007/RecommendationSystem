import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_and_prepare_data(csv_path: str):
    df = pd.read_csv(csv_path)

    # --------------------------------------------------
    # CREATE TARGET VARIABLE IF MISSING
    # --------------------------------------------------
    if "overall_sustainability_score" not in df.columns:
        # Derive eco_pressure → inverse of sustainability priority
        if "sustainability_priority" in df.columns:
            df["eco_pressure"] = 1 - df["sustainability_priority"]
        else:
            raise ValueError("Missing 'sustainability_priority' to derive eco_pressure")

        # cost_efficiency → how cheap material is relative to max allowed
        if {"material_cost", "max_packaging_cost"}.issubset(df.columns):
            df["cost_efficiency"] = 1 - (df["material_cost"] / df["max_packaging_cost"])
        else:
            raise ValueError("Missing cost columns to derive cost_efficiency")

        # durability_pressure → fragility vs durability requirement
        if {"fragility_score", "durability_requirement"}.issubset(df.columns):
            df["durability_pressure"] = df["fragility_score"] / df["durability_requirement"]
        else:
            raise ValueError("Missing durability columns to derive durability_pressure")

        df["overall_sustainability_score"] = (
            0.4 * (1 - df["eco_pressure"]) +
            0.3 * df["cost_efficiency"] +
            0.3 * (1 - df["durability_pressure"])
        )

    # Encode categorical features
    categorical_cols = ["product_category", "innovation_level"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))  # Handle NaN by converting to string

    # Handle NaN values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Add controlled noise to avoid deterministic labels
    df["score_noisy"] = (
        df["overall_sustainability_score"]
        + np.random.normal(0, 0.05, size=len(df))
    )

    # Binary classification target
    df["recommended"] = (df["score_noisy"] >= 0.5).astype(int)

    # Drop leakage columns
    drop_cols = [
        "product_id",
        "overall_sustainability_score",
        "score_noisy",
        "eco_impact_index",
        "recyclability_score",
        "co2_emission"
    ]

    X = df.drop(columns=drop_cols + ["recommended"], errors="ignore")
    y = df["recommended"]

    return X, y
