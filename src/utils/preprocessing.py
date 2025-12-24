import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_and_prepare_data(csv_path: str):
    df = pd.read_csv(csv_path)

    # Encode categorical features
    categorical_cols = ["product_category", "innovation_level"]
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Add controlled noise to avoid deterministic labels
    df["score_noisy"] = (
        df["overall_sustainability_score"]
        + np.random.normal(0, 0.05, size=len(df))
    )

    # Binary classification target
    df["recommended"] = (df["score_noisy"] >= 0.6).astype(int)

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
