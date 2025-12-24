import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
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


def prepare_features(df: pd.DataFrame):
    """
    Cleans, validates, and returns X matrix
    """
    df = df.copy()

    # Replace inf → NaN
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)

    # Fill NaN with median
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].apply(
        lambda col: col.fillna(col.median())
    )

    return df[FEATURE_COLUMNS]
