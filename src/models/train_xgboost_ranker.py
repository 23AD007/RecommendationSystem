import os
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from scipy.stats import spearmanr

from src.etl.feature_engineering import derive_features, CORE_FEATURES


# CONFIG
DATA_PATH = "data/processed/training_dataset.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_ranker.pkl")


# MAIN TRAINING PIPELINE
def main():

    print("Loading training dataset...")
    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("Training dataset is empty")

    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Filter out rows with too many missing values
    # Keep only rows with at least 3 non-null values in key columns
    key_columns = ['fragility_score', 'sustainability_priority', 'durability_requirement',
                   'max_packaging_cost', 'material_cost', 'innovation_level']
    df = df.dropna(subset=key_columns, thresh=3)  # At least 3 non-null
    print(f"Rows after filtering sparse data: {len(df)}")

    if df.empty:
        raise ValueError("No valid training data after filtering")

    # Feature Engineering (same as inference)
    print("Applying feature engineering...")
    df = derive_features(df)

    # Fill NaN for relevance calculation
    df[CORE_FEATURES] = df[CORE_FEATURES].fillna(0)

    # Create relevance score (TARGET)
    print("Creating relevance score...")

    df["relevance"] = (
        0.4 * (1 - df["eco_pressure"]) +
        0.3 * df["cost_efficiency"] +
        0.3 * (1 - df["durability_pressure"])
    )

    if not np.isfinite(df["relevance"]).all():
        raise ValueError("Relevance contains NaN or Inf values")

    # Query groups for ranking
    if "query_id" not in df.columns:
        df["query_id"] = (
            df["product_category"].astype(str) + "_" +
            df.index.astype(str)
        )

    # IMPORTANT: Sort by query_id ONCE
    df = df.sort_values("query_id").reset_index(drop=True)

    # Prepare training inputs
    X = df[CORE_FEATURES]
    y = df["relevance"]

    group_sizes = (
        df.groupby("query_id", sort=False)
        .size()
        .astype(int)
        .tolist()
    )

    assert sum(group_sizes) == len(df), "Group sizes mismatch"

    
    # Train XGBoost Regressor
    print("Training XGBoost Regressor...")

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X, y)

    # Evaluation (sanity check)
    preds = model.predict(X)

    if np.std(preds) == 0:
        print("Warning: Model predictions are constant")
        spearman_corr = 0.0
    else:
        spearman_corr, _ = spearmanr(y, preds)
        spearman_corr = float(np.nan_to_num(spearman_corr))

    print(f"Spearman Rank Correlation: {spearman_corr:.4f}")

    # Save trained model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}")
    print("Training complete")


# ENTRY POINT
if __name__ == "__main__":
    main()
