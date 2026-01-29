import pandas as pd
import numpy as np
import joblib

from xgboost import XGBRanker
from scipy.stats import spearmanr

from src.etl.feature_engineering import derive_features
from src.models.features import CORE_FEATURES

def main():
    df = pd.read_csv("data/processed/training_dataset.csv")

    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())

    df = derive_features(df)
    X = df.reindex(columns=CORE_FEATURES)

    if X.isnull().any().any():
        raise ValueError("NaN detected in training features")

    y = df["recommended"].astype(float)
    # Target
    if "recommended" not in df.columns:
        raise ValueError("Missing target column: recommended")

    df["relevance"] = df["recommended"].astype(float)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)

    X = df[CORE_FEATURES]
    y = df["relevance"]

    # Single query group (simple ranking)
    group = [len(df)]

    model = XGBRanker(
        objective="rank:pairwise",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X, y, group=group)

    preds = model.predict(X)
    corr, _ = spearmanr(y, preds)

    print("Spearman Rank Correlation:", corr)

    joblib.dump(model, "models/xgb_ranker.pkl")
    print("✅ Model trained and saved")

if __name__ == "__main__":
    main()
