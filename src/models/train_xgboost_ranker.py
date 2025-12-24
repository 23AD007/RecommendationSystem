import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost

from xgboost import XGBRanker
from scipy.stats import spearmanr

from src.etl.feature_engineering import derive_features, CORE_FEATURES


def main():
    # --------------------------------------------------
    # 0. 🔑 DISABLE AUTOLOGGING (CRITICAL FIX)
    # --------------------------------------------------
    mlflow.autolog(disable=True)

    # --------------------------------------------------
    # 1. Load training data
    # --------------------------------------------------
    df = pd.read_csv("data/processed/training_dataset.csv")

    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())

    # --------------------------------------------------
    # 2. Feature engineering (same as regressor)
    # --------------------------------------------------
    df = derive_features(df)

    # --------------------------------------------------
    # 3. Create relevance score
    # --------------------------------------------------
    df["relevance"] = (
        0.4 * (1 - df["eco_pressure"]) +
        0.3 * df["cost_efficiency"] +
        0.3 * (1 - df["durability_pressure"])
    )

    if not np.isfinite(df["relevance"]).all():
        raise ValueError("Relevance contains NaN or infinity")

    # --------------------------------------------------
    # 4. Create TRUE query_id (ranking queries)
    # --------------------------------------------------
    if "query_id" not in df.columns:
        if "product_category" not in df.columns:
            raise ValueError(
                "Ranking requires 'product_category' or 'query_id'"
            )

        df["query_id"] = (
            df["product_category"].astype(str) + "_" +
            df.index.astype(str)
        )

    # --------------------------------------------------
    # 5. Sort ONCE by query_id
    # --------------------------------------------------
    df = df.sort_values("query_id").reset_index(drop=True)

    # --------------------------------------------------
    # 6. Build X, y, and group from SAME df
    # --------------------------------------------------
    X = df[CORE_FEATURES]
    y = df["relevance"]

    group_sizes = (
        df.groupby("query_id", sort=False)
        .size()
        .astype(int)
        .tolist()
    )

    assert sum(group_sizes) == len(df), (
        f"Group mismatch: sum(groups)={sum(group_sizes)}, rows={len(df)}"
    )

    # --------------------------------------------------
    # 7. Train XGBoost Ranker
    # --------------------------------------------------
    mlflow.set_experiment("Packaging_XGBoost_Ranker")

    with mlflow.start_run(run_name="xgb_ranker"):
        model = XGBRanker(
            objective="rank:pairwise",
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        model.fit(X, y, group=group_sizes)

        # --------------------------------------------------
        # 8. Evaluate (ONCE)
        # --------------------------------------------------
        preds = model.predict(X)
        spearman_corr, _ = spearmanr(y, preds)

        # --------------------------------------------------
        # 9. LOG METRIC EXACTLY ONCE (SAFE)
        # --------------------------------------------------
        mlflow.log_metric(
            "spearman_rank_corr",
            float(spearman_corr),
            step=0
        )

        mlflow.xgboost.log_model(
            model.get_booster(),
            artifact_path="xgb_ranker"
        )

        print("✅ XGBoost Ranker trained successfully")
        print(f"Spearman Rank Corr: {spearman_corr:.4f}")


if __name__ == "__main__":
    main()
