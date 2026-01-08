import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
from sklearn.ensemble import RandomForestClassifier

from src.etl.features import CORE_FEATURES
from src.etl.feature_engineering import derive_features
import json
import tempfile
from src.models.material_affinity import learn_material_affinity
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def main():
    # --------------------------------------------------
    # 1. Load training data
    # --------------------------------------------------
    df = pd.read_csv("data/processed/training_dataset.csv")

    print("Initial columns:", df.columns.tolist())
    df = derive_features(df)
    df[CORE_FEATURES] = df[CORE_FEATURES].replace([np.inf, -np.inf], np.nan)    
    # --------------------------------------------------
    # 2. DERIVE CORE SUSTAINABILITY FEATURES
    # (because they do NOT exist in raw data)
    # --------------------------------------------------

    # eco_pressure → inverse of sustainability priority
    if "eco_pressure" not in df.columns:
        if "sustainability_priority" not in df.columns:
            raise ValueError("Missing 'sustainability_priority' to derive eco_pressure")
        df["eco_pressure"] = 1 - df["sustainability_priority"]

    # cost_efficiency → how cheap material is relative to max allowed
    if "cost_efficiency" not in df.columns:
        if not {"material_cost", "max_packaging_cost"}.issubset(df.columns):
            raise ValueError("Missing cost columns to derive cost_efficiency")

        df["cost_efficiency"] = 1 - (
            df["material_cost"] / df["max_packaging_cost"]
        )

    # durability_pressure → fragility vs durability requirement
    if "durability_pressure" not in df.columns:
        if not {"fragility_score", "durability_requirement"}.issubset(df.columns):
            raise ValueError("Missing durability columns to derive durability_pressure")

        df["durability_pressure"] = (
            df["fragility_score"] / df["durability_requirement"]
        )

    # --------------------------------------------------
    # 3. CLEAN DERIVED FEATURES (CRITICAL)
    # --------------------------------------------------
    df[CORE_FEATURES] = df[CORE_FEATURES].replace([np.inf, -np.inf], np.nan)

    df[CORE_FEATURES] = df[CORE_FEATURES].apply(
        lambda col: col.fillna(col.median())
    )

    # Clamp to valid range where applicable
    for col in ["eco_pressure", "cost_efficiency", "durability_pressure"]:
        df[col] = df[col].clip(0, 1)

    # --------------------------------------------------
    # 4. CREATE TARGET VARIABLE
    # --------------------------------------------------
    if "overall_sustainability_score" not in df.columns:
        df["overall_sustainability_score"] = (
            0.4 * (1 - df["eco_pressure"]) +
            0.3 * df["cost_efficiency"] +
            0.3 * (1 - df["durability_pressure"])
        )

    # Safety check
    if not np.isfinite(df["overall_sustainability_score"]).all():
        raise ValueError("Target contains NaN or infinite values")

    # --------------------------------------------------
    # 5. FEATURE / TARGET SPLIT
    # --------------------------------------------------
    X = df[CORE_FEATURES]
    y = df["overall_sustainability_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --------------------------------------------------
    # 6. TRAIN XGBOOST MODEL
    # --------------------------------------------------
    mlflow.set_experiment("Packaging_Sustainability_XGBoost")

    with mlflow.start_run(run_name="xgb_regressor"):
        model = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        mlflow.log_metrics({
            "rmse": rmse,
            "mae": mae,
            "r2_score": r2
        })

        # SAFE MLflow logging
        mlflow.xgboost.log_model(
            model.get_booster(),
            artifact_path="xgboost_model"
        )
                # --------------------------------------------------
        # Learn material–product affinity
        # --------------------------------------------------
        affinity = learn_material_affinity(df)

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(affinity, tmp, indent=2)
            affinity_path = tmp.name

        # Log as MLflow artifact
        mlflow.log_artifact(
            affinity_path,
            artifact_path="material_affinity"
        )
        mlflow.log_param("uses_material_affinity", True)
        mlflow.log_param(
            "affinity_categories",
            ",".join(sorted(df["product_category"].unique()))
        )

        print("✅ Material affinity logged to MLflow")


        print("✅ Training completed successfully")
        print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2 Score: {r2:.4f}")
        affinity = learn_material_affinity(df)

        with open("models/material_affinity.json", "w") as f:
            json.dump(affinity, f, indent=2)

        print("✅ Learned material affinity saved")


if __name__ == "__main__":
    main()
