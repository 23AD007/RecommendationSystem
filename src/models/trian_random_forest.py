import pandas as pd
import mlflow
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score

from src.etl.feature_engineering import derive_features, CORE_FEATURES


def main():
    df = pd.read_csv("data/processed/training_dataset.csv")

    if "recommended" not in df.columns:
        raise ValueError("Missing target column: recommended")

    df = derive_features(df)

    X = df[CORE_FEATURES]
    y = df["recommended"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_experiment("Packaging_RF_Classifier")

    with mlflow.start_run(run_name="random_forest"):
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            random_state=42
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        preds_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds_proba))
        mae = mean_absolute_error(y_test, preds_proba)
        r2 = r2_score(y_test, preds_proba)

        mlflow.log_metrics({
            "accuracy": acc,
            "f1_score": f1,
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        })

        mlflow.sklearn.log_model(model, "rf_classifier")

        print("✅ Random Forest trained successfully")
        print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")


if __name__ == "__main__":
    main()
