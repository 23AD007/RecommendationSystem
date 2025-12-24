import pandas as pd
import mlflow

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

    mlflow.set_experiment("Packaging_Logistic_Baseline")

    with mlflow.start_run(run_name="logistic_regression"):
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "logistic_regression")

        print("✅ Logistic Regression trained successfully")
        print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
