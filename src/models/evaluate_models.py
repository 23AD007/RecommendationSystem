import mlflow
import mlflow.sklearn

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

from src.utils.preprocessing import load_and_preprocess_data


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1_score": f1_score(y_test, preds, zero_division=0),
    }

    mlflow.set_experiment("Model_Comparison")

    with mlflow.start_run(run_name=name):
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name)

    return metrics


def main():
    X_train, X_test, y_train, y_test = load_and_preprocess_data(
        csv_path="data/processed/training_dataset.csv",
        target_col="recommended"
    )

    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000),
        "Random_Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=42
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            random_state=42
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            verbose=0,
            random_seed=42
        )
    }

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        metrics = evaluate_model(
            name, model, X_train, X_test, y_train, y_test
        )
        metrics["model"] = name
        results.append(metrics)

    results_df = pd.DataFrame(results).sort_values(
        by="f1_score", ascending=False
    )

    print("\n===== MODEL COMPARISON RESULTS =====")
    print(results_df)


if __name__ == "__main__":
    main()
