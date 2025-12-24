import mlflow
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score

from src.utils.preprocessing import load_and_prepare_data


def main():
    X, y = load_and_prepare_data("data/processed/training_dataset.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_experiment("Packaging_CatBoost")

    with mlflow.start_run():
        model = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            verbose=0,
            random_state=42
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        roc_auc_val = roc_auc_score(y_test, probs)
        rmse = np.sqrt(mean_squared_error(y_test, probs))
        mae = mean_absolute_error(y_test, probs)
        r2 = r2_score(y_test, probs)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc_val)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        print("✅ CatBoost trained successfully")
        print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc_val:.4f}")
        print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")


if __name__ == "__main__":
    main()
