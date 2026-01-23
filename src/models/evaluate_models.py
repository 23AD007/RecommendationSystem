def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """
    Wrapper to evaluate a model for API use. Dispatches to classifier or regressor based on model type.
    """
    # Heuristic: classifier if model has predict_proba, else regressor
    if hasattr(model, 'predict_proba'):
        return evaluate_classifier(name, model, X_train, X_test, y_train, y_test)
    else:
        return evaluate_regressor(name, model, X_train, X_test, y_train, y_test)
import mlflow
import mlflow.sklearn

import pandas as pd
import numpy as np


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, r2_score, root_mean_squared_error, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor
import lightgbm as lgb
# from catboost import CatBoostClassifier
from src.utils.preprocessing import load_and_prepare_data



def evaluate_classifier(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    preds_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1_score": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, preds_proba),
        "rmse": root_mean_squared_error(y_test, preds_proba),
        "mae": mean_absolute_error(y_test, preds_proba),
        "r2": r2_score(y_test, preds_proba),
    }
    mlflow.set_experiment("Model_Comparison")
    with mlflow.start_run(run_name=name):
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name)
    return metrics

def evaluate_regressor(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "rmse": root_mean_squared_error(y_test, preds),
        "mae": mean_absolute_error(y_test, preds),
        "r2": r2_score(y_test, preds),
    }
    mlflow.set_experiment("Model_Comparison")
    with mlflow.start_run(run_name=name):
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name)
    return metrics


def main():

    # Classification task
    X_cls, y_cls = load_and_prepare_data("data/processed/training_dataset.csv")
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
        X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )
    classifiers = {
        "Logistic_Regression": LogisticRegression(max_iter=1000),
        "Random_Forest_Classifier": RandomForestClassifier(n_estimators=300, random_state=42),
        "XGBoost_Classifier": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, eval_metric="logloss", random_state=42),
        "LightGBM_Classifier": lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42)
    }
    cls_results = []
    for name, model in classifiers.items():
        print(f"\nTraining {name}...")
        metrics = evaluate_classifier(name, model, X_train_cls, X_test_cls, y_train_cls, y_test_cls)
        metrics["model"] = name
        cls_results.append(metrics)
    cls_results_df = pd.DataFrame(cls_results).sort_values(by="f1_score", ascending=False)
    print("\n===== CLASSIFICATION MODEL COMPARISON RESULTS =====")
    print(cls_results_df)

    # Regression task: Cost prediction
    df = pd.read_csv("data/processed/training_dataset.csv")
    from src.etl.feature_engineering import derive_features, CORE_FEATURES
    df = derive_features(df)
    X_reg = df[CORE_FEATURES]
    y_cost = df["material_cost"]
    X_train_reg, X_test_reg, y_train_cost, y_test_cost = train_test_split(
        X_reg, y_cost, test_size=0.2, random_state=42
    )
    regressors = {
        "Random_Forest_Regressor": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    }
    reg_results = []
    for name, model in regressors.items():
        print(f"\nTraining {name} (Cost)...")
        metrics = evaluate_regressor(name, model, X_train_reg, X_test_reg, y_train_cost, y_test_cost)
        metrics["model"] = name
        reg_results.append(metrics)
    reg_results_df = pd.DataFrame(reg_results).sort_values(by="rmse")
    print("\n===== COST REGRESSION MODEL COMPARISON RESULTS =====")
    print(reg_results_df)

    # Regression task: CO2 prediction
    if "eco_pressure" in df.columns:
        y_co2 = df["eco_pressure"]
    elif "co2_footprint" in df.columns:
        y_co2 = df["co2_footprint"]
    else:
        raise ValueError("No CO2 target column found in dataset.")
    # Drop NaN/infinite
    mask = np.isfinite(y_co2)
    X_co2 = X_reg[mask]
    y_co2 = y_co2[mask]
    X_train_co2, X_test_co2, y_train_co2, y_test_co2 = train_test_split(
        X_co2, y_co2, test_size=0.2, random_state=42
    )
    co2_regressors = {
        "XGBoost_CO2_Regressor": XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.05, random_state=42)
    }
    co2_results = []
    for name, model in co2_regressors.items():
        print(f"\nTraining {name} (CO2)...")
        metrics = evaluate_regressor(name, model, X_train_co2, X_test_co2, y_train_co2, y_test_co2)
        metrics["model"] = name
        co2_results.append(metrics)
    co2_results_df = pd.DataFrame(co2_results).sort_values(by="rmse")
    print("\n===== CO2 REGRESSION MODEL COMPARISON RESULTS =====")
    print(co2_results_df)


if __name__ == "__main__":
    main()
