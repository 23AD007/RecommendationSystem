
import pandas as pd
import joblib
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from src.etl.feature_engineering import derive_features, CORE_FEATURES

# CONFIG
DATA_PATH = "data/processed/training_dataset.csv"
MODEL_PATH = "models/xgb_co2_regressor.pkl"

def main():
    print("Loading training dataset...")
    df = pd.read_csv(DATA_PATH)
    df = derive_features(df)

    # Target: eco_pressure (as a proxy for CO2 footprint, or replace with actual CO2 column if available)

    if "eco_pressure" in df.columns:
        target_col = "eco_pressure"
    elif "co2_footprint" in df.columns:
        target_col = "co2_footprint"
    else:
        raise ValueError("No CO2 target column found in dataset.")

    # Drop rows with NaN or infinite target values
    df = df[np.isfinite(df[target_col])]
    df = df.dropna(subset=[target_col])

    X = df[CORE_FEATURES]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(f"RMSE: {root_mean_squared_error(y_test, preds):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, preds):.4f}")
    print(f"R2: {r2_score(y_test, preds):.4f}")

    joblib.dump(model, MODEL_PATH)
    print(f"XGBoost CO2 regressor saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
