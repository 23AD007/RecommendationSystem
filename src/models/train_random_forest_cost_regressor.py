import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from src.etl.feature_engineering import derive_features, CORE_FEATURES

# CONFIG
DATA_PATH = "data/processed/training_dataset.csv"
MODEL_PATH = "models/random_forest_cost_regressor.pkl"

def main():
    print("Loading training dataset...")
    df = pd.read_csv(DATA_PATH)
    df = derive_features(df)

    # Target: material_cost
    X = df[CORE_FEATURES]
    y = df["material_cost"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(f"RMSE: {root_mean_squared_error(y_test, preds):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, preds):.4f}")
    print(f"R2: {r2_score(y_test, preds):.4f}")

    joblib.dump(model, MODEL_PATH)
    print(f"Random Forest cost regressor saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
