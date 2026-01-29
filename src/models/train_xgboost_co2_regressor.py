import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
import os

MATERIALS_PATH = "data/materials.csv"
MODEL_PATH = "models/xgb_regressor.pkl"

os.makedirs("models", exist_ok=True)

def main():
    materials = pd.read_csv(MATERIALS_PATH)

    rows = []
    for _, m in materials.iterrows():
        for frag in np.linspace(0, 1, 5):
            for sus in np.linspace(0, 1, 5):
                for dur in np.linspace(0, 1, 5):
                    score = (
                        0.4 * m.eco_score * sus +
                        0.35 * m.durability_score * dur +
                        0.25 * (1 - m.cost_score)
                    )

                    rows.append({
                        "fragility_score": frag,
                        "sustainability_priority": sus,
                        "durability_requirement": dur,
                        "eco_score": m.eco_score,
                        "durability_score": m.durability_score,
                        "cost_score": m.cost_score,
                        "fragility_support": m.fragility_support,
                        "target": score
                    })

    df = pd.DataFrame(rows)

    X = df.drop(columns=["target"])
    y = df["target"]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )

    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

    print("✅ XGBoost Regressor trained and saved")

if __name__ == "__main__":
    main()
