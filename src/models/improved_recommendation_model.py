import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "..", "models", "xgb_regressor.pkl")
MATERIALS_PATH = os.path.join(BASE_DIR, "..", "..", "data", "materials.csv")

class ImprovedRecommendationModel:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.materials = pd.read_csv(MATERIALS_PATH)

    def predict(self, user_df):
        results = []

        for _, m in self.materials.iterrows():
            row = {
                "fragility_score": user_df.fragility_score.iloc[0],
                "sustainability_priority": user_df.sustainability_priority.iloc[0],
                "durability_requirement": user_df.durability_requirement.iloc[0],
                "eco_score": m.eco_score,
                "durability_score": m.durability_score,
                "cost_score": m.cost_score,
                "fragility_support": m.fragility_support,
            }

            X = pd.DataFrame([row])
            confidence = float(self.model.predict(X)[0])
            confidence = round(confidence * 100, 2)

            reasoning = (
                f"Eco match: {round(m.eco_score * user_df.sustainability_priority.iloc[0], 2)}, "
                f"Durability match: {round(m.durability_score * user_df.durability_requirement.iloc[0], 2)}, "
                f"Cost efficiency: {round(1 - m.cost_score, 2)}"
            )

            results.append({
                "material": m.material,
                "confidence": confidence,
                "reasoning": reasoning,
                "model": "xgboost-regressor"
            })

        results = sorted(results, key=lambda x: x["confidence"], reverse=True)

        return {
            "recommendations": results
        }

def get_recommendation_model():
    return ImprovedRecommendationModel()
