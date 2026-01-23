import os
import joblib
import numpy as np
import pandas as pd
import requests

from src.etl.feature_engineering import derive_features, CORE_FEATURES

# Loader for material options from PostgreSQL
from src.sustainability.data_access.loader import load_data as load_material_options

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.normpath(
    os.path.join(BASE_DIR, "..", "..", "models", "xgb_ranker.pkl")
)

# set to local backend when running locally
BACKEND_URL = "http://localhost:5000/api/recommend-materials"


# ==================================================
# FALLBACK MODEL (DATA-DRIVEN, NOT HARDCODED)
# ==================================================
class FallbackRecommendationModel:
    def predict(self, df: pd.DataFrame):
        # Load all material options from PostgreSQL
        material_options = load_material_options()

        # Cross join user/product features with all material options
        user_features = df.iloc[0].to_dict()
        user_df = pd.DataFrame([user_features] * len(material_options)).reset_index(drop=True)
        candidates = pd.concat([user_df.reset_index(drop=True), material_options.reset_index(drop=True)], axis=1)

        # Feature engineering
        candidates = derive_features(candidates)
        candidates["score"] = (
            0.4 * (1 - candidates["eco_pressure"]) +
            0.3 * candidates["cost_efficiency"] +
            0.3 * (1 - candidates["durability_pressure"])
        )
        best = candidates.loc[candidates["score"].idxmax()]
        confidence = float(np.clip(best["score"], 0, 1))
        material = best["material_name"] if "material_name" in best else (best["material"] if "material" in best else "Unknown Material")
        return {
            "material": material,
            "confidence": round(confidence, 3),
            "logic": "fallback-rule-based (feature-derived)"
        }

    def get_model_info(self):
        return {
            "model": "fallback-rule-based",
            "note": "Computed from engineered features (no hardcoding)"
        }


# ==================================================
# REAL TRAINED MODEL
# ==================================================
class ImprovedRecommendationModel:
    def __init__(self):
        print(f"[DEBUG] Looking for model at: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            print("[DEBUG] Model file not found!")
            raise FileNotFoundError("Trained model not found")
        self.model = joblib.load(MODEL_PATH)
        print("[DEBUG] Model loaded successfully.")

    def predict(self, df: pd.DataFrame):
        try:
            # Load all material options from PostgreSQL
            material_options = load_material_options()
            print(f"[DEBUG] Loaded {len(material_options)} material options from DB.")

            # Cross join user/product features with all material options
            user_features = df.iloc[0].to_dict()
            user_df = pd.DataFrame([user_features] * len(material_options)).reset_index(drop=True)
            candidates = pd.concat([user_df.reset_index(drop=True), material_options.reset_index(drop=True)], axis=1)

            # Feature engineering
            candidates = derive_features(candidates)
            print(f"[DEBUG] Candidate columns: {list(candidates.columns)}")
            print(f"[DEBUG] Model expects features: {CORE_FEATURES}")
            X = candidates[CORE_FEATURES]
            scores = self.model.predict(X)
            candidates["score"] = scores
            best = candidates.loc[candidates["score"].idxmax()]
            confidence = float(
                (best["score"] - candidates["score"].min()) /
                (candidates["score"].max() - candidates["score"].min() + 1e-9)
            )
            material = best["material_name"] if "material_name" in best else (best["material"] if "material" in best else "Unknown Material")
            print(f"[DEBUG] Top material: {material}, confidence: {confidence}")
            return {
                "material": material,
                "confidence": round(confidence, 3),
                "logic": "trained-xgboost-ranker"
            }
        except Exception as e:
            print(f"[DEBUG] Exception in model predict: {e}")
            raise

    def get_model_info(self):
        return {
            "model": "xgboost-ranker",
            "source": "local-trained-model"
        }


# ==================================================
# SAFE FACTORY
# ==================================================
def get_recommendation_model():
    try:
        return ImprovedRecommendationModel()
    except Exception as e:
        print("⚠️ Falling back to rule-based model:", e)
        return FallbackRecommendationModel()


# when making request:
def make_request(payload):
    response = requests.post(
        BACKEND_URL,
        json=payload,
        headers={
            "Content-Type": "application/json",
            "X-API-Key": "packaging-api-key-2024"
        },
        timeout=15
    )
    return response.json()
