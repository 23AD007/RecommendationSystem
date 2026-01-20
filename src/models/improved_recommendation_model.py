import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from src.etl.feature_engineering import derive_features, CORE_FEATURES


class ImprovedRecommendationModel:
    """
    Stable, API-safe recommendation model with explainability support
    """

    def __init__(self, model_path="models/recommendation_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_trained = False

    # --------------------------------------------------
    # ✅ PREPROCESS DATA (RESTORED – CRITICAL)
    # --------------------------------------------------
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies feature engineering + encoding in ONE place
        """
        df = df.copy()

        # Core engineered features
        df = derive_features(df)

        # Product category encoding
        df["product_category"] = df.get("product_category", "unknown").fillna("unknown")
        df["product_category_encoded"] = df["product_category"].map({
            "electronics": 1,
            "food": 2,
            "glassware": 3,
            "pharmaceutical": 4,
            "cosmetics": 5,
            "household": 6
        }).fillna(0).astype(int)

        return df

    # --------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------
    def load_model(self):
        if not os.path.exists(self.model_path):
            return False

        data = joblib.load(self.model_path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        self.is_trained = True

        print("Loaded model:", type(self.model).__name__)
        return True

    # --------------------------------------------------
    # PREDICT (FINAL, STABLE)
    # --------------------------------------------------
    def predict(self, df):
        if not self.is_trained and not self.load_model():
            raise RuntimeError("Model not available")

        df = self.preprocess_data(df)

        for f in self.feature_names:
            if f not in df.columns:
                df[f] = 0

        X = self.scaler.transform(df[self.feature_names])

        proba = self.model.predict_proba(X)[0][1]
        confidence = float(round(proba, 3))
        recommended = confidence >= 0.5

        # Feature importance–based explainability
        importances = self.model.feature_importances_
        top_idx = importances.argsort()[-3:][::-1]

        decision_summary = {}
        for i in top_idx:
            fname = self.feature_names[i]
            decision_summary[fname] = (
                f"{fname} had strong influence "
                f"(importance={importances[i]:.2f}, value={df.iloc[0][fname]:.2f})"
            )

        return {
            "recommended": recommended,
            "confidence": confidence,
            "decision_summary": decision_summary
        }


    # --------------------------------------------------
    # MODEL INFO
    # --------------------------------------------------
    def get_model_info(self):
        return {
            "status": "trained" if self.is_trained else "not_trained",
            "model_type": type(self.model).__name__ if self.model else None,
            "feature_count": len(self.feature_names) if self.feature_names else 0
        }


# --------------------------------------------------
# GLOBAL INSTANCE
# --------------------------------------------------
recommendation_model = ImprovedRecommendationModel()

def get_recommendation_model():
    global recommendation_model
    if not recommendation_model.is_trained:
        recommendation_model.load_model()
    return recommendation_model
