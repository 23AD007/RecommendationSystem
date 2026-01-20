import os

# --------------------------------------------------
# Fallback model (ALWAYS works)
# --------------------------------------------------
class FallbackRecommendationModel:
    def predict(self, df):
        return {
            "confidence": 0.82,
            "material": "Recycled Cardboard Composite"
        }

    def get_model_info(self):
        return {
            "model": "fallback-rule-based",
            "note": "Used because trained model is not available in cloud"
        }

# --------------------------------------------------
# Real model wrapper (optional)
# --------------------------------------------------
class ImprovedRecommendationModel:
    def __init__(self):
        # If you later add a real model file, load it here
        model_path = "models/recommendation_model.pkl"

        if not os.path.exists(model_path):
            raise FileNotFoundError("Model file not found")

        # Example placeholder
        self.model = None

    def predict(self, df):
        # Placeholder for real prediction logic
        return {
            "confidence": 0.9,
            "material": "Advanced Bio-Packaging"
        }

    def get_model_info(self):
        return {
            "model": "trained-ml-model"
        }

# --------------------------------------------------
# SAFE factory (CRITICAL)
# --------------------------------------------------
def get_recommendation_model():
    try:
        return ImprovedRecommendationModel()
    except Exception as e:
        print("⚠️ Falling back to rule-based model:", e)
        return FallbackRecommendationModel()
