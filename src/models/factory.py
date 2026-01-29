from src.models.improved_recommendation_model import ImprovedRecommendationModel
from src.models.fallback_model import FallbackRecommendationModel

def get_recommendation_model():
    try:
        return ImprovedRecommendationModel()
    except Exception as e:
        print("⚠️ Falling back:", e)
        return FallbackRecommendationModel()
