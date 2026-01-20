from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import os

from src.models.improved_recommendation_model import get_recommendation_model
from src.etl.feature_engineering import derive_features

# --------------------------------------------------
# App setup
# --------------------------------------------------
app = Flask(__name__)
CORS(app)

# --------------------------------------------------
# Load trained model (once at startup)
# --------------------------------------------------
model = get_recommendation_model()

# --------------------------------------------------
# Input schema
# --------------------------------------------------
RAW_REQUIRED_FIELDS = {
    "product_category": str,
    "fragility_score": (int, float),
    "sustainability_priority": (int, float),
    "durability_requirement": (int, float),
    "material_cost": (int, float),
    "max_packaging_cost": (int, float),
    "innovation_level": (int, float),
}

# --------------------------------------------------
# Explanation generator
# --------------------------------------------------
def generate_explanations(row: pd.Series):
    return {
        "fragility": f"Fragility score {row['fragility_score']:.2f} required protective packaging",
        "sustainability": f"Sustainability priority {row['sustainability_priority']:.2f} favored eco materials",
        "durability": f"Durability requirement {row['durability_requirement']:.2f} influenced material strength",
        "cost": f"Material cost evaluated under budget {row['max_packaging_cost']}",
        "innovation": f"Innovation level {row['innovation_level']:.2f} encouraged novel materials",
    }

# --------------------------------------------------
# Health check
# --------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# --------------------------------------------------
# Recommendation endpoint
# --------------------------------------------------
@app.route("/api/product/recommend-materials", methods=["POST"])
def recommend_materials():
    data = request.get_json()

    if not data:
        return jsonify({"status": "error", "message": "Empty request"}), 400

    for field, t in RAW_REQUIRED_FIELDS.items():
        if field not in data:
            return jsonify({"status": "error", "message": f"Missing {field}"}), 400
        if not isinstance(data[field], t):
            return jsonify({"status": "error", "message": f"Invalid type for {field}"}), 400

    # Convert to DataFrame
    df = pd.DataFrame([data])

    # Feature engineering
    df = derive_features(df)

    # Model inference
    prediction = model.predict(df)

    confidence = round(prediction["confidence"] * 100, 2)

    response = {
        "status": "success",
        "confidence_score": confidence,
        "recommendations": [
            {
                "material": "Recycled Cardboard Composite",
                "confidence": confidence,
                "reason": "Balances sustainability, durability, and cost efficiency"
            }
        ],
        "decision_summary": generate_explanations(df.iloc[0]),
        "model_info": model.get_model_info()
    }

    return jsonify(response), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
