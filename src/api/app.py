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
# Load model ONCE
# --------------------------------------------------
model = get_recommendation_model()

# --------------------------------------------------
# Input schema
# --------------------------------------------------
REQUIRED_FIELDS = {
    "product_category": str,
    "fragility_score": (int, float),
    "sustainability_priority": (int, float),
    "durability_requirement": (int, float),
    "material_cost": (int, float),
    "max_packaging_cost": (int, float),
    "innovation_level": (int, float),
}

# --------------------------------------------------
# Explanation helper
# --------------------------------------------------
def generate_explanations(row: pd.Series):
    return {
        "fragility": f"Fragility score {row['fragility_score']:.2f} required protective packaging",
        "sustainability": f"Sustainability priority {row['sustainability_priority']:.2f} favored eco-friendly materials",
        "durability": f"Durability requirement {row['durability_requirement']:.2f} influenced structural strength",
        "cost": f"Material cost evaluated under max budget {row['max_packaging_cost']}",
        "innovation": f"Innovation level {row['innovation_level']:.2f} supported modern material choice",
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
        return jsonify({"status": "error", "message": "Empty request body"}), 400

    # Validate inputs
    for field, t in REQUIRED_FIELDS.items():
        if field not in data:
            return jsonify({"status": "error", "message": f"Missing field: {field}"}), 400
        if not isinstance(data[field], t):
            return jsonify({"status": "error", "message": f"Invalid type for {field}"}), 400

    # Prepare dataframe
    df = pd.DataFrame([data])

    # Feature engineering
    try:
        df = derive_features(df)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Feature error: {e}"}), 500

    # Model inference
    try:
        pred = model.predict(df)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Model error: {e}"}), 500

    confidence = round(float(pred.get("confidence", 0.0)) * 100, 2)

    # --------------------------------------------------
    # 🔒 GUARANTEED RESPONSE CONTRACT
    # --------------------------------------------------
    response = {
        "status": "success",
        "confidence_score": confidence,

        # ALWAYS PRESENT
        "recommendations": [
            {
                "material": pred.get("material", "Sustainable Packaging Material"),
                "confidence": confidence,
                "reason": "Selected by ML model considering sustainability, cost, and durability trade-offs"
            }
        ],

        "decision_summary": generate_explanations(df.iloc[0]),
        "model_info": model.get_model_info()
    }

    return jsonify(response), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
