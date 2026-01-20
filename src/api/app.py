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
# Load trained recommendation model (ONCE)
# --------------------------------------------------
model = get_recommendation_model()

# --------------------------------------------------
# RAW input schema (frontend → backend)
# --------------------------------------------------
RAW_REQUIRED_FIELDS = {
    "fragility_score": (int, float),
    "sustainability_priority": (int, float),
    "durability_requirement": (int, float),
    "material_cost": (int, float),
    "max_packaging_cost": (int, float),
    "innovation_level": (int, float),
    "product_category": str
}

# --------------------------------------------------
# Heuristic explanations
# --------------------------------------------------
def generate_explanations(row: pd.Series) -> dict:
    return {
        "fragility": (
            f"Fragility score {row['fragility_score']:.2f} "
            "increased the need for protective cushioning"
        ),
        "sustainability": (
            f"Sustainability priority {row['sustainability_priority']:.2f} "
            "favored environmentally friendly materials"
        ),
        "durability": (
            f"Durability requirement {row['durability_requirement']:.2f} "
            "influenced structural strength selection"
        ),
        "cost": (
            f"Material cost was evaluated against the maximum budget "
            f"{row['max_packaging_cost']}"
        ),
        "innovation": (
            f"Innovation level {row['innovation_level']:.2f} "
            "affected preference for novel materials"
        ),
    }

# --------------------------------------------------
# Health check (Render requirement)
# --------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "AI Packaging Recommendation API"
    })

# --------------------------------------------------
# Recommendation endpoint
# --------------------------------------------------
@app.route("/api/product/recommend-materials", methods=["POST"])
def recommend_materials():
    data = request.get_json()

    if not data:
        return jsonify({
            "status": "error",
            "message": "Empty request body"
        }), 400

    # 1. Validate inputs
    for field, expected_type in RAW_REQUIRED_FIELDS.items():
        if field not in data:
            return jsonify({
                "status": "error",
                "message": f"Missing required field: {field}"
            }), 400

        if not isinstance(data[field], expected_type):
            return jsonify({
                "status": "error",
                "message": f"Invalid type for field: {field}"
            }), 400

    # 2. Convert to DataFrame
    df = pd.DataFrame([data])

    # 3. Feature engineering
    try:
        df = derive_features(df)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Feature engineering failed: {str(e)}"
        }), 500

    # 4. Model inference (✅ CORRECT METHOD)
    try:
        prediction = model.predict(df)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Model inference failed: {str(e)}"
        }), 500

    # 5. Build API response
    response = {
        "status": "success",
        "recommended": prediction["recommended"],
        "confidence_score": prediction["confidence"],
        "recommendations": [
            {
                "material": "AI Selected Material",
                "confidence": round(prediction["confidence"] * 100, 1),
                "reason": "Recommended based on sustainability, cost, and durability trade-offs"
            }
        ],
        "decision_summary": generate_explanations(df.iloc[0]),
        "model_info": model.get_model_info()
    }

    return jsonify(response), 200

# --------------------------------------------------
# Local run (ignored by Gunicorn)
# --------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
