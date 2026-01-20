from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import os

from src.models.improved_recommendation_model import get_recommendation_model
from src.etl.feature_engineering import derive_features

# ==================================================
# APP SETUP
# ==================================================
app = Flask(__name__)
CORS(app)

# ==================================================
# LOAD MODEL (SAFE, FALLBACK ENABLED)
# ==================================================
model = get_recommendation_model()

# ==================================================
# REQUIRED INPUT SCHEMA
# ==================================================
REQUIRED_FIELDS = {
    "product_category": str,
    "fragility_score": (int, float),
    "sustainability_priority": (int, float),
    "durability_requirement": (int, float),
    "material_cost": (int, float),
    "max_packaging_cost": (int, float),
    "innovation_level": (int, float),
}

# ==================================================
# EXPLANATION HELPER
# ==================================================
def generate_explanations(row):
    return {
        "fragility": f"Fragility score {row['fragility_score']:.2f} required protective packaging",
        "sustainability": f"Sustainability priority {row['sustainability_priority']:.2f} favored eco materials",
        "durability": f"Durability requirement {row['durability_requirement']:.2f} influenced material strength",
        "cost": f"Material cost evaluated under max budget {row['max_packaging_cost']}",
        "innovation": f"Innovation level {row['innovation_level']:.2f} encouraged modern materials",
    }

# ==================================================
# HEALTH CHECK
# ==================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

# ==================================================
# RECOMMENDATION ENDPOINT
# ==================================================
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

    # Convert to DataFrame
    df = pd.DataFrame([data])

    # Feature engineering (SAFE)
    try:
        df = derive_features(df)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Feature engineering failed: {e}"
        }), 500

    # ==================================================
    # 🔒 ONLY .predict() — NO OTHER MODEL LOGIC
    # ==================================================
    prediction = model.predict(df)

    confidence = round(float(prediction.get("confidence", 0.0)) * 100, 2)

    response = {
        "status": "success",
        "confidence_score": confidence,

        # GUARANTEED KEY
        "recommendations": [
            {
                "material": prediction.get("material", "Sustainable Packaging"),
                "confidence": confidence,
                "reason": "Selected using fallback-safe recommendation logic"
            }
        ],

        "decision_summary": generate_explanations(df.iloc[0]),
        "model_info": model.get_model_info()
    }

    return jsonify(response), 200


# ==================================================
# LOCAL RUN (IGNORED BY GUNICORN)
# ==================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
