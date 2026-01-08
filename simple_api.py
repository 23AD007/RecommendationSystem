from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd

from src.models.improved_recommendation_model import get_recommendation_model
from src.etl.feature_engineering import derive_features

# --------------------------------------------------
# App setup
# --------------------------------------------------
app = Flask(__name__)
CORS(app)

# Load trained recommendation model (once)
model = get_recommendation_model()

# --------------------------------------------------
# RAW input schema (frontend → backend)
# Only RAW features, no derived ones
# --------------------------------------------------
RAW_REQUIRED_FIELDS = {
    "fragility_score": float,
    "sustainability_priority": float,
    "durability_requirement": float,
    "material_cost": (int, float),
    "max_packaging_cost": (int, float),
    "innovation_level": (int, float),
    "product_category": str
}

# --------------------------------------------------
# Heuristic explanations (dataset-faithful)
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
        "eco_pressure": (
            "Environmental pressure influenced sustainability weighting"
        ),
        "innovation": (
            f"Innovation level {row['innovation_level']:.2f} "
            "affected preference for novel materials"
        ),
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
        return jsonify({
            "status": "error",
            "message": "Empty request body"
        }), 400

    # --------------------------------------------------
    # 1. Validate RAW inputs strictly
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 2. Convert input to DataFrame
    # --------------------------------------------------
    df = pd.DataFrame([data])

    # --------------------------------------------------
    # 3. Derive dataset-approved features
    # (same logic as training)
    # --------------------------------------------------
    df = derive_features(df)

    # --------------------------------------------------
    # 4. Model inference (ONCE)
    # --------------------------------------------------
    try:
        result = model.recommend_materials(df)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Model inference failed: {str(e)}"
        }), 500

    # --------------------------------------------------
    # 5. Add heuristic explanations
    # --------------------------------------------------
    result["decision_summary"] = generate_explanations(df.iloc[0])

    return jsonify(result)

# --------------------------------------------------
# Run server
# --------------------------------------------------
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
