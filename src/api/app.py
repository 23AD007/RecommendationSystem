from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import os

from src.models.improved_recommendation_model import get_recommendation_model
from src.etl.feature_engineering import derive_features

app = Flask(__name__)
CORS(app)

model = get_recommendation_model()

REQUIRED_FIELDS = {
    "product_category": str,
    "fragility_score": (int, float),
    "durability_requirement": (int, float),
    "sustainability_priority": (int, float),
    "material_cost": (int, float),
    "max_packaging_cost": (int, float),
    "innovation_level": (int, float),
    "eco_pressure": (int, float),
    "cost_efficiency": (int, float),
}

def generate_explanations(row):
    return {
        "fragility": f"Fragility score {row['fragility_score']:.2f} influenced cushioning needs",
        "durability": f"Durability requirement {row['durability_requirement']:.2f} affected material strength",
        "sustainability": f"Sustainability priority {row['sustainability_priority']:.2f} favored eco materials",
        "cost": f"Material cost evaluated under budget {row['max_packaging_cost']}",
        "innovation": f"Innovation level {row['innovation_level']:.2f} encouraged modern materials",
    }

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/api/product/recommend-materials", methods=["POST"])
def recommend_materials():
    data = request.get_json()

    if not data:
        return jsonify({"status": "error", "message": "Empty request"}), 400

    for field, t in REQUIRED_FIELDS.items():
        if field not in data:
            return jsonify({"status": "error", "message": f"Missing field: {field}"}), 400
        if not isinstance(data[field], t):
            return jsonify({"status": "error", "message": f"Invalid type for {field}"}), 400

    df = pd.DataFrame([data])
    df = derive_features(df)

    prediction = model.predict(df)
    confidence = round(float(prediction.get("confidence", 0.0)) * 100, 2)

    response = {
        "status": "success",
        "confidence_score": confidence,
        "recommendations": [
            {
                "material": prediction.get("material", "Sustainable Packaging"),
                "confidence": confidence,
                "reason": "Selected using fallback-safe ML logic"
            }
        ],
        "decision_summary": generate_explanations(df.iloc[0]),
        "model_info": model.get_model_info()
    }

    return jsonify(response), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
