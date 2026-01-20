from flask import Blueprint, jsonify, request
import pandas as pd
import traceback

from src.models.improved_recommendation_model import get_recommendation_model
from src.api.routes.security_routes import require_api_key
from src.etl.feature_engineering import derive_features

product_bp = Blueprint("product", __name__)

# --------------------------------------------------
# RECOMMEND MATERIALS (MODEL-DRIVEN + EXPLAINABLE)
# --------------------------------------------------
@product_bp.route("/recommend-materials", methods=["POST"])
@require_api_key
def recommend_materials():
    try:
        data = request.get_json(silent=True)

        if not data:
            return jsonify({
                "status": "error",
                "message": "No product data provided"
            }), 400

        # -----------------------------
        # Required fields validation
        # -----------------------------
        required_fields = [
            "product_category",
            "fragility_score",
            "sustainability_priority",
            "material_cost"
        ]

        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({
                "status": "error",
                "message": f"Missing required fields: {missing}"
            }), 400

        # -----------------------------
        # Defaults for optional fields
        # -----------------------------
        data.setdefault("durability_requirement", 0.5)
        data.setdefault("max_packaging_cost", 100.0)
        data.setdefault("innovation_level", 3.0)

        # -----------------------------
        # FEATURE ENGINEERING (CRITICAL)
        # -----------------------------
        df_raw = pd.DataFrame([data])
        df_features = derive_features(df_raw)

        # -----------------------------
        # MODEL PREDICTION
        # -----------------------------
        model = get_recommendation_model()
        prediction = model.predict(df_features)

        confidence = float(prediction["confidence"])
        recommended = bool(prediction["recommended"])
        decision_summary = prediction.get("decision_summary", {})

        # -----------------------------
        # Recommendation level
        # -----------------------------
        if confidence >= 0.75:
            level = "Highly Recommended"
        elif confidence >= 0.55:
            level = "Recommended"
        elif confidence >= 0.4:
            level = "Moderate"
        else:
            level = "Not Recommended"

        # -----------------------------
        # MATERIAL RECOMMENDATION LOGIC
        # (Feature-aware, NOT static)
        # -----------------------------
        recommendations = []

        if recommended:
            # Sustainability-driven
            if data["sustainability_priority"] >= 0.7:
                recommendations.append({
                    "material": "Recycled Cardboard",
                    "confidence": round(confidence * 100, 1),
                    "reason": (
                        "High sustainability priority combined with good cost efficiency "
                        "makes recycled cardboard an optimal choice."
                    )
                })

            # Fragility-driven
            if data["fragility_score"] >= 0.7:
                recommendations.append({
                    "material": "Cork",
                    "confidence": round(confidence * 95, 1),
                    "reason": (
                        "High fragility score indicates the need for superior cushioning "
                        "and shock absorption."
                    )
                })

            # Innovation-driven
            if data["innovation_level"] >= 3:
                recommendations.append({
                    "material": "Bamboo Fiber",
                    "confidence": round(confidence * 90, 1),
                    "reason": (
                        "Higher innovation preference aligns well with renewable "
                        "and modern packaging materials."
                    )
                })

            # Fallback (balanced)
            if not recommendations:
                recommendations.append({
                    "material": "Sustainable Composite",
                    "confidence": round(confidence * 85, 1),
                    "reason": (
                        "Provides a balanced trade-off between durability, cost, "
                        "and sustainability when no single constraint dominates."
                    )
                })

        else:
            recommendations.append({
                "material": "Requirement Review Needed",
                "confidence": round((1 - confidence) * 100, 1),
                "reason": (
                    "Current product constraints conflict with sustainable "
                    "packaging objectives based on model evaluation."
                )
            })

        # -----------------------------
        # FINAL RESPONSE
        # -----------------------------
        return jsonify({
            "status": "success",
            "confidence_score": round(confidence, 3),
            "recommendation_level": level,
            "recommended": recommended,
            "decision_summary": decision_summary,   # ✅ EXPLAINABILITY
            "recommendations": recommendations,
            "model_info": model.get_model_info()
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
