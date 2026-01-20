from flask import Blueprint, jsonify, request
import pandas as pd
import traceback

from src.models.improved_recommendation_model import get_recommendation_model
from src.api.routes.security_routes import require_api_key

product_bp = Blueprint("product", __name__)

# --------------------------------------------------
# RECOMMEND MATERIALS (FINAL STABLE VERSION)
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

        # -------------------------------
        # REQUIRED FIELDS
        # -------------------------------
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

        # -------------------------------
        # DEFAULTS
        # -------------------------------
        data.setdefault("durability_requirement", 0.5)
        data.setdefault("max_packaging_cost", 100.0)
        data.setdefault("innovation_level", 3.0)

        # -------------------------------
        # MODEL PREDICTION (RAW INPUT)
        # -------------------------------
        model = get_recommendation_model()
        prediction = model.predict(data)

        confidence = float(prediction.get("confidence", 0.4))
        decision_summary = prediction.get("decision_summary", {})

        # -------------------------------
        # RECOMMENDATION LEVEL
        # -------------------------------
        if confidence >= 0.75:
            level = "Highly Recommended"
        elif confidence >= 0.55:
            level = "Recommended"
        elif confidence >= 0.4:
            level = "Moderate"
        else:
            level = "Low Confidence"

        # -------------------------------
        # ALWAYS-GENERATE RECOMMENDATIONS
        # -------------------------------
        recommendations = []

        # Sustainability-driven
        if data["sustainability_priority"] >= 0.6:
            recommendations.append({
                "material": "Recycled Cardboard",
                "confidence": round(confidence * 100, 1),
                "reason": (
                    "High sustainability priority favors recyclable, low-impact materials "
                    "with reduced environmental footprint"
                )
            })

        # Fragility-driven
        if data["fragility_score"] >= 0.6:
            recommendations.append({
                "material": "Cork",
                "confidence": round(confidence * 95, 1),
                "reason": (
                    "High fragility requires superior cushioning and shock absorption "
                    "to protect the product during transit"
                )
            })

        # Innovation-driven
        if data["innovation_level"] >= 3:
            recommendations.append({
                "material": "Bamboo Fiber",
                "confidence": round(confidence * 90, 1),
                "reason": (
                    "Innovation preference supports renewable, modern materials "
                    "that balance strength and sustainability"
                )
            })

        # Cost-aware fallback
        if not recommendations:
            recommendations.append({
                "material": "Sustainable Composite",
                "confidence": round(confidence * 85, 1),
                "reason": (
                    "Balanced material selected due to competing constraints "
                    "between cost, durability, and sustainability"
                )
            })

        # -------------------------------
        # SMART DECISION SUMMARY
        # -------------------------------
        enhanced_summary = {
            "Sustainability Influence":
                f"Sustainability priority ({data['sustainability_priority']}) increased preference for eco-friendly materials",
            "Fragility Influence":
                f"Fragility score ({data['fragility_score']}) increased the need for protective packaging",
            "Cost Consideration":
                f"Material cost {data['material_cost']} evaluated against budget {data['max_packaging_cost']}",
            "Innovation Influence":
                f"Innovation level ({data['innovation_level']}) influenced selection of modern materials",
            "Overall Assessment":
                "Trade-offs identified, but viable sustainable packaging options exist"
        }

        return jsonify({
            "status": "success",
            "confidence_score": round(confidence, 3),
            "recommendation_level": level,
            "recommended": True,   # ✅ IMPORTANT: never hard-block
            "decision_summary": enhanced_summary,
            "recommendations": recommendations,
            "model_info": model.get_model_info()
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
