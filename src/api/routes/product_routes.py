from flask import Blueprint, request, jsonify
import pandas as pd
from src.models.improved_recommendation_model import get_recommendation_model

product_bp = Blueprint("product", __name__)

@product_bp.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    df = pd.DataFrame([data])

    model = get_recommendation_model()
    result = model.predict(df)

    return jsonify({
        "status": "success",
        **result
    })
