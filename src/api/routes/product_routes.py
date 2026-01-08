from flask import Blueprint, jsonify, request
from src.models.improved_recommendation_model import get_recommendation_model
from src.api.routes.security_routes import require_api_key
from src.nlp.extract_features import extract_attributes
from src.etl.feature_engineering import derive_features
import pandas as pd

product_bp = Blueprint('product', __name__)

@product_bp.route('/input', methods=['POST'])
@require_api_key
def handle_product_input():
    """Handle product input and return processed data"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No product data provided'
            }), 400

        # Validate required fields
        required_fields = ['product_category', 'fragility_score', 'sustainability_priority',
                          'durability_requirement', 'max_packaging_cost', 'material_cost']

        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'status': 'error',
                'message': f'Missing required fields: {missing_fields}'
            }), 400

        # Add default values for missing optional fields
        data.setdefault('innovation_level', 0.5)  # Default innovation level

        # Create DataFrame from input
        df = pd.DataFrame([data])

        # Apply feature engineering
        processed_df = derive_features(df)

        # Return processed product data
        result = processed_df.iloc[0].to_dict()

        return jsonify({
            'status': 'success',
            'product_data': result,
            'processed_features': ['eco_pressure', 'cost_efficiency', 'durability_pressure']
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@product_bp.route('/recommend-materials', methods=['POST'])
@require_api_key
def recommend_materials():
    """AI-powered material recommendation using improved ML model"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No product data provided'
            }), 400

        # Validate required fields (more flexible now with missing value handling)
        essential_fields = ['product_category', 'fragility_score', 'sustainability_priority', 'material_cost']
        missing_essential = [field for field in essential_fields if field not in data]
        if missing_essential:
            return jsonify({
                'status': 'error',
                'message': f'Missing essential fields: {missing_essential}'
            }), 400

        # Optional fields will be handled by feature engineering
        optional_fields = ['durability_requirement', 'max_packaging_cost', 'innovation_level']
        for field in optional_fields:
            if field not in data:
                # Add reasonable defaults for missing optional fields
                if field == 'durability_requirement':
                    data[field] = 0.5
                elif field == 'max_packaging_cost':
                    data[field] = 100.0
                elif field == 'innovation_level':
                    data[field] = 3

        # Get the improved recommendation model
        model = get_recommendation_model()

        # Make prediction
        prediction_result = model.predict(data)

        # Determine recommendation level based on confidence
        confidence = prediction_result['confidence']
        if confidence >= 0.8:
            level = "Highly Recommended"
        elif confidence >= 0.6:
            level = "Recommended"
        elif confidence >= 0.4:
            level = "Moderate"
        else:
            level = "Not Recommended"

        # Generate material recommendations based on prediction
        recommendations = []
        if prediction_result['recommended']:
            if confidence >= 0.7:
                recommendations.append({
                    "material": "Recycled Cardboard",
                    "confidence": round(confidence * 100, 1),
                    "reason": "High recommendation confidence with excellent sustainability profile"
                })
            if confidence >= 0.5:
                recommendations.append({
                    "material": "Biodegradable Plastic",
                    "confidence": round(confidence * 90, 1),
                    "reason": "Good balance of durability and environmental impact"
                })
            recommendations.append({
                "material": "Sustainable Composite",
                "confidence": round(confidence * 85, 1),
                "reason": "Modern material with balanced properties"
            })
        else:
            recommendations.append({
                "material": "Review Requirements",
                "confidence": round((1 - confidence) * 100, 1),
                "reason": "Current specifications may not be optimal for sustainable packaging"
            })

        # Get top similar materials from training data (optional enhancement)
        try:
            training_df = pd.read_csv("data/processed/training_dataset.csv")
            # Find similar products based on key features
            similar_products = training_df[
                (training_df['recommended'] == int(prediction_result['recommended'])) &
                (abs(training_df['fragility_score'] - data.get('fragility_score', 0.5)) < 0.2) &
                (abs(training_df['sustainability_priority'] - data.get('sustainability_priority', 0.5)) < 0.2)
            ].head(3)

            if len(similar_products) > 0:
                similar_materials = similar_products[['fragility_score', 'sustainability_priority',
                                                    'durability_requirement', 'material_cost']].to_dict('records')
            else:
                similar_materials = []
        except Exception:
            similar_materials = []

        return jsonify({
            'status': 'success',
            'product_score': round(confidence, 3),
            'recommendation_level': level,
            'recommended': prediction_result['recommended'],
            'confidence_score': round(confidence, 3),
            'recommendations': recommendations,
            'similar_materials': similar_materials,
            'model_info': model.get_model_info()
        }), 200

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@product_bp.route('/environmental-score', methods=['POST'])
@require_api_key
def compute_environmental_score():
    """Compute environmental score for product/material combination"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400

        # Extract environmental features
        text_description = data.get('description', '')
        features = extract_attributes(text_description)

        # Compute environmental score based on multiple factors
        eco_pressure = features.get('eco_pressure', 0.5)
        sustainability_priority = data.get('sustainability_priority', 0.5)

        # Environmental score calculation (higher is better for environment)
        environmental_score = (1 - eco_pressure) * 0.6 + sustainability_priority * 0.4

        # Categorize environmental impact
        if environmental_score > 0.8:
            impact_level = 'Excellent'
            recommendation = 'Highly recommended for eco-friendly packaging'
        elif environmental_score > 0.6:
            impact_level = 'Good'
            recommendation = 'Good environmental choice'
        elif environmental_score > 0.4:
            impact_level = 'Moderate'
            recommendation = 'Consider more sustainable alternatives'
        else:
            impact_level = 'Poor'
            recommendation = 'High environmental impact - not recommended'

        return jsonify({
            'status': 'success',
            'environmental_score': float(environmental_score),
            'impact_level': impact_level,
            'recommendation': recommendation,
            'factors': {
                'eco_pressure': float(eco_pressure),
                'sustainability_priority': float(sustainability_priority)
            }
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500