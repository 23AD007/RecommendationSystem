from flask import Blueprint, jsonify, request
from src.utils.schema_validation import validate_schema, NLP_REQUIRED_COLUMNS
from src.utils.preprocessing import load_and_prepare_data
import pandas as pd

utils_bp = Blueprint('utils', __name__)

@utils_bp.route('/validate-schema', methods=['POST'])
def validate_data_schema():
    """Validate data schema against NLP requirements"""
    try:
        data = request.get_json()
        if not data or 'data' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Data field is required'
            }), 400

        df = pd.DataFrame(data['data'])
        validate_schema(df)

        return jsonify({
            'status': 'success',
            'message': 'Schema validation passed',
            'required_columns': list(NLP_REQUIRED_COLUMNS)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@utils_bp.route('/preprocess-data', methods=['GET'])
def preprocess_data():
    """Load and preprocess training data"""
    try:
        X, y = load_and_prepare_data("data/processed/training_dataset.csv")

        return jsonify({
            'status': 'success',
            'features_shape': X.shape,
            'target_shape': y.shape,
            'feature_columns': list(X.columns) if hasattr(X, 'columns') else None
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@utils_bp.route('/schema-requirements', methods=['GET'])
def get_schema_requirements():
    """Get NLP schema requirements"""
    return jsonify({
        'status': 'success',
        'required_columns': list(NLP_REQUIRED_COLUMNS)
    })