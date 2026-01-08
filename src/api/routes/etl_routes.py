from flask import Blueprint, jsonify, request
from src.etl.data_cleaning import load_raw, clean_data
from src.etl.feature_engineering import derive_features
import pandas as pd

etl_bp = Blueprint('etl', __name__)

@etl_bp.route('/load-raw-data', methods=['GET'])
def load_raw_data():
    """Load raw data from database"""
    try:
        df = load_raw()
        preview = df.head(10).to_dict('records')
        return jsonify({
            'status': 'success',
            'data': preview,
            'total_rows': len(df),
            'columns': list(df.columns)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@etl_bp.route('/clean-data', methods=['POST'])
def clean_data_endpoint():
    """Clean the loaded data"""
    try:
        df = load_raw()
        cleaned_df = clean_data(df)
        preview = cleaned_df.head(10).to_dict('records')
        return jsonify({
            'status': 'success',
            'data': preview,
            'original_shape': df.shape,
            'cleaned_shape': cleaned_df.shape
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@etl_bp.route('/feature-engineering', methods=['POST'])
def feature_engineering():
    """Apply feature engineering to data"""
    try:
        df = load_raw()
        cleaned_df = clean_data(df)
        engineered_df = derive_features(cleaned_df)
        preview = engineered_df.head(10).to_dict('records')
        return jsonify({
            'status': 'success',
            'data': preview,
            'features_added': ['eco_pressure', 'cost_efficiency', 'durability_pressure']
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500