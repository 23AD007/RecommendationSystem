from flask import Blueprint, jsonify, request
from src.models.evaluate_models import evaluate_model
from src.models.rank_materials import rank_materials
from src.utils.preprocessing import load_and_prepare_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
# from catboost import CatBoostClassifier
import pandas as pd

models_bp = Blueprint('models', __name__)

@models_bp.route('/evaluate/<model_name>', methods=['POST'])
def evaluate_single_model(model_name):
    """Evaluate a specific model"""
    try:
        X, y = load_and_prepare_data("data/processed/training_dataset.csv")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        models = {
            'logistic_regression': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(random_state=42),
            'xgboost': XGBClassifier(random_state=42),
            'lightgbm': lgb.LGBMClassifier(random_state=42, verbose=-1)
            # 'catboost': CatBoostClassifier(random_state=42, verbose=False)
        }

        if model_name not in models:
            return jsonify({
                'status': 'error',
                'message': f'Model {model_name} not found. Available: {list(models.keys())}'
            }), 400

        metrics = evaluate_model(model_name, models[model_name], X_train, X_test, y_train, y_test)

        return jsonify({
            'status': 'success',
            'model': model_name,
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@models_bp.route('/rank-materials', methods=['GET'])
def rank_materials_endpoint():
    """Rank materials based on recommendation scores"""
    try:
        # This function prints results, so we'll capture it differently
        # For now, return a success message
        rank_materials()
        return jsonify({
            'status': 'success',
            'message': 'Material ranking completed. Check console output.'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@models_bp.route('/available-models', methods=['GET'])
def available_models():
    """Get list of available models"""
    models = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']
    return jsonify({
        'status': 'success',
        'models': models
    })