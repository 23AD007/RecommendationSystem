from flask import Blueprint, jsonify
from src.data.load_data import test_db_connection, get_engine
from src.data.export_training_dataset import main as export_training_dataset
import pandas as pd

data_bp = Blueprint('data', __name__)

@data_bp.route('/test-db', methods=['GET'])
def test_db():
    """Test database connection"""
    status = test_db_connection()
    return jsonify({
        'database_connected': status,
        'status': 'success' if status else 'error'
    })

@data_bp.route('/export-training-dataset', methods=['POST'])
def export_dataset():
    """Export training dataset to CSV"""
    try:
        export_training_dataset()
        return jsonify({
            'status': 'success',
            'message': 'Training dataset exported successfully'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@data_bp.route('/training-dataset-preview', methods=['GET'])
def preview_dataset():
    """Get preview of training dataset"""
    try:
        df = pd.read_csv('data/processed/training_dataset.csv')
        preview = df.head(10).to_dict('records')
        return jsonify({
            'status': 'success',
            'data': preview,
            'shape': df.shape
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500