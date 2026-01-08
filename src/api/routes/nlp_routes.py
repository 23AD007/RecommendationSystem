from flask import Blueprint, jsonify, request
from src.nlp.extract_features import extract_attributes
from src.nlp.ingest_documents import extract_attributes as extract_from_text, estimate_confidence
import json

nlp_bp = Blueprint('nlp', __name__)

@nlp_bp.route('/extract-features', methods=['POST'])
def extract_features():
    """Extract features from text input"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Text field is required'
            }), 400

        text = data['text']
        features = extract_attributes(text)

        return jsonify({
            'status': 'success',
            'features': features
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@nlp_bp.route('/extract-from-document', methods=['POST'])
def extract_from_document():
    """Extract attributes from document text"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Text field is required'
            }), 400

        text = data['text']
        features = extract_from_text(text)
        confidence = estimate_confidence(features)

        return jsonify({
            'status': 'success',
            'features': features,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@nlp_bp.route('/process-sample-text', methods=['GET'])
def process_sample():
    """Process sample text for demonstration"""
    sample_text = """
    Product Category: Electronics
    Fragility: 0.7
    Sustainability: 0.8
    Durability: 0.75
    Max Packaging Cost: 120
    Material Cost: 60
    Innovation Level: Improved
    """

    features = extract_attributes(sample_text)

    return jsonify({
        'status': 'success',
        'sample_text': sample_text.strip(),
        'extracted_features': features
    })