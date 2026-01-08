from flask import Blueprint, request, jsonify
from functools import wraps
import os

# Simple API key authentication
API_KEY = os.getenv('API_KEY', 'packaging-api-key-2024')

security_bp = Blueprint('security', __name__)

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key != API_KEY:
            return jsonify({
                'status': 'error',
                'message': 'Invalid or missing API key'
            }), 401
        return f(*args, **kwargs)
    return decorated_function

@security_bp.route('/health', methods=['GET'])
@require_api_key
def secure_health():
    """Secure health check endpoint"""
    return jsonify({
        'status': 'secure',
        'message': 'API is running with authentication',
        'timestamp': '2026-01-02'
    })

@security_bp.route('/validate-key', methods=['POST'])
def validate_api_key():
    """Validate API key"""
    data = request.get_json()
    provided_key = data.get('api_key') if data else None

    if provided_key == API_KEY:
        return jsonify({
            'status': 'valid',
            'message': 'API key is valid'
        })
    else:
        return jsonify({
            'status': 'invalid',
            'message': 'API key is invalid'
        }), 401