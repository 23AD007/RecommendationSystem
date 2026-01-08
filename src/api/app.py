from flask import Flask, jsonify
from src.config import Config
from src.data.load_data import test_db_connection

# Import blueprints
from src.api.routes.data_routes import data_bp
from src.api.routes.etl_routes import etl_bp
from src.api.routes.models_routes import models_bp
from src.api.routes.nlp_routes import nlp_bp
from src.api.routes.utils_routes import utils_bp
from src.api.routes.visualization_routes import visualization_bp
from src.api.routes.db_routes import db_bp
from src.api.routes.product_routes import product_bp
from src.api.routes.security_routes import security_bp, require_api_key

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Register blueprints
    app.register_blueprint(data_bp, url_prefix='/api/data')
    app.register_blueprint(etl_bp, url_prefix='/api/etl')
    app.register_blueprint(models_bp, url_prefix='/api/models')
    app.register_blueprint(nlp_bp, url_prefix='/api/nlp')
    app.register_blueprint(utils_bp, url_prefix='/api/utils')
    app.register_blueprint(visualization_bp, url_prefix='/api/visualization')
    app.register_blueprint(db_bp, url_prefix='/api/database')
    app.register_blueprint(product_bp, url_prefix='/api/product')
    app.register_blueprint(security_bp, url_prefix='/api/security')

    @app.route("/health", methods=["GET"])
    def health_check():
        db_status = test_db_connection()
        return jsonify({
            "status": "ok" if db_status else "error",
            "database": "connected" if db_status else "not connected"
        })

    @app.route("/api", methods=["GET"])
    def api_info():
        """API information and available endpoints"""
        endpoints = {
            "data": [
                "GET /api/data/test-db - Test database connection",
                "POST /api/data/export-training-dataset - Export training dataset",
                "GET /api/data/training-dataset-preview - Preview training dataset"
            ],
            "etl": [
                "GET /api/etl/load-raw-data - Load raw data from database",
                "POST /api/etl/clean-data - Clean loaded data",
                "POST /api/etl/feature-engineering - Apply feature engineering"
            ],
            "models": [
                "POST /api/models/evaluate/<model_name> - Evaluate specific model",
                "GET /api/models/rank-materials - Rank materials",
                "GET /api/models/available-models - List available models"
            ],
            "nlp": [
                "POST /api/nlp/extract-features - Extract features from text",
                "POST /api/nlp/extract-from-document - Extract from document text",
                "GET /api/nlp/process-sample-text - Process sample text"
            ],
            "utils": [
                "POST /api/utils/validate-schema - Validate data schema",
                "GET /api/utils/preprocess-data - Preprocess training data",
                "GET /api/utils/schema-requirements - Get schema requirements"
            ],
            "visualization": [
                "GET /api/visualization/model-comparison-plot - Generate model comparison plot",
                "GET /api/visualization/available-plots - List available plots"
            ],
            "database": [
                "GET /api/database/tables - List database tables",
                "POST /api/database/query - Execute SELECT query",
                "GET /api/database/table-info/<table_name> - Get table information"
            ],
            "product": [
                "POST /api/product/input - Handle product input and processing",
                "POST /api/product/recommend-materials - AI material recommendation",
                "POST /api/product/environmental-score - Environmental score computation"
            ],
            "security": [
                "GET /api/security/health - Secure health check (requires API key)",
                "POST /api/security/validate-key - Validate API key"
            ]
        }

        return jsonify({
            "status": "success",
            "message": "Packaging Recommendation System API",
            "endpoints": endpoints,
            "security": {
                "api_key_required": "Include 'X-API-Key' header with requests to secure endpoints",
                "default_key": "packaging-api-key-2024 (set API_KEY environment variable for production)"
            }
        })

    return app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
    # from waitress import serve
    # serve(app, host="0.0.0.0", port=5000)
