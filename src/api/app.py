from flask import Flask
from flask_cors import CORS
import os

from src.api.routes.product_routes import product_bp
from src.api.routes.models_routes import models_bp
from src.api.routes.data_routes import data_bp
from src.api.routes.db_routes import db_bp
from src.api.routes.etl_routes import etl_bp
from src.api.routes.nlp_routes import nlp_bp
from src.api.routes.security_routes import security_bp
from src.api.routes.utils_routes import utils_bp
from src.api.routes.visualization_routes import visualization_bp

app = Flask(__name__)
CORS(app)

# Register blueprints
app.register_blueprint(product_bp, url_prefix='/api')
app.register_blueprint(models_bp, url_prefix='/api')
app.register_blueprint(data_bp, url_prefix='/api')
app.register_blueprint(db_bp, url_prefix='/api')
app.register_blueprint(etl_bp, url_prefix='/api')
app.register_blueprint(nlp_bp, url_prefix='/api')
app.register_blueprint(security_bp, url_prefix='/api')
app.register_blueprint(utils_bp, url_prefix='/api')
app.register_blueprint(visualization_bp, url_prefix='/api')

# Health check
@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
