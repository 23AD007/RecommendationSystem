from flask import Flask
from flask_cors import CORS
from src.api.routes.product_routes import product_bp

app = Flask(__name__)
CORS(app)

app.register_blueprint(product_bp, url_prefix="/api")

if __name__ == "__main__":
    app.run(port=5000)
