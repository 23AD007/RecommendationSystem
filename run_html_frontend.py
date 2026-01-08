#!/usr/bin/env python3
"""
Flask server to serve the HTML frontend for the Packaging Recommendation System
"""

from flask import Flask, send_from_directory
from flask_cors import CORS
from pathlib import Path

# Get the frontend directory path
frontend_dir = Path(__file__).parent / "src" / "frontend"

app = Flask(__name__, static_folder=str(frontend_dir))
CORS(app)  # Enable CORS for all routes

@app.route('/')
def index():
    return send_from_directory(frontend_dir, 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(frontend_dir, path)

def main():
    """Run the Flask server"""
    if not frontend_dir.exists():
        print(f"Error: Frontend directory not found at {frontend_dir}")
        return

    # Run the server
    port = 8000
    print(f"🚀 Frontend server running at http://localhost:{port}")
    print(f"📁 Serving files from: {frontend_dir}")
    print("📋 Open your browser and navigate to the URL above")
    print("🔧 Press Ctrl+C to stop the server")
    print("\nNote: Make sure the backend API is running on port 5000")

    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

if __name__ == "__main__":
    main()