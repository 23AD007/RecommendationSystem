#!/usr/bin/env python3
"""
Demo script showing how to use the HTML frontend
"""

import webbrowser
import time
import subprocess
import sys
import os

def main():
    """Demo the HTML frontend"""
    print("🌟 Packaging Recommendation System - HTML Frontend Demo")
    print("=" * 60)

    # Check if backend is running
    try:
        import requests
        response = requests.get('http://localhost:5000/health', timeout=2)
        if response.status_code == 200:
            print("✅ Backend API is running on port 5000")
        else:
            print("⚠️  Backend API might not be responding correctly")
    except:
        print("❌ Backend API is not running on port 5000")
        print("   Please start the backend first: python -m src.api.app")
        return

    # Start the HTML frontend server
    print("\n🚀 Starting HTML frontend server...")
    try:
        # Run the server in background
        server_process = subprocess.Popen([
            sys.executable, 'run_html_frontend.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait a moment for server to start
        time.sleep(2)

        # Open browser
        print("📱 Opening browser to http://localhost:8000")
        webbrowser.open('http://localhost:8000')

        print("\n" + "=" * 60)
        print("🎯 Demo Instructions:")
        print("1. The web page should open in your browser")
        print("2. Fill in the product specifications form")
        print("3. Click 'Get Recommendations' to see AI suggestions")
        print("4. Try different values to see how recommendations change")
        print("\n🔧 Server is running in the background")
        print("   Press Ctrl+C to stop the demo")

        # Keep running until user stops
        try:
            server_process.wait()
        except KeyboardInterrupt:
            print("\n👋 Stopping demo...")
            server_process.terminate()
            server_process.wait()

    except Exception as e:
        print(f"❌ Error starting demo: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()