#!/usr/bin/env python3
"""
Test script to verify the Packaging Recommendation System is working
"""

import urllib.request
import json
import sys

def test_backend_api():
    """Test the backend API health and recommendations"""
    print("🔍 Testing Packaging Recommendation System...")
    print()

    # Test 1: Backend API Health
    print("1. Testing Backend API Health...")
    try:
        with urllib.request.urlopen('http://localhost:5000/health') as response:
            data = json.loads(response.read().decode())
            if data.get('status') == 'ok':
                print("   ✅ Backend API: OK (Database connected)")
                return True
            else:
                print("   ❌ Backend API: Database not connected")
                return False
    except Exception as e:
        print(f"   ❌ Backend API: Failed - {e}")
        return False

def test_api_recommendations():
    """Test the API recommendation endpoint"""
    print("2. Testing API Recommendation Endpoint...")
    try:
        test_data = {
            'product_category': 'electronics',
            'fragility_score': 0.8,
            'sustainability_priority': 0.9,
            'material_cost': 45.0
        }
        req = urllib.request.Request(
            'http://localhost:5000/api/product/recommend-materials',
            data=json.dumps(test_data).encode('utf-8'),
            headers={'Content-Type': 'application/json', 'X-API-Key': 'packaging-api-key-2024'}
        )
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode())
            if result.get('status') == 'success':
                print("   ✅ API Recommendations: Working")
                return True
            else:
                print("   ❌ API Recommendations: Failed")
                return False
    except Exception as e:
        print(f"   ❌ API Recommendations: Failed - {e}")
        return False

def main():
    """Run all tests"""
    backend_ok = test_backend_api()
    api_ok = test_api_recommendations()

    print()
    if backend_ok and api_ok:
        print("🎉 All systems operational!")
        print("📱 To use the web interface:")
        print("   1. Start the HTML frontend: python run_html_frontend.py")
        print("   2. Open http://localhost:8000 in your browser")
        return 0
    else:
        print("❌ Some systems are not working. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())