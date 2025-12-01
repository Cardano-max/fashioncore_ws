#!/usr/bin/env python3
"""
Test script for 11za webhook integration
Tests the webhook endpoint with sample messages
"""

import requests
import json
import sys

# Configuration
BASE_URL = "http://localhost:8080"  # Change to your Railway URL for production testing
WEBHOOK_ENDPOINT = f"{BASE_URL}/webhook"

def test_health_check():
    """Test the health check endpoint"""
    print("\n🔍 Testing health check endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed:", response.json())
            return True
        else:
            print("❌ Health check failed:", response.status_code)
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_landing_page():
    """Test the landing page"""
    print("\n🔍 Testing landing page...")
    try:
        response = requests.get(BASE_URL, timeout=5)
        if response.status_code == 200:
            print("✅ Landing page loaded successfully")
            return True
        else:
            print("❌ Landing page failed:", response.status_code)
            return False
    except Exception as e:
        print(f"❌ Landing page error: {e}")
        return False

def test_webhook_verification():
    """Test webhook verification (GET request)"""
    print("\n🔍 Testing webhook verification...")
    try:
        params = {
            'hub.mode': 'subscribe',
            'hub.verify_token': '1122',
            'hub.challenge': 'test_challenge_12345'
        }
        response = requests.get(WEBHOOK_ENDPOINT, params=params, timeout=5)
        if response.status_code == 200 and response.text == 'test_challenge_12345':
            print("✅ Webhook verification passed")
            return True
        else:
            print("❌ Webhook verification failed:", response.status_code, response.text)
            return False
    except Exception as e:
        print(f"❌ Webhook verification error: {e}")
        return False

def test_text_message():
    """Test handling of text message via webhook"""
    print("\n🔍 Testing text message handling...")
    try:
        # Sample webhook payload for text message
        payload = {
            "messages": [
                {
                    "from": "919999999999",
                    "type": "text",
                    "text": "start",
                    "timestamp": "1234567890"
                }
            ]
        }

        response = requests.post(
            WEBHOOK_ENDPOINT,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=5
        )

        if response.status_code == 200:
            print("✅ Text message webhook processed successfully")
            return True
        else:
            print("❌ Text message webhook failed:", response.status_code)
            print("Response:", response.text)
            return False
    except Exception as e:
        print(f"❌ Text message webhook error: {e}")
        return False

def test_image_message():
    """Test handling of image message via webhook"""
    print("\n🔍 Testing image message handling...")
    try:
        # Sample webhook payload for image message
        payload = {
            "messages": [
                {
                    "from": "919999999999",
                    "type": "image",
                    "image": {
                        "url": "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=500",
                        "caption": "Test person image"
                    },
                    "timestamp": "1234567891"
                }
            ]
        }

        response = requests.post(
            WEBHOOK_ENDPOINT,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=5
        )

        if response.status_code == 200:
            print("✅ Image message webhook processed successfully")
            return True
        else:
            print("❌ Image message webhook failed:", response.status_code)
            print("Response:", response.text)
            return False
    except Exception as e:
        print(f"❌ Image message webhook error: {e}")
        return False

def test_garment_selection():
    """Test garment selection endpoint"""
    print("\n🔍 Testing garment selection...")
    try:
        garment_id = "test_garment_123"
        garment_url = "https://images.unsplash.com/photo-1594938298603-c8148c4dae35?w=500"

        response = requests.get(
            f"{BASE_URL}/select-garment/{garment_id}",
            params={'url': garment_url},
            timeout=5
        )

        if response.status_code == 200:
            data = response.json()
            if data.get('success') and 'whatsapp_url' in data:
                print("✅ Garment selection successful")
                print(f"   WhatsApp URL: {data['whatsapp_url']}")
                return True

        print("❌ Garment selection failed:", response.status_code)
        return False
    except Exception as e:
        print(f"❌ Garment selection error: {e}")
        return False

def test_admin_dashboard():
    """Test admin dashboard"""
    print("\n🔍 Testing admin dashboard...")
    try:
        response = requests.get(f"{BASE_URL}/admin", timeout=5)
        if response.status_code == 200:
            print("✅ Admin dashboard loaded successfully")
            return True
        else:
            print("❌ Admin dashboard failed:", response.status_code)
            return False
    except Exception as e:
        print(f"❌ Admin dashboard error: {e}")
        return False

def run_all_tests():
    """Run all tests and report results"""
    print("="*60)
    print("🧪 FashionCore 11za Integration Test Suite")
    print("="*60)

    tests = [
        ("Health Check", test_health_check),
        ("Landing Page", test_landing_page),
        ("Webhook Verification", test_webhook_verification),
        ("Text Message Webhook", test_text_message),
        ("Image Message Webhook", test_image_message),
        ("Garment Selection", test_garment_selection),
        ("Admin Dashboard", test_admin_dashboard),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Application is ready.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    if len(sys.argv) > 1:
        BASE_URL = sys.argv[1]
        WEBHOOK_ENDPOINT = f"{BASE_URL}/webhook"
        print(f"Testing against: {BASE_URL}")

    exit_code = run_all_tests()
    sys.exit(exit_code)
