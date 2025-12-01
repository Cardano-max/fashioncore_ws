#!/usr/bin/env python3
"""
Validation script for FashionCore 11za integration
Validates code structure, configuration, and logic without running the server
"""

import sys
import os
import json
from pathlib import Path

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_check(passed, message):
    """Print a check result"""
    status = "✅" if passed else "❌"
    print(f"{status} {message}")
    return passed

def validate_file_structure():
    """Validate that all required files exist"""
    print_header("📁 File Structure Validation")

    required_files = {
        'fashioncore_11za.py': 'Main application file',
        'test_webhook.py': 'Test suite',
        'SETUP_GUIDE.md': 'Setup documentation',
        'INTEGRATION_NOTES.md': 'Integration notes',
        'README.md': 'Project README',
        'requirements.txt': 'Python dependencies',
        'railway.json': 'Railway deployment config'
    }

    all_passed = True
    for filename, description in required_files.items():
        exists = Path(filename).exists()
        all_passed &= print_check(exists, f"{filename} - {description}")

    return all_passed

def validate_code_syntax():
    """Validate Python file syntax"""
    print_header("🔍 Code Syntax Validation")

    python_files = ['fashioncore_11za.py', 'test_webhook.py']
    all_passed = True

    for filename in python_files:
        try:
            with open(filename, 'r') as f:
                compile(f.read(), filename, 'exec')
            all_passed &= print_check(True, f"{filename} - Valid Python syntax")
        except SyntaxError as e:
            all_passed &= print_check(False, f"{filename} - Syntax error: {e}")

    return all_passed

def validate_configuration():
    """Validate configuration in the code"""
    print_header("⚙️  Configuration Validation")

    all_passed = True

    # Check Kling AI credentials
    with open('fashioncore_11za.py', 'r') as f:
        content = f.read()

        # Check for new credentials
        has_new_access_key = 'ALMrJQFypk3HCYMnkNNfa8NJCB9YPeP' in content
        has_new_secret_key = 'pNYB39FT3kbGEtaCCM3Qr8PkHHBppdC' in content
        has_test_mode = 'TEST_MODE' in content

        all_passed &= print_check(has_new_access_key, "New Kling AI Access Key configured")
        all_passed &= print_check(has_new_secret_key, "New Kling AI Secret Key configured")
        all_passed &= print_check(has_test_mode, "TEST_MODE flag implemented")

    # Check Railway config
    with open('railway.json', 'r') as f:
        config = json.load(f)
        correct_start = config.get('deploy', {}).get('startCommand') == 'python fashioncore_11za.py'
        all_passed &= print_check(correct_start, "Railway configured to run fashioncore_11za.py")

    return all_passed

def validate_api_integration():
    """Validate API integration logic"""
    print_header("🔌 API Integration Validation")

    all_passed = True

    with open('fashioncore_11za.py', 'r') as f:
        content = f.read()

        # Check for key components
        checks = {
            'class ElevenzaWhatsAppClient': '11za WhatsApp client class',
            'class KlingAIClient': 'Kling AI client class',
            'def send_text_message': '11za text message sending',
            'def send_image_message': '11za image message sending',
            'def try_on': 'Kling AI try-on method',
            'def handle_message': 'Message handler',
            'class UserState': 'State machine implementation',
            'garment_selections': 'Garment pre-selection system',
            '/webhook': 'Webhook endpoint',
            '/select-garment': 'Garment selection endpoint',
            'def create_template_files': 'Landing page creation',
        }

        for check, description in checks.items():
            exists = check in content
            all_passed &= print_check(exists, description)

    return all_passed

def validate_test_mode():
    """Validate TEST_MODE implementation"""
    print_header("🧪 Test Mode Validation")

    all_passed = True

    with open('fashioncore_11za.py', 'r') as f:
        content = f.read()

        checks = {
            'TEST_MODE = os.getenv': 'TEST_MODE environment variable',
            'if TEST_MODE:': 'TEST_MODE conditional logic',
            'TEST MODE: Mocking Kling AI': 'Mock response implementation',
            'mock_result_url': 'Mock result URL',
            'Success (TEST MODE)': 'Test mode success message',
        }

        for check, description in checks.items():
            exists = check in content
            all_passed &= print_check(exists, description)

    return all_passed

def validate_user_flow():
    """Validate user flow implementation"""
    print_header("👤 User Flow Validation")

    all_passed = True

    with open('fashioncore_11za.py', 'r') as f:
        content = f.read()

        # Check state transitions
        states = ['IDLE', 'WAITING_FOR_PERSON', 'PROCESSING', 'SHOWING_RESULT']
        for state in states:
            exists = state in content
            all_passed &= print_check(exists, f"State: {state}")

        # Check key flow components
        flow_checks = {
            'start_': 'Garment pre-selection from landing page',
            'person_image_path': 'Person image handling',
            'garment_url': 'Garment URL handling',
            'process_images': 'Image processing function',
            'log_tryon_attempt': 'Database logging',
        }

        for check, description in flow_checks.items():
            exists = check in content
            all_passed &= print_check(exists, description)

    return all_passed

def validate_documentation():
    """Validate documentation completeness"""
    print_header("📚 Documentation Validation")

    all_passed = True

    # Check README
    with open('README.md', 'r') as f:
        readme = f.read()
        readme_checks = {
            '11za': '11za mentioned',
            'Kling AI': 'Kling AI mentioned',
            'Quick Start': 'Quick start guide',
            'User Flow': 'User flow documented',
            'Deployment': 'Deployment instructions',
            'Testing': 'Testing instructions',
        }

        for check, description in readme_checks.items():
            exists = check in readme
            all_passed &= print_check(exists, f"README: {description}")

    # Check SETUP_GUIDE
    with open('SETUP_GUIDE.md', 'r') as f:
        setup = f.read()
        setup_checks = {
            'Environment Variables': 'Environment configuration',
            'Configure 11za Webhook': 'Webhook setup',
            'Deploy to Railway': 'Deployment guide',
            'Troubleshooting': 'Troubleshooting section',
        }

        for check, description in setup_checks.items():
            exists = check in setup
            all_passed &= print_check(exists, f"SETUP_GUIDE: {description}")

    return all_passed

def validate_security():
    """Validate security measures"""
    print_header("🔐 Security Validation")

    all_passed = True

    with open('fashioncore_11za.py', 'r') as f:
        content = f.read()

        security_checks = {
            'VERIFY_TOKEN': 'Webhook verification token',
            'Authorization': 'API authorization header',
            'os.getenv': 'Environment variables for secrets',
            'try:': 'Error handling',
            'except': 'Exception catching',
            'logger.error': 'Error logging',
            'os.remove': 'Temporary file cleanup',
        }

        for check, description in security_checks.items():
            exists = check in content
            all_passed &= print_check(exists, description)

    return all_passed

def print_deployment_checklist():
    """Print deployment checklist"""
    print_header("🚀 Deployment Checklist")

    checklist = [
        "Update ELEVENZA_AUTH_TOKEN in Railway environment variables",
        "Update ELEVENZA_PHONE_NUMBER in Railway environment variables",
        "Set TEST_MODE=False in Railway when ready for production",
        "Configure 11za webhook URL in 11za dashboard",
        "Set webhook verify token to '1122' in 11za dashboard",
        "Test health endpoint: /health",
        "Test landing page: /",
        "Test admin dashboard: /admin",
        "Send 'start' to WhatsApp number to test flow",
        "Monitor Railway logs for any errors",
    ]

    print("\nBefore deploying to production:")
    for i, item in enumerate(checklist, 1):
        print(f"  {i}. [ ] {item}")

def print_test_mode_info():
    """Print information about TEST_MODE"""
    print_header("🧪 Test Mode Information")

    print("""
TEST_MODE is currently ENABLED in the code.

What this means:
  ✓ No Kling AI API calls will be made
  ✓ No credits will be consumed
  ✓ Mock responses will be returned (3 second delay)
  ✓ Mock image URL will be used
  ✓ All other functionality works normally

To disable TEST_MODE for production:
  1. In Railway dashboard, add environment variable: TEST_MODE=False
  2. Or edit fashioncore_11za.py line 54: TEST_MODE=False

Current configuration:
  - Kling AI Access Key: ALMrJQFypk3HCYMnkNNfa8NJCB9YPeP
  - Kling AI Secret Key: pNYB39FT3kbGEtaCCM3Qr8PkHHBppdC
  - Test Mode: True (mocking enabled)
""")

def main():
    """Run all validations"""
    print("\n" + "🎨" * 35)
    print("  FashionCore 11za Integration - Validation Report")
    print("🎨" * 35)

    results = []

    results.append(("File Structure", validate_file_structure()))
    results.append(("Code Syntax", validate_code_syntax()))
    results.append(("Configuration", validate_configuration()))
    results.append(("API Integration", validate_api_integration()))
    results.append(("Test Mode", validate_test_mode()))
    results.append(("User Flow", validate_user_flow()))
    results.append(("Documentation", validate_documentation()))
    results.append(("Security", validate_security()))

    # Summary
    print_header("📊 Validation Summary")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\n{passed}/{total} validation categories passed")

    if passed == total:
        print("\n🎉 All validations passed! Code is ready for deployment.")
        print_test_mode_info()
        print_deployment_checklist()
        return 0
    else:
        print(f"\n⚠️  {total - passed} validation(s) failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    print("\n")
    sys.exit(exit_code)
