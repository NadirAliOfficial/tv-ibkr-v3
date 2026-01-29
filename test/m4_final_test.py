#!/usr/bin/env python3
"""
TV-IBKR Test Suite
M4: Paper Trading Validation and System Testing

Tests all 4 milestones:
- M1: Security and webhook ingestion
- M2: Risk engine and order execution
- M3: Database persistence and reconciliation
- M4: Admin endpoints and monitoring

Run with: python test_paper_trading_fixed.py
"""

import json
import hmac
import hashlib
import time
import requests
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, Any
from pathlib import Path

# ============================================================================
# CONFIGURATION - Load from .env file
# ============================================================================

def load_env_file():
    """Load .env file and return configuration"""
    env_path = Path(__file__).parent.parent / '.env'
    
    config = {
        'WEBHOOK_SECRET': None,
        'ADMIN_API_KEY': None
    }
    
    if not env_path.exists():
        print(f"⚠️  .env file not found at {env_path}")
        print("Please create .env file with WEBHOOK_SECRET and ADMIN_API_KEY")
        return config
    
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key in config:
                    config[key] = value
    
    return config

config = load_env_file()

BASE_URL = os.getenv('TEST_BASE_URL', 'http://localhost:8000')
WEBHOOK_SECRET = config.get('WEBHOOK_SECRET') or os.getenv('WEBHOOK_SECRET')
ADMIN_API_KEY = config.get('ADMIN_API_KEY') or os.getenv('ADMIN_API_KEY')

# Validate configuration
if not WEBHOOK_SECRET or WEBHOOK_SECRET == 'your-super-secret-key-here-change-this':
    print("❌ ERROR: WEBHOOK_SECRET not configured in .env file")
    print("Please set a valid WEBHOOK_SECRET in your .env file")
    exit(1)

if not ADMIN_API_KEY or ADMIN_API_KEY == 'your-admin-api-key-change-this':
    print("❌ ERROR: ADMIN_API_KEY not configured in .env file")
    print("Please set a valid ADMIN_API_KEY in your .env file")
    exit(1)

print(f"✅ Configuration loaded successfully")
print(f"   Base URL: {BASE_URL}")
print(f"   Webhook Secret: {WEBHOOK_SECRET[:8]}...{WEBHOOK_SECRET[-4:]}")
print(f"   Admin API Key: {ADMIN_API_KEY[:8]}...{ADMIN_API_KEY[-4:]}")
print()

# ============================================================================
# TEST HELPERS
# ============================================================================

class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.message = ""
        
    def pass_test(self, message: str = ""):
        self.passed = True
        self.message = message
        print(f"✅ {self.name}: PASSED {message}")
        
    def fail_test(self, message: str):
        self.passed = False
        self.message = message
        print(f"❌ {self.name}: FAILED - {message}")

def create_webhook_payload(ticker: str, action: str, quantity: int = 10) -> dict:
    """Create a properly signed webhook payload"""
    payload = {
        "ticker": ticker,
        "action": action,
        "quantity": quantity,
        "order_type": "MARKET",
        "strategy": "test_strategy",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    # Calculate signature - must match exact format sent to webhook
    payload_json = json.dumps(payload, separators=(',', ':'))
    signature = hmac.new(
        WEBHOOK_SECRET.encode(),
        payload_json.encode(),
        hashlib.sha256
    ).hexdigest()
    
    payload['signature'] = signature
    return payload

# ============================================================================
# MILESTONE 1 TESTS: Core Ingestion & Security
# ============================================================================

def test_m1_health_endpoint():
    """Test basic health endpoint"""
    result = TestResult("M1: Health Endpoint")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'healthy':
                result.pass_test()
            else:
                result.fail_test("Unexpected health status")
        else:
            result.fail_test(f"HTTP {response.status_code}")
    except Exception as e:
        result.fail_test(str(e))
    return result

def test_m1_webhook_valid_signature():
    """Test webhook with valid HMAC signature"""
    result = TestResult("M1: Valid Signature")
    try:
        payload = create_webhook_payload("TSLA", "BUY", 10)
        response = requests.post(f"{BASE_URL}/webhook", json=payload, timeout=35)
        
        # Should get 200 (success) or 409 (duplicate) or 200 with "rejected" status (kill switch active)
        if response.status_code == 200:
            data = response.json()
            # If kill switch is active, trade will be rejected but signature was valid
            if data.get('status') in ['executed', 'rejected', 'validated']:
                result.pass_test(f"Status: {data.get('status')}")
            else:
                result.fail_test(f"Unexpected status: {data.get('status')}")
        elif response.status_code == 409:
            result.pass_test("Duplicate (expected)")
        else:
            result.fail_test(f"HTTP {response.status_code}: {response.text}")
    except Exception as e:
        result.fail_test(str(e))
    return result

def test_m1_webhook_invalid_signature():
    """Test webhook with invalid signature (should be rejected)"""
    result = TestResult("M1: Invalid Signature Rejection")
    try:
        payload = create_webhook_payload("TSLA", "BUY", 10)
        payload['signature'] = "invalid_signature_12345678"
        
        response = requests.post(f"{BASE_URL}/webhook", json=payload, timeout=5)
        
        if response.status_code == 401:
            result.pass_test()
        else:
            result.fail_test(f"Expected 401, got {response.status_code}")
    except Exception as e:
        result.fail_test(str(e))
    return result

def test_m1_webhook_old_timestamp():
    """Test webhook with old timestamp (should be rejected)"""
    result = TestResult("M1: Old Timestamp Rejection")
    try:
        payload = {
            "ticker": "TSLA",
            "action": "BUY",
            "quantity": 10,
            "order_type": "MARKET",
            "strategy": "test",
            "timestamp": (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        }
        
        payload_json = json.dumps(payload, separators=(',', ':'))
        signature = hmac.new(
            WEBHOOK_SECRET.encode(),
            payload_json.encode(),
            hashlib.sha256
        ).hexdigest()
        payload['signature'] = signature
        
        response = requests.post(f"{BASE_URL}/webhook", json=payload, timeout=5)
        
        if response.status_code == 400:
            result.pass_test()
        else:
            result.fail_test(f"Expected 400, got {response.status_code}")
    except Exception as e:
        result.fail_test(str(e))
    return result

# ============================================================================
# MILESTONE 2 TESTS: Risk Engine & Order Execution
# ============================================================================

def test_m2_system_status():
    """Test admin system status endpoint"""
    result = TestResult("M2: System Status")
    try:
        response = requests.get(
            f"{BASE_URL}/admin/status",
            headers={"X-API-Key": ADMIN_API_KEY},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            required_fields = ['trading_enabled', 'kill_switch_active', 'ibkr_connected', 
                             'daily_trade_count', 'positions']
            if all(field in data for field in required_fields):
                result.pass_test(f"IBKR: {data.get('ibkr_connected')}, Kill Switch: {data.get('kill_switch_active')}")
            else:
                result.fail_test("Missing required fields in response")
        else:
            result.fail_test(f"HTTP {response.status_code}: {response.text}")
    except Exception as e:
        result.fail_test(str(e))
    return result

def test_m2_kill_switch():
    """Test kill switch activation and deactivation"""
    result = TestResult("M2: Kill Switch")
    try:
        # First, resume in case it's already active
        requests.post(
            f"{BASE_URL}/admin/resume",
            headers={"X-API-Key": ADMIN_API_KEY},
            json={"reason": "Test setup", "actor": "TEST"},
            timeout=5
        )
        time.sleep(1)
        
        # Activate kill switch
        response = requests.post(
            f"{BASE_URL}/admin/kill",
            headers={"X-API-Key": ADMIN_API_KEY},
            json={"reason": "Test", "actor": "TEST"},
            timeout=5
        )
        
        if response.status_code != 200:
            result.fail_test(f"Failed to activate: HTTP {response.status_code}")
            return result
        
        # Verify it's active
        status = requests.get(
            f"{BASE_URL}/admin/status",
            headers={"X-API-Key": ADMIN_API_KEY},
            timeout=5
        ).json()
        
        if not status.get('kill_switch_active'):
            result.fail_test("Kill switch not active after activation")
            return result
        
        # Try to send webhook (should be rejected)
        payload = create_webhook_payload("TEST", "BUY", 10)
        webhook_response = requests.post(f"{BASE_URL}/webhook", json=payload, timeout=5)
        
        if webhook_response.status_code == 200:
            response_data = webhook_response.json()
            if response_data.get('status') != 'rejected':
                result.fail_test("Webhook not rejected during kill switch")
                return result
        
        # Deactivate kill switch
        response = requests.post(
            f"{BASE_URL}/admin/resume",
            headers={"X-API-Key": ADMIN_API_KEY},
            json={"reason": "Test complete", "actor": "TEST"},
            timeout=5
        )
        
        if response.status_code != 200:
            result.fail_test(f"Failed to deactivate: HTTP {response.status_code}")
            return result
        
        result.pass_test()
    except Exception as e:
        result.fail_test(str(e))
    return result

def test_m2_position_limit():
    """Test position size limit enforcement"""
    result = TestResult("M2: Position Size Limit")
    try:
        # Resume trading first
        requests.post(
            f"{BASE_URL}/admin/resume",
            headers={"X-API-Key": ADMIN_API_KEY},
            json={"reason": "Test", "actor": "TEST"},
            timeout=5
        )
        
        # Try to exceed max position size
        payload = create_webhook_payload("TESTLIMIT", "BUY", 10000)  # Way over limit
        response = requests.post(f"{BASE_URL}/webhook", json=payload, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'rejected' and 'Position limit' in data.get('message', ''):
                result.pass_test()
            else:
                result.fail_test(f"Large position not properly rejected: {data.get('message')}")
        else:
            result.fail_test(f"HTTP {response.status_code}")
    except Exception as e:
        result.fail_test(str(e))
    return result

# ============================================================================
# MILESTONE 3 TESTS: Persistence & Reconciliation
# ============================================================================

def test_m3_trade_logging():
    """Test that trades are logged to database"""
    result = TestResult("M3: Trade Logging")
    try:
        # Check trades endpoint
        trades_response = requests.get(
            f"{BASE_URL}/admin/trades?limit=10",
            headers={"X-API-Key": ADMIN_API_KEY},
            timeout=5
        )
        
        if trades_response.status_code == 200:
            data = trades_response.json()
            trades = data.get('trades', [])
            result.pass_test(f"Found {len(trades)} trades in database")
        else:
            result.fail_test(f"HTTP {trades_response.status_code}")
    except Exception as e:
        result.fail_test(str(e))
    return result

def test_m3_expected_positions():
    """Test expected positions tracking"""
    result = TestResult("M3: Expected Positions")
    try:
        response = requests.get(
            f"{BASE_URL}/admin/expected-positions",
            headers={"X-API-Key": ADMIN_API_KEY},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'positions' in data:
                positions = data['positions']
                result.pass_test(f"{len(positions)} position(s)")
            else:
                result.fail_test("No positions field in response")
        else:
            result.fail_test(f"HTTP {response.status_code}")
    except Exception as e:
        result.fail_test(str(e))
    return result

def test_m3_reconciliation():
    """Test manual reconciliation trigger"""
    result = TestResult("M3: Manual Reconciliation")
    try:
        response = requests.post(
            f"{BASE_URL}/admin/reconcile",
            headers={"X-API-Key": ADMIN_API_KEY},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            required_fields = ['timestamp', 'status', 'expected_positions', 'actual_positions']
            if all(field in data for field in required_fields):
                result.pass_test(f"Status: {data['status']}, Mismatches: {len(data.get('mismatches', []))}")
            else:
                result.fail_test("Missing required fields in reconciliation report")
        else:
            result.fail_test(f"HTTP {response.status_code}")
    except Exception as e:
        result.fail_test(str(e))
    return result

def test_m3_audit_log():
    """Test audit log retrieval"""
    result = TestResult("M3: Audit Log")
    try:
        response = requests.get(
            f"{BASE_URL}/admin/audit-log?limit=50",
            headers={"X-API-Key": ADMIN_API_KEY},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            logs = data.get('logs', [])
            result.pass_test(f"Found {len(logs)} audit entries")
        else:
            result.fail_test(f"HTTP {response.status_code}")
    except Exception as e:
        result.fail_test(str(e))
    return result

# ============================================================================
# MILESTONE 4 TESTS: Admin & Monitoring
# ============================================================================

def test_m4_admin_auth():
    """Test admin endpoint authentication"""
    result = TestResult("M4: Admin Authentication")
    try:
        # Test without API key (should fail)
        response = requests.get(f"{BASE_URL}/admin/status", timeout=5)
        
        if response.status_code == 422:  # FastAPI validation error for missing header
            # Test with valid key
            response = requests.get(
                f"{BASE_URL}/admin/status",
                headers={"X-API-Key": ADMIN_API_KEY},
                timeout=5
            )
            
            if response.status_code == 200:
                result.pass_test()
            else:
                result.fail_test(f"Valid API key rejected: HTTP {response.status_code}")
        else:
            result.fail_test(f"Expected 422 for missing key, got {response.status_code}")
    except Exception as e:
        result.fail_test(str(e))
    return result

def test_m4_reconciliation_history():
    """Test reconciliation history endpoint"""
    result = TestResult("M4: Reconciliation History")
    try:
        response = requests.get(
            f"{BASE_URL}/admin/reconciliation-history?limit=5",
            headers={"X-API-Key": ADMIN_API_KEY},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'history' in data:
                result.pass_test(f"Found {len(data['history'])} entries")
            else:
                result.fail_test("No history field in response")
        else:
            result.fail_test(f"HTTP {response.status_code}")
    except Exception as e:
        result.fail_test(str(e))
    return result

def test_m4_reset_limits():
    """Test reset daily limits"""
    result = TestResult("M4: Reset Daily Limits")
    try:
        response = requests.post(
            f"{BASE_URL}/admin/reset-limits",
            headers={"X-API-Key": ADMIN_API_KEY},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'reset':
                result.pass_test()
            else:
                result.fail_test("Unexpected response")
        else:
            result.fail_test(f"HTTP {response.status_code}")
    except Exception as e:
        result.fail_test(str(e))
    return result

# ============================================================================
# TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run complete test suite"""
    print("=" * 70)
    print("TV-IBKR COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print(f"Base URL: {BASE_URL}")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)
    
    results = []
    
    # Milestone 1 Tests
    print("\n📋 MILESTONE 1: Core Ingestion & Security")
    print("-" * 70)
    results.append(test_m1_health_endpoint())
    results.append(test_m1_webhook_valid_signature())
    results.append(test_m1_webhook_invalid_signature())
    results.append(test_m1_webhook_old_timestamp())
    
    # Milestone 2 Tests
    print("\n📋 MILESTONE 2: Risk Engine & Order Execution")
    print("-" * 70)
    results.append(test_m2_system_status())
    results.append(test_m2_kill_switch())
    results.append(test_m2_position_limit())
    
    # Milestone 3 Tests
    print("\n📋 MILESTONE 3: Persistence & Reconciliation")
    print("-" * 70)
    results.append(test_m3_trade_logging())
    results.append(test_m3_expected_positions())
    results.append(test_m3_reconciliation())
    results.append(test_m3_audit_log())
    
    # Milestone 4 Tests
    print("\n📋 MILESTONE 4: Admin & Monitoring")
    print("-" * 70)
    results.append(test_m4_admin_auth())
    results.append(test_m4_reconciliation_history())
    results.append(test_m4_reset_limits())
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total = len(results)
    
    print(f"Total Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if failed > 0:
        print("\nFailed Tests:")
        for r in results:
            if not r.passed:
                print(f"  ❌ {r.name}: {r.message}")
    
    print("=" * 70)
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)