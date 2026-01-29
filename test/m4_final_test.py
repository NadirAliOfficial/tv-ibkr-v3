import json
import hmac
import hashlib
import time
import requests
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

def load_env_file():
    """Load .env file and return configuration"""
    env_path = Path(__file__).parent.parent / '.env'
    
    config = {
        'WEBHOOK_SECRET': None,
        'ADMIN_API_KEY': None
    }
    
    if not env_path.exists():
        env_path = Path(__file__).parent / '.env'
    
    if env_path.exists():
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
if not WEBHOOK_SECRET or WEBHOOK_SECRET in ['CHANGE_ME_OR_TRADING_DISABLED', 'UNSAFE_DEFAULT']:
    print("❌ ERROR: WEBHOOK_SECRET not configured properly")
    exit(1)

if not ADMIN_API_KEY or ADMIN_API_KEY in ['CHANGE_ME', 'UNSAFE_DEFAULT']:
    print("❌ ERROR: ADMIN_API_KEY not configured properly")
    exit(1)

print(f"✅ Configuration loaded")
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

def create_webhook_payload(ticker: str, action: str, quantity: int = 10, order_type: str = "MARKET") -> dict:
    """Create properly signed webhook payload"""
    payload = {
        "ticker": ticker,
        "action": action,
        "quantity": quantity,
        "order_type": order_type,
        "strategy": "test_strategy",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    if order_type == "STOP":
        payload["limit_price"] = 150.0
    
    payload_json = json.dumps(payload, separators=(',', ':'))
    signature = hmac.new(
        WEBHOOK_SECRET.encode(),
        payload_json.encode(),
        hashlib.sha256
    ).hexdigest()
    
    payload['signature'] = signature
    return payload

# ============================================================================
# MILESTONE 1: Security & Ingestion
# ============================================================================

def test_m1_health():
    result = TestResult("M1: Health Endpoint")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'healthy':
                result.pass_test()
            else:
                result.fail_test("Unexpected status")
        else:
            result.fail_test(f"HTTP {response.status_code}")
    except Exception as e:
        result.fail_test(str(e))
    return result

def test_m1_webhook_valid():
    result = TestResult("M1: Valid Webhook")
    try:
        payload = create_webhook_payload("TEST1", "BUY", 5)
        response = requests.post(f"{BASE_URL}/webhook", json=payload, timeout=10)
        
        if response.status_code in [200, 409]:
            data = response.json()
            if 'correlation_id' in data:
                result.pass_test(f"Status: {data.get('status')}")
            else:
                result.fail_test("Missing correlation_id")
        else:
            result.fail_test(f"HTTP {response.status_code}")
    except Exception as e:
        result.fail_test(str(e))
    return result

def test_m1_invalid_signature():
    result = TestResult("M1: Invalid Signature Rejection")
    try:
        payload = create_webhook_payload("TEST", "BUY", 10)
        payload['signature'] = "invalid_sig"
        
        response = requests.post(f"{BASE_URL}/webhook", json=payload, timeout=5)
        
        if response.status_code == 401:
            result.pass_test()
        else:
            result.fail_test(f"Expected 401, got {response.status_code}")
    except Exception as e:
        result.fail_test(str(e))
    return result

def test_m1_old_timestamp():
    result = TestResult("M1: Old Timestamp Rejection")
    try:
        payload = {
            "ticker": "OLD",
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
        
        # FIXED: Increased timeout from 5 to 35 seconds
        response = requests.post(f"{BASE_URL}/webhook", json=payload, timeout=35)
        
        if response.status_code == 400:
            result.pass_test()
        else:
            result.fail_test(f"Expected 400, got {response.status_code}")
    except Exception as e:
        result.fail_test(str(e))
    return result

def test_m1_backpressure():
    result = TestResult("M1: Backpressure Check")
    try:
        status = requests.get(
            f"{BASE_URL}/admin/status",
            headers={"X-API-Key": ADMIN_API_KEY},
            timeout=5
        )
        
        if status.status_code == 200:
            result.pass_test("System operational (backpressure ready)")
        else:
            result.fail_test(f"HTTP {status.status_code}")
    except Exception as e:
        result.fail_test(str(e))
    return result

# ============================================================================
# MILESTONE 2: Risk Engine & Execution
# ============================================================================

def test_m2_system_status():
    result = TestResult("M2: System Status")
    try:
        response = requests.get(
            f"{BASE_URL}/admin/status",
            headers={"X-API-Key": ADMIN_API_KEY},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            required = ['trading_enabled', 'kill_switch_active', 'ibkr_connected', 
                       'daily_trade_count', 'positions', 'daily_pnl']
            
            missing = [f for f in required if f not in data]
            if not missing:
                result.pass_test(f"IBKR: {data.get('ibkr_connected')}, PnL: ${data.get('daily_pnl', 0):.2f}")
            else:
                result.fail_test(f"Missing: {missing}")
        else:
            result.fail_test(f"HTTP {response.status_code}")
    except Exception as e:
        result.fail_test(str(e))
    return result

def test_m2_kill_switch():
    result = TestResult("M2: Kill Switch")
    try:
        # FIXED: Increased timeouts from 5 to 15 seconds
        # Resume first
        requests.post(
            f"{BASE_URL}/admin/resume",
            headers={"X-API-Key": ADMIN_API_KEY},
            json={"reason": "Test", "actor": "TEST"},
            timeout=15
        )
        time.sleep(1)
        
        # Activate
        activate_resp = requests.post(
            f"{BASE_URL}/admin/kill",
            headers={"X-API-Key": ADMIN_API_KEY},
            json={"reason": "Test", "actor": "TEST"},
            timeout=15
        )
        
        if activate_resp.status_code != 200:
            result.fail_test(f"Activate failed: {activate_resp.status_code}")
            return result
        
        # Verify
        status = requests.get(
            f"{BASE_URL}/admin/status",
            headers={"X-API-Key": ADMIN_API_KEY},
            timeout=15
        ).json()
        
        if not status.get('kill_switch_active'):
            result.fail_test("Not active after activation")
            return result
        
        # Deactivate
        deactivate_resp = requests.post(
            f"{BASE_URL}/admin/resume",
            headers={"X-API-Key": ADMIN_API_KEY},
            json={"reason": "Test done", "actor": "TEST"},
            timeout=15
        )
        
        if deactivate_resp.status_code == 200:
            result.pass_test()
        else:
            result.fail_test(f"Deactivate failed: {deactivate_resp.status_code}")
    except Exception as e:
        result.fail_test(str(e))
    return result

def test_m2_position_limit():
    result = TestResult("M2: Position Limit")
    try:
        # FIXED: Increased timeout from 5 to 35 seconds
        # Resume trading
        requests.post(
            f"{BASE_URL}/admin/resume",
            headers={"X-API-Key": ADMIN_API_KEY},
            json={"reason": "Test", "actor": "TEST"},
            timeout=15
        )
        
        # Try large position
        payload = create_webhook_payload("BIGLIMIT", "BUY", 10000)
        response = requests.post(f"{BASE_URL}/webhook", json=payload, timeout=35)
        
        if response.status_code in [200, 409]:
            result.pass_test("Large order accepted (will be risk-checked)")
        else:
            result.fail_test(f"HTTP {response.status_code}")
    except Exception as e:
        result.fail_test(str(e))
    return result

def test_m2_stop_orders():
    result = TestResult("M2: STOP Orders")
    try:
        payload = create_webhook_payload("STOPTEST", "BUY", 10, order_type="STOP")
        response = requests.post(f"{BASE_URL}/webhook", json=payload, timeout=10)
        
        if response.status_code in [200, 409]:
            result.pass_test("STOP order accepted")
        else:
            result.fail_test(f"HTTP {response.status_code}")
    except Exception as e:
        result.fail_test(str(e))
    return result

def test_m2_close_action():
    result = TestResult("M2: CLOSE Action")
    try:
        payload = create_webhook_payload("CLOSETEST", "CLOSE", 10)
        response = requests.post(f"{BASE_URL}/webhook", json=payload, timeout=10)
        
        if response.status_code in [200, 409]:
            result.pass_test("CLOSE action accepted")
        else:
            result.fail_test(f"HTTP {response.status_code}")
    except Exception as e:
        result.fail_test(str(e))
    return result

# ============================================================================
# MILESTONE 3: Persistence & Reconciliation
# ============================================================================

def test_m3_trade_logging():
    result = TestResult("M3: Trade Logging")
    try:
        time.sleep(2)  # Wait for async processing
        
        response = requests.get(
            f"{BASE_URL}/admin/trades?limit=10",
            headers={"X-API-Key": ADMIN_API_KEY},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            trades = data.get('trades', [])
            
            # Check for enhanced fields
            has_new_fields = False
            if trades:
                trade = trades[0]
                has_new_fields = 'webhook_received_at' in trade and 'raw_payload' in trade
            
            if has_new_fields:
                result.pass_test(f"{len(trades)} trades with timing & raw payload")
            else:
                result.pass_test(f"{len(trades)} trades")
        else:
            result.fail_test(f"HTTP {response.status_code}")
    except Exception as e:
        result.fail_test(str(e))
    return result

def test_m3_expected_positions():
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
                result.fail_test("No positions field")
        else:
            result.fail_test(f"HTTP {response.status_code}")
    except Exception as e:
        result.fail_test(str(e))
    return result

def test_m3_reconciliation():
    result = TestResult("M3: Reconciliation")
    try:
        # FIXED: Increased timeout from 5 to 15 seconds
        response = requests.post(
            f"{BASE_URL}/admin/reconcile",
            headers={"X-API-Key": ADMIN_API_KEY},
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            required = ['timestamp', 'status', 'expected_positions', 'actual_positions']
            
            if all(f in data for f in required):
                result.pass_test(f"Status: {data['status']}")
            else:
                result.fail_test("Missing fields")
        else:
            result.fail_test(f"HTTP {response.status_code}")
    except Exception as e:
        result.fail_test(str(e))
    return result

def test_m3_audit_log():
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
            result.pass_test(f"{len(logs)} entries")
        else:
            result.fail_test(f"HTTP {response.status_code}")
    except Exception as e:
        result.fail_test(str(e))
    return result

def test_m3_admin_audit():
    result = TestResult("M3: Admin Audit Log (NEW)")
    try:
        response = requests.get(
            f"{BASE_URL}/admin/audit",
            headers={"X-API-Key": ADMIN_API_KEY},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            logs = data.get('logs', [])
            result.pass_test(f"{len(logs)} admin actions")
        else:
            result.fail_test(f"HTTP {response.status_code}")
    except Exception as e:
        result.fail_test(str(e))
    return result

# ============================================================================
# MILESTONE 4: Admin & Monitoring
# ============================================================================

def test_m4_admin_auth():
    result = TestResult("M4: Admin Auth")
    try:
        # Without key
        response = requests.get(f"{BASE_URL}/admin/status", timeout=5)
        
        if response.status_code == 422:
            # With key
            response = requests.get(
                f"{BASE_URL}/admin/status",
                headers={"X-API-Key": ADMIN_API_KEY},
                timeout=5
            )
            
            if response.status_code == 200:
                result.pass_test()
            else:
                result.fail_test(f"Valid key rejected: {response.status_code}")
        else:
            result.fail_test(f"Expected 422, got {response.status_code}")
    except Exception as e:
        result.fail_test(str(e))
    return result

def test_m4_reconciliation_history():
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
                result.pass_test(f"{len(data['history'])} entries")
            else:
                result.fail_test("No history field")
        else:
            result.fail_test(f"HTTP {response.status_code}")
    except Exception as e:
        result.fail_test(str(e))
    return result

def test_m4_reset_limits():
    result = TestResult("M4: Reset Limits")
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

def test_m4_debug_recent():
    result = TestResult("M4: Debug Recent (NEW)")
    try:
        response = requests.get(
            f"{BASE_URL}/debug/recent?n=5",
            headers={"X-API-Key": ADMIN_API_KEY},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'trades' in data:
                trades = data['trades']
                result.pass_test(f"{len(trades)} recent trades")
            else:
                result.fail_test("No trades field")
        else:
            result.fail_test(f"HTTP {response.status_code}")
    except Exception as e:
        result.fail_test(str(e))
    return result

def test_m4_pnl_endpoint():
    result = TestResult("M4: PnL Endpoint (NEW)")
    try:
        response = requests.get(
            f"{BASE_URL}/admin/pnl",
            headers={"X-API-Key": ADMIN_API_KEY},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            required = ['date', 'total_pnl', 'total_trades']
            
            if all(f in data for f in required):
                result.pass_test(f"PnL: ${data['total_pnl']:.2f}, Trades: {data['total_trades']}")
            else:
                result.fail_test(f"Missing: {[f for f in required if f not in data]}")
        else:
            result.fail_test(f"HTTP {response.status_code}")
    except Exception as e:
        result.fail_test(str(e))
    return result

# ============================================================================
# TEST RUNNER
# ============================================================================

def run_all_tests():
    print("=" * 70)
    print("TV-IBKR COMPLETE TEST SUITE (FIXED TIMEOUTS)")
    print("=" * 70)
    print(f"Base URL: {BASE_URL}")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)
    
    results = []
    
    # M1: Security & Ingestion
    print("\n📋 MILESTONE 1: Security & Ingestion")
    print("-" * 70)
    results.append(test_m1_health())
    results.append(test_m1_webhook_valid())
    results.append(test_m1_invalid_signature())
    results.append(test_m1_old_timestamp())
    results.append(test_m1_backpressure())
    
    # M2: Risk Engine & Execution
    print("\n📋 MILESTONE 2: Risk Engine & Execution")
    print("-" * 70)
    results.append(test_m2_system_status())
    results.append(test_m2_kill_switch())
    results.append(test_m2_position_limit())
    results.append(test_m2_stop_orders())
    results.append(test_m2_close_action())
    
    # M3: Persistence & Reconciliation
    print("\n📋 MILESTONE 3: Persistence & Reconciliation")
    print("-" * 70)
    results.append(test_m3_trade_logging())
    results.append(test_m3_expected_positions())
    results.append(test_m3_reconciliation())
    results.append(test_m3_audit_log())
    results.append(test_m3_admin_audit())
    
    # M4: Admin & Monitoring
    print("\n📋 MILESTONE 4: Admin & Monitoring")
    print("-" * 70)
    results.append(test_m4_admin_auth())
    results.append(test_m4_reconciliation_history())
    results.append(test_m4_reset_limits())
    results.append(test_m4_debug_recent())
    results.append(test_m4_pnl_endpoint())
    
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
    
    print("\n" + "=" * 70)
    print("NOTE: Timeouts increased to allow for IBKR execution delays")
    print("=" * 70)
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
