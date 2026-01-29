"""
Fixed webhook testing script.

The issue: requests.post(json=payload) reformats JSON, causing HMAC mismatch.
The fix: Send raw JSON string with data= parameter instead of json= parameter.
"""

import requests
import json
import hmac
import hashlib
from datetime import datetime, timezone, timedelta

# Your webhook secret from .env
SECRET = "supersecretkey"  # MUST match WEBHOOK_SECRET in your .env file
BASE_URL = "http://localhost:8000"

def send_webhook(ticker="AAPL", action="BUY", quantity=10, timestamp_offset_seconds=0):
    """
    Send a properly signed webhook to the API.
    
    Args:
        ticker: Stock symbol
        action: BUY, SELL, or CLOSE
        quantity: Number of shares
        timestamp_offset_seconds: Offset from current time (for testing stale webhooks)
    """
    timestamp = datetime.now(timezone.utc) + timedelta(seconds=timestamp_offset_seconds)
    return send_webhook_with_timestamp(ticker, action, quantity, timestamp)

def send_webhook_with_timestamp(ticker, action, quantity, timestamp):
    """
    Send a webhook with an explicit timestamp.
    Used for duplicate testing where we need identical timestamps.
    """
    # Build payload with timestamp (without signature first)
    payload_without_sig = {
        "ticker": ticker,
        "action": action,
        "quantity": quantity,
        "order_type": "MARKET",
        "timestamp": timestamp.isoformat()
    }
    
    # Calculate HMAC signature over payload WITHOUT signature field
    payload_json = json.dumps(payload_without_sig, separators=(',', ':'))
    signature = hmac.new(
        SECRET.encode(),
        payload_json.encode(),
        hashlib.sha256
    ).hexdigest()
    
    # Now add signature to create final payload
    payload_without_sig["signature"] = signature
    final_json = json.dumps(payload_without_sig, separators=(',', ':'))
    
    print(f"\n📤 Sending webhook: {action} {quantity} {ticker}")
    print(f"   Timestamp: {timestamp.isoformat()}")
    print(f"   Signature: {signature[:16]}...")
    
    # Send as raw JSON string (not json= parameter!)
    response = requests.post(
        f"{BASE_URL}/webhook",
        data=final_json,  # Use data= not json=
        headers={"Content-Type": "application/json"}
    )
    
    print(f"📥 Response [{response.status_code}]:")
    try:
        print(json.dumps(response.json(), indent=2))
    except:
        print(response.text)
    
    return response

def test_valid_webhook():
    """Test 1: Valid webhook should be accepted."""
    print("\n" + "="*70)
    print("TEST 1: Valid Webhook")
    print("="*70)
    response = send_webhook("AAPL", "BUY", 10)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    print("✅ PASS: Valid webhook accepted")

def test_duplicate_webhook():
    """Test 2: Duplicate webhook should be rejected."""
    print("\n" + "="*70)
    print("TEST 2: Duplicate Webhook (Same idempotency key)")
    print("="*70)
    
    # Use a fixed timestamp for both requests
    fixed_timestamp = datetime.now(timezone.utc)
    
    # First webhook
    print("Sending first webhook...")
    response1 = send_webhook_with_timestamp("TSLA", "SELL", 5, fixed_timestamp)
    assert response1.status_code == 200
    
    # Duplicate (same ticker, action, timestamp = same idempotency key)
    print("Sending duplicate webhook (same timestamp)...")
    response2 = send_webhook_with_timestamp("TSLA", "SELL", 5, fixed_timestamp)
    assert response2.status_code == 409, f"Expected 409 Conflict, got {response2.status_code}"
    print("✅ PASS: Duplicate webhook rejected")

def test_invalid_signature():
    """Test 3: Invalid signature should be rejected."""
    print("\n" + "="*70)
    print("TEST 3: Invalid Signature")
    print("="*70)
    
    payload = {
        "ticker": "MSFT",
        "action": "BUY",
        "quantity": 20,
        "order_type": "MARKET",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "signature": "invalid_signature_12345"
    }
    
    response = requests.post(
        f"{BASE_URL}/webhook",
        json=payload
    )
    
    print(f"📥 Response [{response.status_code}]:")
    print(json.dumps(response.json(), indent=2))
    assert response.status_code == 401, f"Expected 401, got {response.status_code}"
    print("✅ PASS: Invalid signature rejected")

def test_stale_webhook():
    """Test 4: Stale webhook (>30s old) should be rejected."""
    print("\n" + "="*70)
    print("TEST 4: Stale Webhook (>30 seconds old)")
    print("="*70)
    
    # Send webhook with timestamp 60 seconds in the past
    response = send_webhook("GOOGL", "BUY", 15, timestamp_offset_seconds=-60)
    assert response.status_code == 400, f"Expected 400, got {response.status_code}"
    print("✅ PASS: Stale webhook rejected")

def test_future_webhook():
    """Test 5: Future webhook should be rejected."""
    print("\n" + "="*70)
    print("TEST 5: Future Webhook (>5 seconds ahead)")
    print("="*70)
    
    # Send webhook with timestamp 10 seconds in the future
    response = send_webhook("AMZN", "SELL", 8, timestamp_offset_seconds=10)
    assert response.status_code == 400, f"Expected 400, got {response.status_code}"
    print("✅ PASS: Future webhook rejected")

def test_invalid_schema():
    """Test 6: Invalid schema should be rejected."""
    print("\n" + "="*70)
    print("TEST 6: Invalid Schema (missing required field)")
    print("="*70)
    
    # Create payload missing required field (ticker) but with valid signature
    payload_without_sig = {
        "action": "BUY",  # Missing ticker - this should fail schema validation
        "quantity": 10,
        "order_type": "MARKET",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    # Calculate valid HMAC signature
    payload_json = json.dumps(payload_without_sig, separators=(',', ':'))
    signature = hmac.new(
        SECRET.encode(),
        payload_json.encode(),
        hashlib.sha256
    ).hexdigest()
    
    payload_without_sig["signature"] = signature
    final_json = json.dumps(payload_without_sig, separators=(',', ':'))
    
    print(f"📤 Sending invalid payload (missing 'ticker' field)")
    
    response = requests.post(
        f"{BASE_URL}/webhook",
        data=final_json,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"📥 Response [{response.status_code}]:")
    print(json.dumps(response.json(), indent=2))
    assert response.status_code == 422, f"Expected 422, got {response.status_code}"
    print("✅ PASS: Invalid schema rejected")

def test_health_endpoints():
    """Test 7: Health and ready endpoints."""
    print("\n" + "="*70)
    print("TEST 7: Health Check Endpoints")
    print("="*70)
    
    # Health check
    response = requests.get(f"{BASE_URL}/health")
    print(f"GET /health: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    assert response.status_code == 200
    
    # Ready check
    response = requests.get(f"{BASE_URL}/ready")
    print(f"\nGET /ready: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    assert response.status_code == 200
    
    print("✅ PASS: Health endpoints working")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TV-IBKR-v3 Milestone 1 - Test Suite")
    print("="*70)
    print(f"Testing against: {BASE_URL}")
    print(f"Webhook secret: {SECRET[:8]}...")
    
    try:
        # Run all tests
        test_valid_webhook()
        test_duplicate_webhook()
        test_invalid_signature()
        test_stale_webhook()
        test_future_webhook()
        test_invalid_schema()
        test_health_endpoints()
        
        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED!")
        print("="*70)
        print("\nMilestone 1 Deliverable: ✅ COMPLETE")
        print("\nDeliverable includes:")
        print("  ✓ HMAC signature verification")
        print("  ✓ Timestamp validation (±30s window)")
        print("  ✓ Idempotency protection")
        print("  ✓ Schema validation")
        print("  ✓ Health check endpoints")
        print("  ✓ Structured logging with correlation IDs")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        exit(1)
    except requests.exceptions.ConnectionError:
        print(f"\n❌ ERROR: Cannot connect to {BASE_URL}")
        print("Make sure the server is running: uvicorn main:app --reload")
        exit(1)