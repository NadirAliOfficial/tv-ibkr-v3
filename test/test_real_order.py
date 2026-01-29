import requests
import json
import hmac
import hashlib
from datetime import datetime, timezone

SECRET = "supersecretkey"  # From your .env
BASE_URL = "http://localhost:8000"

# Build payload WITHOUT signature
payload = {
    "ticker": "AAPL",
    "action": "BUY",
    "quantity": 1,
    "order_type": "MARKET",
    "timestamp": datetime.now(timezone.utc).isoformat()
}

# Calculate HMAC
payload_json = json.dumps(payload, separators=(',', ':'))
signature = hmac.new(SECRET.encode(), payload_json.encode(), hashlib.sha256).hexdigest()

# Add signature
payload["signature"] = signature
final_json = json.dumps(payload, separators=(',', ':'))

print(f"📤 Sending REAL order: BUY 1 AAPL")
print(f"Signature: {signature[:16]}...")

response = requests.post(
    f"{BASE_URL}/webhook",
    data=final_json,
    headers={"Content-Type": "application/json"}
)

print(f"\n📥 Response [{response.status_code}]:")
print(json.dumps(response.json(), indent=2))

if response.status_code == 200:
    print("\n✅ ORDER SENT TO TWS!")
    print("Check your TWS paper trading account for the order!")
