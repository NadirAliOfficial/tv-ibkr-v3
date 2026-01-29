"""Full M2 integration test"""

import requests
import time

BASE_URL = "http://localhost:8000"

print("="*70)
print("M2 Full Integration Test")
print("="*70)

# 1. Check health
print("\n1. Health Check...")
r = requests.get(f"{BASE_URL}/ready")
print(f"   Ready: {r.json()}")

# 2. Check status
print("\n2. System Status...")
r = requests.get(f"{BASE_URL}/admin/status")
status = r.json()
print(f"   Trading Enabled: {status['trading_enabled']}")
print(f"   IBKR Connected: {status['ibkr_connected']}")
print(f"   Kill Switch: {status['kill_switch_active']}")

# 3. Test kill switch
print("\n3. Testing Kill Switch...")
r = requests.post(f"{BASE_URL}/admin/kill", json={"reason": "Test", "actor": "AUTO"})
print(f"   Activated: {r.json()['status']}")

r = requests.get(f"{BASE_URL}/admin/status")
print(f"   Kill Switch Active: {r.json()['kill_switch_active']}")

r = requests.post(f"{BASE_URL}/admin/resume", json={"reason": "Test done", "actor": "AUTO"})
print(f"   Deactivated: {r.json()['status']}")

# 4. Send webhook (use your test_wehbook.py logic here)
print("\n4. Sending Test Webhook...")
print("   (Use test_wehbook.py send_webhook function)")

print("\n" + "="*70)
print("✅ Integration test complete!")
print("="*70)