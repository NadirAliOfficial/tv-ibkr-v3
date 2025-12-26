# TV-IBKR-v3: TradingView to Interactive Brokers Execution System

**Milestone 1: Core Ingestion & Security** ✅

A production-hardened webhook receiver that validates TradingView algorithmic trading signals with institutional-grade security controls before execution on Interactive Brokers.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 What This Does

This is **Milestone 1** of a 4-milestone project that connects TradingView alerts to Interactive Brokers for automated trading. This milestone implements the critical security layer:

- ✅ **HMAC-SHA256 webhook authentication** - Prevents unauthorized order injection
- ✅ **Timestamp validation** - Rejects stale webhooks (>30s old) to prevent replay attacks
- ✅ **Schema validation** - Type-safe payload parsing with Pydantic
- ✅ **Idempotency protection** - Detects and rejects duplicate webhooks
- ✅ **Structured logging** - Every request traced with correlation IDs
- ✅ **Health endpoints** - Production-ready liveness and readiness probes
- ✅ **Cloudflare-ready** - Designed to sit behind Cloudflare WAF

**What's NOT in M1:** Risk engine, order execution, database persistence, position reconciliation (coming in M2-M4)

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- pip or uv package manager
- TradingView account (for webhook testing)


### Configuration

Create a `.env` file in the project root:

```env
# REQUIRED: Webhook Security
WEBHOOK_SECRET=your-super-secret-key-min-32-chars

# OPTIONAL: Replay Protection
WEBHOOK_TIMESTAMP_TOLERANCE_SECONDS=30

# OPTIONAL: Logging
LOG_LEVEL=INFO
```

**Security Note:** Use a strong random secret for `WEBHOOK_SECRET`. Generate one with:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### Running the Server

```bash
# Development mode (auto-reload on code changes)
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

The API will be available at:
- **API Base:** http://localhost:8000
- **Interactive Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

## 📡 API Endpoints

### `POST /webhook`
Main webhook ingestion endpoint. Accepts TradingView alerts.

**Request Body:**
```json
{
  "ticker": "AAPL",
  "action": "BUY",
  "quantity": 10,
  "order_type": "MARKET",
  "timestamp": "2025-12-26T12:30:00Z",
  "signature": "hmac-sha256-hex-signature"
}
```

**Response (200 OK):**
```json
{
  "status": "accepted",
  "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Webhook validated: BUY 10 AAPL",
  "idempotency_key": "a1b2c3d4e5f6g7h8"
}
```

**Error Responses:**
- `401` - Invalid HMAC signature
- `400` - Stale timestamp (>30s old) or future timestamp
- `409` - Duplicate webhook (already processed)
- `422` - Invalid payload schema

### `GET /health`
Liveness probe - returns 200 if service is running.

### `GET /ready`
Readiness probe - returns 200 if service can accept traffic.

### `GET /`
API information and available endpoints.

## 🧪 Testing

### Manual Testing with curl

```bash
# 1. Generate a valid test webhook
python -c "
import json
import hmac
import hashlib
from datetime import datetime, timezone

payload = {
    'ticker': 'AAPL',
    'action': 'BUY',
    'quantity': 10,
    'order_type': 'MARKET',
    'timestamp': datetime.now(timezone.utc).isoformat(),
    'signature': ''
}

secret = 'your-webhook-secret-here'
payload_bytes = json.dumps(payload).encode()
signature = hmac.new(secret.encode(), payload_bytes, hashlib.sha256).hexdigest()
payload['signature'] = signature

print(json.dumps(payload, indent=2))
"

# 2. Send the webhook
curl -X POST http://localhost:8000/webhook \
  -H "Content-Type: application/json" \
  -d @webhook-payload.json
```

### Testing via Interactive Docs

1. Navigate to http://localhost:8000/docs
2. Click on `POST /webhook` endpoint
3. Click "Try it out"
4. Use the built-in test payload generator
5. Execute and view response

### Test Scenarios Checklist

- [ ] ✅ Valid webhook with correct signature → 200 OK
- [ ] ❌ Webhook with invalid signature → 401 Unauthorized
- [ ] ❌ Webhook with old timestamp (>30s) → 400 Bad Request
- [ ] ❌ Webhook with future timestamp → 400 Bad Request
- [ ] ❌ Duplicate webhook (same idempotency key) → 409 Conflict
- [ ] ❌ Malformed JSON → 400 Bad Request
- [ ] ❌ Missing required fields → 422 Unprocessable Entity
- [ ] ✅ Health check → 200 OK
- [ ] ✅ Ready check → 200 OK

## 📊 Structured Logging

All logs are emitted as JSON for easy parsing by log aggregation tools:

```json
{
  "timestamp": "2025-12-26T12:30:45",
  "level": "INFO",
  "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Webhook validated successfully",
  "extra": {
    "idempotency_key": "a1b2c3d4e5f6g7h8",
    "ticker": "AAPL",
    "action": "BUY",
    "quantity": 10
  }
}
```

**Correlation IDs** allow you to trace a webhook from ingestion through validation in your logs.

## 🔒 Security Features

### HMAC Signature Verification
Every webhook must include a valid HMAC-SHA256 signature computed over the entire request body using the shared secret. This prevents:
- Unauthorized webhook injection
- Man-in-the-middle tampering
- Replay attacks from captured traffic

### Timestamp Validation
Webhooks older than 30 seconds (configurable) are rejected. This prevents:
- Replay attacks using old captured webhooks
- Processing of stale market signals
- Out-of-order execution

### Idempotency Protection
Each webhook generates a unique key from `ticker + action + timestamp`. Duplicate submissions are detected and rejected, preventing:
- Accidental duplicate orders from TradingView
- Intentional replay after timestamp window expires
- Network retry issues causing double execution

### Cloudflare Integration (Recommended)

This service is designed to run behind Cloudflare for additional protection:

**DNS Setup:**
- Create A record pointing to your VPS IP
- Enable Cloudflare proxy (orange cloud)
- Never expose origin IP directly

**WAF Rules:**
```
Rule 1: Block if (Request Method != POST AND URI Path == /webhook)
Rule 2: Rate limit /webhook to 100 req/min per IP
Rule 3: Challenge requests without User-Agent header
```

**Firewall Rules:**
- Allowlist TradingView webhook IPs (recommended)
- Block non-US geo locations if trading US stocks only

## 📁 Project Structure

```
tv-ibkr-v3/
├── main.py              # Complete M1 implementation (single file)
├── requirements.txt     # Python dependencies
├── .env.example         # Environment template
├── .env                 # Your secrets (git-ignored)
├── README.md            # This file
├── Dockerfile           # Container image (optional)
└── tests/
    └── test_webhook.py  # Automated tests (optional)
```

## 🐳 Docker Deployment (Optional)

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY .env .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t tv-ibkr-m1 .
docker run -p 8000:8000 --env-file .env tv-ibkr-m1
```

## 🗺️ Roadmap

### ✅ Milestone 1: Core Ingestion & Security (COMPLETE)
- Webhook endpoint with HMAC verification
- Timestamp and schema validation
- Replay protection
- Health checks and logging

### 🚧 Milestone 2: Risk Engine & Order Execution
- Kill switch and daily loss limits
- Position sizing and exposure controls
- Interactive Brokers integration via ib_insync
- Circuit breaker and retry logic
- Telegram alerting

### 📋 Milestone 3: Persistence & Reconciliation
- SQLite with WAL mode
- Trade logging and audit trails
- Position reconciliation (60s loop)
- Auto-halt on broker mismatch

### 🎓 Milestone 4: Admin, Testing & Handoff
- Admin endpoints (status, kill/resume)
- Audit trail queries
- Paper trading validation
- Complete documentation

## 🤝 Contributing

This is a milestone-based development project. Current milestone (M1) is feature-complete and locked for testing/delivery.

For issues or questions about M1:
1. Check logs for correlation_id of failed request
2. Verify your .env configuration
3. Test with the provided test scenarios
4. Open an issue with full context

## 📄 License

MIT License - see LICENSE file for details

## ⚠️ Disclaimer

This software is for educational and development purposes. Automated trading involves substantial risk. The authors are not responsible for trading losses. Always test thoroughly with paper trading before using real capital.

## 📞 Support

- **Documentation:** Full PRD available in `/docs/PRD-v3.2.pdf`
- **API Docs:** http://localhost:8000/docs (when running)
- **Issues:** GitHub Issues for bug reports
- **Milestone Delivery:** This completes M1 requirements

---

**Built with:** FastAPI, Pydantic, Python 3.11+ | **Next Milestone:** Risk Engine & IBKR Integration
