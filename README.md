# TV-IBKR: Complete TradingView to Interactive Brokers Integration

Production-ready automated trading system connecting TradingView alerts to Interactive Brokers with enterprise-grade security, risk management, and monitoring.

## 🎯 Overview

This system provides a secure, reliable bridge between TradingView webhooks and Interactive Brokers for automated strategy execution. It includes comprehensive risk controls, position tracking, reconciliation, and audit logging.



## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Interactive Brokers TWS or IB Gateway (paper or live account)
- TradingView account with webhook support

### Installation

```bash
# 1. Clone or download the project
cd tv-ibkr-system

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# 1. Copy environment template
cp .env.template .env

# 2. Generate secrets
python3 -c "import secrets; print(secrets.token_hex(32))"
# Copy output to WEBHOOK_SECRET in .env

python3 -c "import secrets; print(secrets.token_hex(32))"
# Copy output to ADMIN_API_KEY in .env

# 3. Edit .env file with your settings
nano .env  # or use your favorite editor
```

**Minimum Required Configuration:**
```bash
# Security (REQUIRED - use generated secrets above)
WEBHOOK_SECRET=your-generated-secret-here
ADMIN_API_KEY=your-generated-admin-key-here

# IBKR Connection
IBKR_HOST=127.0.0.1
IBKR_PORT=7497  # 7497 for TWS paper, 7496 for TWS live

# Start with dry run for testing
DRY_RUN=true
```

### Start IBKR TWS

1. Open Interactive Brokers TWS or IB Gateway
2. Go to **Configure → Settings → API → Settings**
3. Enable "**Enable ActiveX and Socket Clients**"
4. Verify "**Socket port**" is 7497 (paper) or 7496 (live)
5. Uncheck "**Read-Only API**"
6. Click **OK**

### Run the Application

```bash
# Activate virtual environment
source venv/bin/activate

# Run the application
uvicorn m4_final:app --host 0.0.0.0 --port 8000

# You should see:
# INFO: 🎉 IBKR CONNECTED!
# INFO: Application ready
```

### Verify Setup

```bash
# Check health
curl http://localhost:8000/health

# Check system status (replace YOUR_ADMIN_KEY)
curl -H "X-API-Key: YOUR_ADMIN_KEY" http://localhost:8000/admin/status

# Run test suite
python test/m4_final_test.py
```

**Expected:** All 14 tests should pass ✅

## 📋 Features

### Security (Milestone 1)
- ✅ HMAC-SHA256 webhook authentication
- ✅ Timestamp validation with replay protection
- ✅ Idempotency key generation
- ✅ Pydantic schema validation
- ✅ Cloudflare-ready architecture

### Risk Management (Milestone 2)
- ✅ Emergency kill switch
- ✅ Daily loss limits
- ✅ Position size limits
- ✅ Maximum trade count limits
- ✅ Circuit breaker for failures
- ✅ Telegram alerts (optional)
- ✅ Dry-run mode for testing

### Persistence (Milestone 3)
- ✅ SQLite database with WAL mode
- ✅ Single-writer queue pattern
- ✅ Complete trade logging
- ✅ Expected position tracking
- ✅ Automated reconciliation (60s)
- ✅ Auto-halt on position mismatch
- ✅ Comprehensive audit trail

### Administration (Milestone 4)
- ✅ Full REST API for management
- ✅ System status monitoring
- ✅ Trade history queries
- ✅ Reconciliation reports
- ✅ Kill switch controls
- ✅ Health/readiness endpoints
- ✅ Complete test suite

## 🏗️ Architecture

```
┌──────────────┐
│ TradingView  │  Alerts with HMAC signatures
│   Webhooks   │
└──────┬───────┘
       │ HTTPS
       ▼
┌──────────────────────────────────────────┐
│         FastAPI Application              │
├──────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────────┐   │
│  │   Security  │  │  Risk Engine    │   │
│  │   • HMAC    │  │  • Kill Switch  │   │
│  │   • Replay  │  │  • Limits       │   │
│  └─────────────┘  └─────────────────┘   │
│                                          │
│  ┌─────────────┐  ┌─────────────────┐   │
│  │ Persistence │  │ Reconciliation  │   │
│  │   • SQLite  │  │  • Auto-check   │   │
│  │   • Audit   │  │  • Auto-halt    │   │
│  └─────────────┘  └─────────────────┘   │
└──────┬──────────────────────┬────────────┘
       │                      │
       ▼                      ▼
┌──────────────┐      ┌──────────────┐
│    SQLite    │      │     IBKR     │
│   Database   │      │  TWS/Gateway │
│   (WAL mode) │      │              │
└──────────────┘      └──────────────┘
```

## 📡 API Endpoints

### Public Endpoints
```
GET  /                  # Service info
GET  /health           # Health check
GET  /ready            # Readiness check
POST /webhook          # TradingView webhook (HMAC auth)
```

### Admin Endpoints (requires X-API-Key header)
```
GET  /admin/status                   # System status
POST /admin/kill                     # Activate kill switch
POST /admin/resume                   # Deactivate kill switch
POST /admin/reconcile                # Force reconciliation
GET  /admin/trades?limit=50          # Trade history
GET  /admin/audit-log?limit=100      # Audit log
GET  /admin/reconciliation-history   # Reconciliation history
GET  /admin/expected-positions       # Expected positions
GET  /admin/actual-positions         # Actual IBKR positions
POST /admin/reset-limits             # Reset daily counters
```

## 🧪 Testing

### Run Complete Test Suite

```bash
# All tests (recommended)
python test/m4_final_test.py

# Expected output:
# ✅ M1: Health Endpoint: PASSED 
# ✅ M1: Valid Signature: PASSED
# ✅ M1: Invalid Signature Rejection: PASSED 
# ✅ M1: Old Timestamp Rejection: PASSED
# ✅ M2: System Status: PASSED
# ✅ M2: Kill Switch: PASSED 
# ✅ M2: Position Size Limit: PASSED
# ✅ M3: Trade Logging: PASSED
# ✅ M3: Expected Positions: PASSED
# ✅ M3: Manual Reconciliation: PASSED
# ✅ M3: Audit Log: PASSED
# ✅ M4: Admin Authentication: PASSED 
# ✅ M4: Reconciliation History: PASSED
# ✅ M4: Reset Daily Limits: PASSED
#
# Success Rate: 100.0%
```

### Manual Testing

**1. Check Health:**
```bash
curl http://localhost:8000/health
```

**2. Check System Status:**
```bash
curl -H "X-API-Key: YOUR_ADMIN_KEY" http://localhost:8000/admin/status
```

**3. Send Test Webhook:**
```python
import hmac, hashlib, json, requests
from datetime import datetime, timezone

payload = {
    "ticker": "AAPL",
    "action": "BUY",
    "quantity": 10,
    "order_type": "MARKET",
    "strategy": "test",
    "timestamp": datetime.now(timezone.utc).isoformat()
}

secret = "YOUR_WEBHOOK_SECRET"
payload_json = json.dumps(payload, separators=(',', ':'))
signature = hmac.new(secret.encode(), payload_json.encode(), hashlib.sha256).hexdigest()
payload['signature'] = signature

response = requests.post('http://localhost:8000/webhook', json=payload)
print(response.json())
```

## 🔧 Configuration

### Essential Environment Variables

```bash
# Security (REQUIRED)
WEBHOOK_SECRET=<strong-random-key>      # Generate with secrets.token_hex(32)
ADMIN_API_KEY=<strong-random-key>       # Generate with secrets.token_hex(32)
WEBHOOK_TIMESTAMP_TOLERANCE_SECONDS=30

# IBKR Connection (REQUIRED)
IBKR_HOST=127.0.0.1
IBKR_PORT=7497                          # 7497=paper, 7496=live
IBKR_CLIENT_ID=1

# Risk Limits (Adjust based on account size)
MAX_POSITION_SIZE=100
MAX_DAILY_LOSS=500.0
MAX_PORTFOLIO_EXPOSURE=0.25
MAX_DAILY_TRADES=50

# Circuit Breaker
CIRCUIT_BREAKER_THRESHOLD=3
CIRCUIT_BREAKER_TIMEOUT=60

# Persistence
DATABASE_PATH=trading.db
RECONCILIATION_INTERVAL=60
RECONCILIATION_TOLERANCE=0
AUTO_HALT_ON_MISMATCH=true

# Features
TRADING_ENABLED=true
DRY_RUN=false                           # Set true for testing
ENABLE_RISK_ENGINE=true
ENABLE_ORDER_EXECUTION=true
ENABLE_TELEGRAM=false                   # Optional
ENABLE_RECONCILIATION=true

# Logging
LOG_LEVEL=INFO
```

See `.env.template` for all options with detailed comments.

## 🚨 Emergency Procedures

### Activate Kill Switch

```bash
curl -X POST http://localhost:8000/admin/kill \
  -H "X-API-Key: YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "reason": "Emergency halt - market volatility",
    "actor": "TRADER"
  }'
```

### Resume Trading

```bash
curl -X POST http://localhost:8000/admin/resume \
  -H "X-API-Key: YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "reason": "All clear",
    "actor": "TRADER"
  }'
```

### Check System Status

```bash
curl -H "X-API-Key: YOUR_ADMIN_KEY" http://localhost:8000/admin/status
```

### Force Reconciliation

```bash
curl -X POST -H "X-API-Key: YOUR_ADMIN_KEY" http://localhost:8000/admin/reconcile
```

## 📊 Database Schema

### trades
Complete record of all trades with status tracking (PENDING → FILLED/REJECTED/FAILED)

### audit_log
All system events with timestamps, searchable by event type

### expected_positions
Position tracking from trade log, updated on each trade, used for reconciliation

### reconciliation_log
Automated reconciliation results with mismatch detection and action taken

## 🔒 Security Best Practices

1. **Change Default Secrets**
   - Generate strong random values for WEBHOOK_SECRET and ADMIN_API_KEY
   - Never commit secrets to version control

2. **Use HTTPS in Production**
   - Always use SSL/TLS for webhook endpoint
   - Consider Cloudflare for additional DDoS protection

3. **Restrict API Access**
   - Use firewall rules to limit access
   - Consider IP whitelisting for admin endpoints

4. **Monitor Audit Log**
   - Regularly review for suspicious activity
   - Set up alerts for unusual patterns

5. **Backup Database**
   ```bash
   sqlite3 trading.db ".backup trading_backup_$(date +%Y%m%d).db"
   ```

## 📈 Monitoring

### Key Metrics to Track

- Daily trade count and P&L
- Position sizes and exposure
- Reconciliation status
- Circuit breaker state
- Database size and performance

### Database Queries

```sql
-- Daily trade summary
SELECT 
    date(timestamp) as date,
    COUNT(*) as trades,
    SUM(CASE WHEN status='FILLED' THEN 1 ELSE 0 END) as filled,
    SUM(CASE WHEN status='REJECTED' THEN 1 ELSE 0 END) as rejected
FROM trades
GROUP BY date(timestamp)
ORDER BY date DESC;

-- Current positions
SELECT 
    ticker,
    SUM(CASE WHEN action='BUY' THEN quantity ELSE -quantity END) as net_position
FROM trades
WHERE status='FILLED'
GROUP BY ticker;

-- Recent reconciliation results
SELECT * FROM reconciliation_log
ORDER BY timestamp DESC
LIMIT 10;
```

## 🐛 Troubleshooting

### IBKR Connection Failed

**Symptoms:**
```
{"level":"ERROR","msg":"IBKR connection failed error=Connection refused"}
```

**Solutions:**
- Verify TWS/Gateway is running
- Check API settings are enabled in TWS
- Verify port number (7497 for paper, 7496 for live)
- Check firewall settings
- Ensure client ID is not in use

### Webhook Signature Invalid

**Symptoms:**
```
HTTP 401: Invalid signature
```

**Solutions:**
- Verify WEBHOOK_SECRET matches in both .env and signature calculation
- Check timestamp is current (within 30 seconds)
- Ensure JSON payload format is correct (no extra spaces)

### Position Mismatch

**Symptoms:**
```
{"level":"WARNING","msg":"Position mismatch detected"}
Kill switch activated
```

**Solutions:**
- Check audit log for unexpected fills
- Verify no manual trades were made in IBKR
- Review reconciliation history
- Manually adjust expected positions if needed:
  ```bash
  sqlite3 trading.db "UPDATE expected_positions SET quantity=0 WHERE ticker='AAPL'"
  ```

### Tests Timing Out

**Symptoms:**
```
❌ M1: Valid Signature: FAILED - Read timed out
```

**Solutions:**
- Ensure IBKR is connected
- Set `DRY_RUN=true` for faster tests
- Disable Telegram: `ENABLE_TELEGRAM=false`
- Check kill switch is not active

## 🎓 Prerequisites

- Python 3.9 or higher
- Interactive Brokers account (paper or live)
- TWS or IB Gateway installed
- TradingView account with webhook support
- Basic command line knowledge

## 💾 Dependencies

Install with: `pip install -r requirements.txt`

**Main Dependencies:**
- FastAPI + Uvicorn (web framework)
- IB-Insync (IBKR integration)
- aiosqlite (async database)
- pydantic (validation)
- python-telegram-bot (optional alerts)

## 📂 Project Structure

```
tv-ibkr-system/
├── m4_final.py                 # Main application
├── test/
│   └── m4_final_test.py       # Test suite
├── .env                        # Configuration (create from template)
├── .env.template               # Configuration template
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── trading.db                  # SQLite database (auto-created)
├── trading.db-wal              # Write-ahead log (auto-created)
└── trading.db-shm              # Shared memory (auto-created)
```

## 🚀 Deployment

### Development

```bash
uvicorn m4_final:app --host 0.0.0.0 --port 8000 --reload
```

### Production with systemd

Create `/etc/systemd/system/tv-ibkr.service`:

```ini
[Unit]
Description=TV-IBKR Trading System
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/tv-ibkr-system
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/uvicorn m4_final:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable tv-ibkr
sudo systemctl start tv-ibkr
sudo systemctl status tv-ibkr
```

### Production with Docker

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "m4_final:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t tv-ibkr .
docker run -d -p 8000:8000 --env-file .env tv-ibkr
```

## 📝 TradingView Webhook Setup

### 1. Create Alert in TradingView
- Open your strategy/indicator
- Click "Create Alert"
- Set condition
- Select "Webhook URL"

### 2. Webhook URL
```
https://your-domain.com/webhook
```

### 3. Message Format
```json
{
  "ticker": "{{ticker}}",
  "action": "BUY",
  "quantity": 10,
  "order_type": "MARKET",
  "strategy": "my_strategy",
  "timestamp": "{{timenow}}",
  "signature": "calculated_hmac_signature"
}
```

**Note:** You'll need to calculate the HMAC signature. See test file for example.

## 📚 Additional Documentation

- **Quick Start Guide**: See installation section above
- **Configuration Reference**: See `.env.template`
- **API Reference**: See API Endpoints section
- **Troubleshooting**: See Troubleshooting section

## ⚠️ Disclaimer

This software is provided for educational and research purposes. Trading involves risk. Past performance does not guarantee future results. Always test thoroughly with paper trading before using real money. The authors assume no liability for trading losses.

## 📊 System Status

**Status:** ✅ Production Ready  
**Version:** 3.0.0  
**Milestones:** 4/4 Complete  
**Test Coverage:** 14/14 Tests Passing (100%)  
**Documentation:** Complete  

## 🎯 What's Included

- ✅ Complete FastAPI application (`m4_final.py`)
- ✅ Comprehensive test suite (`test/m4_final_test.py`)
- ✅ All dependencies listed (`requirements.txt`)
- ✅ Configuration template (`.env.template`)
- ✅ Professional documentation (this README)
- ✅ Production-ready with all 4 milestones complete

## 🤝 Support

For issues or questions:
1. Check this documentation
2. Review logs for errors
3. Run test suite: `python test/m4_final_test.py`
4. Verify configuration in `.env`

## 📄 License

Team NAK - All Rights Reserved

