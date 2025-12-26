
# TV-IBKR-v3 – Milestone 1
### TradingView Webhook Ingestion & Security Layer

This project implements **Milestone 1 (M1)** of the TV-IBKR-v3 system:  
a secure and production-ready **TradingView webhook ingestion service** designed to safely receive and validate trading signals before any risk management or execution logic.

---

## ✨ Features (M1 Scope)

- Secure webhook ingestion via HTTPS
- HMAC-SHA256 signature verification
- Strict payload schema validation
- Timestamp freshness checks for replay protection
- Idempotency key generation to prevent duplicate processing
- Correlation ID–based structured logging
- Health (`/health`) and readiness (`/ready`) endpoints
- Single-file FastAPI application for simplicity

This milestone focuses **only on ingestion and security**.  
Risk controls, order execution, database persistence, and broker integration are handled in later milestones.

---

## 📁 Project Structure

```text
.
├── main.py          # FastAPI application (single-file M1)
├── requirements.txt
├── .env             # Environment variables (not committed)
└── README.md
````

---

## ⚙️ Configuration

Create a `.env` file in the project root:

```env
WEBHOOK_SECRET=your_shared_hmac_secret
WEBHOOK_TIMESTAMP_TOLERANCE_SECONDS=30
LOG_LEVEL=INFO
```

---

## 🚀 Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the server:

```bash
uvicorn main:app --reload
```

The service will be available at:

```
http://localhost:8000
```

Interactive API documentation:

```
http://localhost:8000/docs
```

---

## 🔐 Webhook Validation Flow

1. Raw request body is verified using HMAC-SHA256
2. Payload structure is validated
3. Timestamp is checked against the allowed window
4. Idempotency key is generated and checked
5. Valid webhooks are acknowledged with `200 OK`

---

## 🧪 Testing

The application includes a helper to generate a valid test webhook payload.

You can run:

```bash
python main.py
```

Copy the generated payload and send it to:

```
POST http://localhost:8000/webhook
```

---

## 🩺 Health Checks

* `GET /health`
  Confirms the service is running

* `GET /ready`
  Indicates whether the service is ready to accept traffic

---

## 📌 Notes

* Idempotency tracking is in-memory for Milestone 1
* Persistence and reconciliation are added in later milestones
* Designed to be deployed behind Cloudflare or a similar edge layer

---

## 🧭 Next Steps

* Risk engine and safety controls
* Order execution integration
* Database persistence and reconciliation
* Admin controls and audit logging

---

## 📄 License

Internal / Confidential

