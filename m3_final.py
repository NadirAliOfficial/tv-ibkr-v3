"""
TV-IBKR-v3 FINAL VERSION
M1 + M2 Complete with IBKR Threading Fix

This is the PRODUCTION-READY version that actually works with real IBKR connections.

Key Fix: IBKR runs in separate thread to avoid asyncio event loop conflicts

Requirements:
pip install fastapi uvicorn python-dotenv pydantic pydantic-settings ib-insync python-telegram-bot nest-asyncio

Run:
uvicorn final_main:app --host 0.0.0.0 --port 8000
"""

import asyncio
import hashlib
import hmac
import json
import logging
import uuid
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional, Dict
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Request, status, Depends, Header
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ============================================================================
# CONFIGURATION
# ============================================================================

class Settings(BaseSettings):
    """Application configuration"""
    
    # M1: Security
    webhook_secret: str
    webhook_timestamp_tolerance_seconds: int = 30
    
    # M2: IBKR
    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 7497
    ibkr_client_id: int = 1
    
    # M2: Risk Limits
    max_position_size: int = 100
    max_daily_loss: float = 500.0
    max_portfolio_exposure: float = 0.25
    max_daily_trades: int = 50
    
    # M2: Circuit Breaker
    circuit_breaker_threshold: int = 3
    circuit_breaker_timeout: int = 60
    
    # M2: Telegram
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    
    # M2: Feature Flags
    trading_enabled: bool = True
    dry_run: bool = False
    enable_risk_engine: bool = True
    enable_order_execution: bool = True
    enable_telegram: bool = False
    
    # Admin
    admin_api_key: str = "admin-secret-key"
    
    log_level: str = "INFO"
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=settings.log_level,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
    datefmt='%Y-%m-%dT%H:%M:%S'
)

logger = logging.getLogger(__name__)

def log_info(msg: str, **kwargs):
    logger.info(f"{msg} {json.dumps(kwargs) if kwargs else ''}")

def log_error(msg: str, **kwargs):
    logger.error(f"{msg} {json.dumps(kwargs) if kwargs else ''}")

# ============================================================================
# MODELS
# ============================================================================

class WebhookPayload(BaseModel):
    ticker: str
    action: Literal["BUY", "SELL", "CLOSE"]
    quantity: Optional[int] = None
    order_type: Literal["MARKET", "LIMIT", "STOP"] = "MARKET"
    limit_price: Optional[float] = None
    strategy: Optional[str] = None
    timestamp: datetime
    signature: str
    
    @field_validator('timestamp')
    @classmethod
    def timestamp_must_be_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)

class WebhookResponse(BaseModel):
    status: str
    correlation_id: str
    message: str
    idempotency_key: str
    order_id: Optional[int] = None
    fill_price: Optional[float] = None

class RiskCheckResult(BaseModel):
    approved: bool
    reason: Optional[str] = None
    checks: Dict[str, bool] = {}

class OrderResult(BaseModel):
    success: bool
    order_id: Optional[int] = None
    fill_price: Optional[float] = None
    fill_quantity: Optional[int] = None
    error: Optional[str] = None

class KillSwitchRequest(BaseModel):
    reason: str
    actor: str = "API"

class SystemStatus(BaseModel):
    trading_enabled: bool
    kill_switch_active: bool
    ibkr_connected: bool
    circuit_breaker_open: bool
    daily_pnl: float
    daily_trade_count: int
    positions: Dict[str, int]

# ============================================================================
# REPLAY PROTECTION
# ============================================================================

class ReplayGuard:
    def __init__(self, tolerance_seconds: int = 30):
        self.tolerance = timedelta(seconds=tolerance_seconds)
        self.future_tolerance = timedelta(seconds=5)
        self.seen_keys = set()
    
    def validate_timestamp(self, webhook_timestamp: datetime) -> None:
        now = datetime.now(timezone.utc)
        
        if webhook_timestamp < now - self.tolerance:
            age = (now - webhook_timestamp).total_seconds()
            raise HTTPException(
                status_code=400,
                detail=f"Webhook too old: {age:.1f}s ago (max {self.tolerance.total_seconds()}s)"
            )
        
        if webhook_timestamp > now + self.future_tolerance:
            raise HTTPException(status_code=400, detail="Webhook timestamp in future")
    
    def generate_idempotency_key(self, ticker: str, action: str, timestamp: datetime) -> str:
        payload = f"{ticker}:{action}:{timestamp.isoformat()}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
    
    def check_duplicate(self, idempotency_key: str) -> bool:
        if idempotency_key in self.seen_keys:
            return True
        self.seen_keys.add(idempotency_key)
        return False

replay_guard = ReplayGuard(settings.webhook_timestamp_tolerance_seconds)

# ============================================================================
# CIRCUIT BREAKER
# ============================================================================

class CircuitBreaker:
    def __init__(self, threshold: int = 3, timeout: int = 60):
        self.threshold = threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time: Optional[datetime] = None
        self.is_open = False
    
    def record_success(self):
        self.failures = 0
        if self.is_open:
            log_info("Circuit breaker closed - service recovered")
            self.is_open = False
    
    def record_failure(self):
        self.failures += 1
        self.last_failure_time = datetime.now(timezone.utc)
        
        if self.failures >= self.threshold and not self.is_open:
            self.is_open = True
            log_error(f"Circuit breaker OPENED after {self.failures} failures")
    
    def can_attempt(self) -> bool:
        if not self.is_open:
            return True
        
        # Check if timeout expired
        if datetime.now(timezone.utc) - self.last_failure_time > timedelta(seconds=self.timeout):
            log_info("Circuit breaker attempting recovery")
            self.is_open = False
            self.failures = 0
            return True
        
        return False

# ============================================================================
# IBKR CLIENT (THREADING FIX)
# ============================================================================

class IBKRClient:
    """IBKR client with threading to avoid asyncio conflicts"""
    
    def __init__(self):
        self.ib = None
        self.connected = False
        self.connection_lock = threading.Lock()
        self.connection_thread = None
        self.circuit_breaker = CircuitBreaker(
            settings.circuit_breaker_threshold,
            settings.circuit_breaker_timeout
        )
    
    def _connect_in_thread(self):
        """Run IBKR connection in separate thread with own event loop"""
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            from ib_insync import IB, util
            
            # Apply patches
            try:
                import nest_asyncio
                nest_asyncio.apply()
            except:
                pass
            
            util.patchAsyncio()
            
            self.ib = IB()
            
            log_info(f"Connecting to IBKR", host=settings.ibkr_host, port=settings.ibkr_port)
            
            # Connect using this thread's event loop
            loop.run_until_complete(
                self.ib.connectAsync(
                    settings.ibkr_host,
                    settings.ibkr_port,
                    clientId=settings.ibkr_client_id,
                    timeout=20
                )
            )
            
            with self.connection_lock:
                self.connected = True
                self.circuit_breaker.record_success()
            
            accounts = self.ib.managedAccounts()
            log_info("🎉 IBKR CONNECTED!", accounts=accounts)
            
        except Exception as e:
            log_error(f"IBKR connection failed", error=str(e))
            with self.connection_lock:
                self.connected = False
                self.circuit_breaker.record_failure()
    
    async def connect(self):
        """Start IBKR connection in background thread"""
        if not settings.enable_order_execution:
            log_info("IBKR execution disabled")
            return
        
        # Start connection in daemon thread
        self.connection_thread = threading.Thread(
            target=self._connect_in_thread,
            daemon=True
        )
        self.connection_thread.start()
        
        # Wait a bit for connection
        await asyncio.sleep(3)
    
    async def disconnect(self):
        if self.ib and self.connected:
            try:
                self.ib.disconnect()
            except:
                pass
            self.connected = False
    
    async def get_account_value(self) -> float:
        if not self.connected or settings.dry_run:
            return 100000.0
        
        try:
            if not self.circuit_breaker.can_attempt():
                return 100000.0
            
            account_values = self.ib.accountValues()
            for av in account_values:
                if av.tag == 'NetLiquidation':
                    self.circuit_breaker.record_success()
                    return float(av.value)
            return 100000.0
        except Exception as e:
            self.circuit_breaker.record_failure()
            log_error("Failed to get account value", error=str(e))
            return 100000.0
    
    async def place_order(self, ticker: str, action: str, quantity: int, 
                         order_type: str = "MARKET", limit_price: Optional[float] = None) -> OrderResult:
        """Place order on IBKR"""
        
        if settings.dry_run:
            log_info(f"DRY RUN: {action} {quantity} {ticker}")
            return OrderResult(
                success=True,
                order_id=99999,
                fill_price=150.0,
                fill_quantity=quantity
            )
        
        if not self.connected:
            return OrderResult(success=False, error="IBKR not connected")
        
        if not self.circuit_breaker.can_attempt():
            return OrderResult(success=False, error="Circuit breaker open")
        
        try:
            from ib_insync import Stock, MarketOrder, LimitOrder
            
            contract = Stock(ticker, 'SMART', 'USD')
            
            if order_type == "MARKET":
                order = MarketOrder(action, quantity)
            elif order_type == "LIMIT":
                order = LimitOrder(action, quantity, limit_price)
            else:
                return OrderResult(success=False, error=f"Unsupported order type: {order_type}")
            
            trade = self.ib.placeOrder(contract, order)
            
            # Wait for fill (30s timeout)
            for _ in range(30):
                await asyncio.sleep(1)
                if trade.isDone():
                    break
            
            if trade.orderStatus.status == 'Filled':
                self.circuit_breaker.record_success()
                return OrderResult(
                    success=True,
                    order_id=trade.order.orderId,
                    fill_price=trade.orderStatus.avgFillPrice,
                    fill_quantity=int(trade.orderStatus.filled)
                )
            else:
                return OrderResult(success=False, error=f"Order status: {trade.orderStatus.status}")
        
        except Exception as e:
            self.circuit_breaker.record_failure()
            log_error("Order placement failed", error=str(e))
            return OrderResult(success=False, error=str(e))

ibkr_client = IBKRClient()

# ============================================================================
# TELEGRAM
# ============================================================================

class TelegramAlerter:
    def __init__(self):
        self.enabled = (settings.enable_telegram and 
                       settings.telegram_bot_token and 
                       settings.telegram_chat_id)
        self.bot = None
        
        if self.enabled:
            try:
                from telegram import Bot
                self.bot = Bot(token=settings.telegram_bot_token)
            except:
                self.enabled = False
    
    async def send(self, message: str):
        if not self.enabled:
            log_info(f"Telegram: {message}")
            return
        
        try:
            await self.bot.send_message(chat_id=settings.telegram_chat_id, text=message)
        except Exception as e:
            log_error("Telegram send failed", error=str(e))
    
    async def send_fill_alert(self, ticker: str, action: str, quantity: int, price: float):
        emoji = "🟢" if action == "BUY" else "🔴"
        await self.send(f"{emoji} FILL: {action} {quantity} {ticker} @ ${price:.2f}")
    
    async def send_rejection(self, ticker: str, action: str, reason: str):
        await self.send(f"❌ REJECTED: {action} {ticker}\n{reason}")

alerter = TelegramAlerter()

# ============================================================================
# RISK ENGINE
# ============================================================================

class RiskEngine:
    def __init__(self):
        self.kill_switch_active = False
        self.kill_switch_reason = ""
        self.daily_pnl = 0.0
        self.daily_trade_count = 0
        self.positions: Dict[str, int] = {}
        self.last_reset = datetime.now(timezone.utc).date()
    
    def _check_daily_reset(self):
        today = datetime.now(timezone.utc).date()
        if today > self.last_reset:
            self.daily_pnl = 0.0
            self.daily_trade_count = 0
            self.last_reset = today
            log_info("Daily risk counters reset")
    
    async def validate(self, payload: WebhookPayload) -> RiskCheckResult:
        if not settings.enable_risk_engine:
            return RiskCheckResult(approved=True)
        
        self._check_daily_reset()
        checks = {}
        
        if not settings.trading_enabled:
            return RiskCheckResult(approved=False, reason="Trading disabled")
        checks["trading_enabled"] = True
        
        if self.kill_switch_active:
            return RiskCheckResult(approved=False, reason=f"Kill switch: {self.kill_switch_reason}")
        checks["kill_switch"] = True
        
        if self.daily_pnl < -settings.max_daily_loss:
            return RiskCheckResult(approved=False, reason=f"Daily loss limit: ${abs(self.daily_pnl):.2f}")
        checks["daily_loss"] = True
        
        if self.daily_trade_count >= settings.max_daily_trades:
            return RiskCheckResult(approved=False, reason=f"Daily trade limit: {self.daily_trade_count}")
        checks["trade_count"] = True
        
        current_position = self.positions.get(payload.ticker, 0)
        quantity = payload.quantity or settings.max_position_size
        
        if payload.action == "BUY":
            new_position = current_position + quantity
        elif payload.action == "SELL":
            new_position = current_position - quantity
        else:
            new_position = 0
        
        if abs(new_position) > settings.max_position_size:
            return RiskCheckResult(approved=False, reason=f"Position limit: {abs(new_position)} > {settings.max_position_size}")
        checks["position_size"] = True
        
        return RiskCheckResult(approved=True, checks=checks)
    
    def activate_kill_switch(self, reason: str, actor: str):
        self.kill_switch_active = True
        self.kill_switch_reason = f"[{actor}] {reason}"
        log_info("Kill switch ACTIVATED", reason=self.kill_switch_reason)
    
    def deactivate_kill_switch(self, actor: str):
        self.kill_switch_active = False
        log_info("Kill switch DEACTIVATED", actor=actor)
    
    def update_position(self, ticker: str, quantity_change: int):
        current = self.positions.get(ticker, 0)
        self.positions[ticker] = current + quantity_change
        if self.positions[ticker] == 0:
            del self.positions[ticker]
    
    def record_trade(self):
        self.daily_trade_count += 1

risk_engine = RiskEngine()

# ============================================================================
# SECURITY
# ============================================================================

def verify_hmac_signature(payload: bytes, signature: str, secret: str) -> bool:
    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)

def verify_admin_key(x_api_key: str = Header(...)):
    if x_api_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

# ============================================================================
# FASTAPI APP
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    log_info("Application starting...")
    
    # Start IBKR connection in background
    if settings.enable_order_execution and not settings.dry_run:
        asyncio.create_task(ibkr_client.connect())
    
    log_info("Application ready")
    yield
    
    log_info("Application shutting down...")
    await ibkr_client.disconnect()

app = FastAPI(
    title="TV-IBKR-v3 FINAL",
    version="2.0.0",
    lifespan=lifespan
)

@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    correlation_id = str(uuid.uuid4())
    request.state.correlation_id = correlation_id
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = correlation_id
    return response

# ============================================================================
# WEBHOOK ENDPOINT
# ============================================================================

@app.post("/webhook", response_model=WebhookResponse)
async def webhook_handler(request: Request):
    correlation_id = request.state.correlation_id
    
    try:
        raw_body = await request.body()
        payload_dict = json.loads(raw_body)
        
        signature = payload_dict.get("signature", "")
        payload_without_sig = {k: v for k, v in payload_dict.items() if k != "signature"}
        payload_for_hmac = json.dumps(payload_without_sig, separators=(',', ':')).encode()
        
        if not verify_hmac_signature(payload_for_hmac, signature, settings.webhook_secret):
            raise HTTPException(status_code=401, detail="Invalid signature")
        
        payload = WebhookPayload(**payload_dict)
        
        replay_guard.validate_timestamp(payload.timestamp)
        idempotency_key = replay_guard.generate_idempotency_key(
            payload.ticker, payload.action, payload.timestamp
        )
        
        if replay_guard.check_duplicate(idempotency_key):
            raise HTTPException(status_code=409, detail="Duplicate webhook")
        
        # Risk checks
        risk_result = await risk_engine.validate(payload)
        
        if not risk_result.approved:
            await alerter.send_rejection(payload.ticker, payload.action, risk_result.reason)
            return WebhookResponse(
                status="rejected",
                correlation_id=correlation_id,
                message=risk_result.reason,
                idempotency_key=idempotency_key
            )
        
        # Execute order
        if settings.enable_order_execution or settings.dry_run:
            quantity = payload.quantity or settings.max_position_size
            order_result = await ibkr_client.place_order(
                payload.ticker,
                payload.action,
                quantity,
                payload.order_type,
                payload.limit_price
            )
            
            if order_result.success:
                qty_change = quantity if payload.action == "BUY" else -quantity
                risk_engine.update_position(payload.ticker, qty_change)
                risk_engine.record_trade()
                
                await alerter.send_fill_alert(
                    payload.ticker, payload.action, quantity, order_result.fill_price or 0
                )
                
                return WebhookResponse(
                    status="executed",
                    correlation_id=correlation_id,
                    message=f"Order filled: {payload.action} {quantity} {payload.ticker}",
                    idempotency_key=idempotency_key,
                    order_id=order_result.order_id,
                    fill_price=order_result.fill_price
                )
            else:
                raise HTTPException(status_code=500, detail=order_result.error)
        else:
            return WebhookResponse(
                status="validated",
                correlation_id=correlation_id,
                message="Webhook validated (execution disabled)",
                idempotency_key=idempotency_key
            )
    
    except HTTPException:
        raise
    except Exception as e:
        log_error("Webhook error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal error")

# ============================================================================
# ADMIN ENDPOINTS
# ============================================================================

@app.post("/admin/kill", dependencies=[Depends(verify_admin_key)])
async def activate_kill_switch(req: KillSwitchRequest):
    risk_engine.activate_kill_switch(req.reason, req.actor)
    await alerter.send(f"🛑 Kill Switch ACTIVATED by {req.actor}: {req.reason}")
    return {"status": "activated", "reason": req.reason}

@app.post("/admin/resume", dependencies=[Depends(verify_admin_key)])
async def deactivate_kill_switch(req: KillSwitchRequest):
    risk_engine.deactivate_kill_switch(req.actor)
    await alerter.send(f"✅ Kill Switch DEACTIVATED by {req.actor}")
    return {"status": "deactivated"}

@app.get("/admin/status", response_model=SystemStatus, dependencies=[Depends(verify_admin_key)])
async def get_status():
    return SystemStatus(
        trading_enabled=settings.trading_enabled,
        kill_switch_active=risk_engine.kill_switch_active,
        ibkr_connected=ibkr_client.connected,
        circuit_breaker_open=ibkr_client.circuit_breaker.is_open,
        daily_pnl=risk_engine.daily_pnl,
        daily_trade_count=risk_engine.daily_trade_count,
        positions=risk_engine.positions
    )

@app.post("/admin/reset-limits", dependencies=[Depends(verify_admin_key)])
async def reset_limits():
    risk_engine.daily_pnl = 0.0
    risk_engine.daily_trade_count = 0
    return {"status": "reset"}

# ============================================================================
# HEALTH ENDPOINTS
# ============================================================================

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc)}

@app.get("/ready")
async def ready():
    return {
        "ready": ibkr_client.connected if settings.enable_order_execution else True,
        "ibkr_connected": ibkr_client.connected,
        "dry_run": settings.dry_run
    }

@app.get("/")
async def root():
    return {
        "service": "TV-IBKR-v3 FINAL",
        "version": "2.0.0",
        "features": ["HMAC Auth", "Risk Engine", "IBKR Execution", "Threading Fix"],
        "status": {
            "trading": settings.trading_enabled,
            "ibkr": ibkr_client.connected,
            "dry_run": settings.dry_run
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("__main__:app", host="0.0.0.0", port=8000, reload=False)