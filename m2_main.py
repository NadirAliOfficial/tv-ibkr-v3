

import asyncio
import hashlib
import hmac
import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Literal, Optional, Dict, Any
from decimal import Decimal

from fastapi import FastAPI, HTTPException, Request, Response, status
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ============================================================================
# CONFIGURATION
# ============================================================================

class Settings(BaseSettings):
    """Application configuration from environment variables."""
    
    # M1: Webhook Security
    webhook_secret: str = Field(..., description="HMAC secret for webhook verification")
    webhook_timestamp_tolerance_seconds: int = Field(default=30)
    
    # M2: IBKR Connection
    ibkr_host: str = Field(default="127.0.0.1")
    ibkr_port: int = Field(default=4001, description="4001=paper, 4002=live")
    ibkr_client_id: int = Field(default=1)
    ibkr_account: Optional[str] = Field(default=None)
    
    # M2: Risk Limits
    max_position_size: int = Field(default=100)
    max_daily_loss: float = Field(default=500.0)
    max_portfolio_exposure: float = Field(default=0.25)
    max_daily_trades: int = Field(default=50)
    
    # M2: Circuit Breaker
    circuit_breaker_threshold: int = Field(default=3)
    circuit_breaker_timeout: int = Field(default=60)
    
    # M2: Telegram Alerts
    telegram_bot_token: Optional[str] = Field(default=None)
    telegram_chat_id: Optional[str] = Field(default=None)
    
    # M2: Feature Flags
    trading_enabled: bool = Field(default=True)
    dry_run: bool = Field(default=False)
    enable_risk_engine: bool = Field(default=True)
    enable_order_execution: bool = Field(default=True)
    enable_telegram: bool = Field(default=False)
    
    # Logging
    log_level: str = Field(default="INFO")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=settings.log_level,
    format='{"timestamp":"%(asctime)s","level":"%(levelname)s","correlation_id":"%(correlation_id)s","message":"%(message)s","extra":%(extra)s}',
    datefmt='%Y-%m-%dT%H:%M:%S'
)

class CorrelationFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = 'N/A'
        if not hasattr(record, 'extra'):
            record.extra = '{}'
        return True

logger = logging.getLogger(__name__)
logger.addFilter(CorrelationFilter())

def log_with_context(level: str, message: str, correlation_id: str = "N/A", **kwargs):
    extra_data = json.dumps(kwargs) if kwargs else '{}'
    getattr(logger, level)(message, extra={'correlation_id': correlation_id, 'extra': extra_data})

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class WebhookPayload(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10)
    action: Literal["BUY", "SELL", "CLOSE"]
    quantity: Optional[int] = Field(default=None, gt=0)
    order_type: Literal["MARKET", "LIMIT", "STOP"] = Field(default="MARKET")
    limit_price: Optional[float] = Field(default=None, gt=0)
    strategy: Optional[str] = Field(default=None, max_length=50)
    timestamp: datetime
    signature: str = Field(..., min_length=32)
    
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
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "STALE_WEBHOOK",
                    "message": f"Webhook timestamp too old: {age:.1f}s ago",
                    "timestamp": webhook_timestamp.isoformat(),
                    "max_age_seconds": self.tolerance.total_seconds()
                }
            )
        
        if webhook_timestamp > now + self.future_tolerance:
            delta = (webhook_timestamp - now).total_seconds()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "FUTURE_WEBHOOK",
                    "message": f"Webhook timestamp in future: +{delta:.1f}s",
                    "timestamp": webhook_timestamp.isoformat()
                }
            )
    
    def generate_idempotency_key(self, ticker: str, action: str, timestamp: datetime) -> str:
        payload = f"{ticker}:{action}:{timestamp.isoformat()}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
    
    def check_duplicate(self, idempotency_key: str) -> bool:
        if idempotency_key in self.seen_keys:
            return True
        self.seen_keys.add(idempotency_key)
        return False

replay_guard = ReplayGuard(tolerance_seconds=settings.webhook_timestamp_tolerance_seconds)

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
    
    async def call(self, func, *args, **kwargs):
        if self.is_open:
            if datetime.now(timezone.utc) - self.last_failure_time > timedelta(seconds=self.timeout):
                log_with_context("info", "Circuit breaker: attempting recovery")
                self.is_open = False
                self.failures = 0
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            self.failures = 0
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure_time = datetime.now(timezone.utc)
            
            if self.failures >= self.threshold:
                self.is_open = True
                log_with_context("error", f"Circuit breaker OPENED after {self.failures} failures")
            
            raise e

# ============================================================================
# IBKR CLIENT
# ============================================================================

class IBKRClient:
    def __init__(self):
        self.ib = None
        self.connected = False
        self.circuit_breaker = CircuitBreaker(
            threshold=settings.circuit_breaker_threshold,
            timeout=settings.circuit_breaker_timeout
        )
    
    async def connect(self):
        if not settings.enable_order_execution:
            log_with_context("info", "IBKR execution disabled - skipping connection")
            return
        
        try:
            from ib_insync import IB, util
            self.ib = IB()
            
            await self.circuit_breaker.call(
                self.ib.connectAsync,
                settings.ibkr_host,
                settings.ibkr_port,
                clientId=settings.ibkr_client_id
            )
            
            self.connected = True
            log_with_context("info", "IBKR connected successfully", 
                           host=settings.ibkr_host, port=settings.ibkr_port)
        except Exception as e:
            log_with_context("error", f"IBKR connection failed: {e}")
            self.connected = False
            if settings.enable_order_execution and not settings.dry_run:
                raise
    
    async def disconnect(self):
        if self.ib and self.connected:
            self.ib.disconnect()
            self.connected = False
            log_with_context("info", "IBKR disconnected")
    
    async def get_account_value(self) -> float:
        if not self.connected or settings.dry_run:
            return 100000.0  # Mock value
        
        try:
            account_values = await self.circuit_breaker.call(self.ib.accountValues)
            for av in account_values:
                if av.tag == 'NetLiquidation':
                    return float(av.value)
            return 100000.0
        except Exception as e:
            log_with_context("error", f"Failed to get account value: {e}")
            return 100000.0
    
    async def get_positions(self) -> Dict[str, int]:
        if not self.connected or settings.dry_run:
            return {}
        
        try:
            positions = await self.circuit_breaker.call(self.ib.positions)
            return {pos.contract.symbol: int(pos.position) for pos in positions}
        except Exception as e:
            log_with_context("error", f"Failed to get positions: {e}")
            return {}
    
    async def place_order(self, ticker: str, action: str, quantity: int, order_type: str = "MARKET", limit_price: Optional[float] = None) -> OrderResult:
        if settings.dry_run:
            log_with_context("info", f"DRY RUN: Would place order {action} {quantity} {ticker}")
            return OrderResult(
                success=True,
                order_id=99999,
                fill_price=150.0,  # Mock price
                fill_quantity=quantity
            )
        
        if not self.connected:
            return OrderResult(success=False, error="IBKR not connected")
        
        try:
            from ib_insync import Stock, MarketOrder, LimitOrder, StopOrder
            
            # Create contract
            contract = Stock(ticker, 'SMART', 'USD')
            
            # Create order
            if order_type == "MARKET":
                order = MarketOrder(action, quantity)
            elif order_type == "LIMIT":
                order = LimitOrder(action, quantity, limit_price)
            elif order_type == "STOP":
                order = StopOrder(action, quantity, limit_price)
            else:
                return OrderResult(success=False, error=f"Unknown order type: {order_type}")
            
            # Place order
            trade = await self.circuit_breaker.call(self.ib.placeOrder, contract, order)
            
            # Wait for fill (timeout 30s)
            for _ in range(30):
                await asyncio.sleep(1)
                if trade.isDone():
                    break
            
            if trade.orderStatus.status == 'Filled':
                return OrderResult(
                    success=True,
                    order_id=trade.order.orderId,
                    fill_price=trade.orderStatus.avgFillPrice,
                    fill_quantity=int(trade.orderStatus.filled)
                )
            else:
                return OrderResult(
                    success=False,
                    error=f"Order not filled: {trade.orderStatus.status}"
                )
        
        except Exception as e:
            log_with_context("error", f"Order placement failed: {e}")
            return OrderResult(success=False, error=str(e))

ibkr_client = IBKRClient()

# ============================================================================
# TELEGRAM ALERTER
# ============================================================================

class TelegramAlerter:
    def __init__(self):
        self.enabled = settings.enable_telegram and settings.telegram_bot_token and settings.telegram_chat_id
        self.bot = None
        
        if self.enabled:
            try:
                from telegram import Bot
                self.bot = Bot(token=settings.telegram_bot_token)
            except Exception as e:
                log_with_context("warning", f"Telegram init failed: {e}")
                self.enabled = False
    
    async def send(self, message: str):
        if not self.enabled:
            log_with_context("info", f"Telegram disabled - would send: {message}")
            return
        
        try:
            await self.bot.send_message(chat_id=settings.telegram_chat_id, text=message)
        except Exception as e:
            log_with_context("error", f"Telegram send failed: {e}")
    
    async def send_fill_alert(self, ticker: str, action: str, quantity: int, price: float):
        emoji = "🟢" if action == "BUY" else "🔴"
        await self.send(f"{emoji} FILL: {action} {quantity} {ticker} @ ${price:.2f}")
    
    async def send_rejection(self, ticker: str, action: str, reason: str):
        await self.send(f"❌ REJECTED: {action} {ticker}\nReason: {reason}")
    
    async def send_kill_switch_alert(self, active: bool, reason: str, actor: str):
        status = "🛑 ACTIVATED" if active else "✅ DEACTIVATED"
        await self.send(f"Kill Switch {status}\nBy: {actor}\nReason: {reason}")

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
            log_with_context("info", "Daily risk counters reset")
    
    async def validate(self, payload: WebhookPayload, correlation_id: str) -> RiskCheckResult:
        if not settings.enable_risk_engine:
            return RiskCheckResult(approved=True, checks={"risk_engine": False})
        
        self._check_daily_reset()
        
        checks = {}
        
        # Check 1: Trading enabled
        if not settings.trading_enabled:
            return RiskCheckResult(approved=False, reason="Trading globally disabled", checks=checks)
        checks["trading_enabled"] = True
        
        # Check 2: Kill switch
        if self.kill_switch_active:
            return RiskCheckResult(approved=False, reason=f"Kill switch active: {self.kill_switch_reason}", checks=checks)
        checks["kill_switch"] = True
        
        # Check 3: Daily loss limit
        if self.daily_pnl < -settings.max_daily_loss:
            return RiskCheckResult(approved=False, reason=f"Daily loss limit exceeded: ${abs(self.daily_pnl):.2f}", checks=checks)
        checks["daily_loss"] = True
        
        # Check 4: Daily trade count
        if self.daily_trade_count >= settings.max_daily_trades:
            return RiskCheckResult(approved=False, reason=f"Daily trade limit exceeded: {self.daily_trade_count}", checks=checks)
        checks["trade_count"] = True
        
        # Check 5: Position size
        current_position = self.positions.get(payload.ticker, 0)
        new_quantity = payload.quantity or settings.max_position_size
        
        if payload.action == "BUY":
            new_position = current_position + new_quantity
        elif payload.action == "SELL":
            new_position = current_position - new_quantity
        else:  # CLOSE
            new_position = 0
        
        if abs(new_position) > settings.max_position_size:
            return RiskCheckResult(approved=False, reason=f"Position size limit: {abs(new_position)} > {settings.max_position_size}", checks=checks)
        checks["position_size"] = True
        
        # Check 6: Portfolio exposure (simplified - would need account value)
        account_value = await ibkr_client.get_account_value()
        estimated_exposure = abs(new_position) * 150.0  # Mock price
        exposure_pct = estimated_exposure / account_value
        
        if exposure_pct > settings.max_portfolio_exposure:
            return RiskCheckResult(approved=False, reason=f"Exposure limit: {exposure_pct:.1%} > {settings.max_portfolio_exposure:.1%}", checks=checks)
        checks["exposure"] = True
        
        log_with_context("info", "All risk checks passed", correlation_id, checks=checks)
        return RiskCheckResult(approved=True, checks=checks)
    
    def activate_kill_switch(self, reason: str, actor: str = "SYSTEM"):
        self.kill_switch_active = True
        self.kill_switch_reason = f"[{actor}] {reason}"
        log_with_context("warning", "Kill switch ACTIVATED", reason=self.kill_switch_reason)
    
    def deactivate_kill_switch(self, reason: str, actor: str = "ADMIN"):
        self.kill_switch_active = False
        log_with_context("info", "Kill switch DEACTIVATED", actor=actor, reason=reason)
    
    def update_position(self, ticker: str, quantity_change: int):
        current = self.positions.get(ticker, 0)
        self.positions[ticker] = current + quantity_change
        if self.positions[ticker] == 0:
            del self.positions[ticker]
    
    def record_trade(self, pnl: float = 0.0):
        self.daily_trade_count += 1
        self.daily_pnl += pnl

risk_engine = RiskEngine()

# ============================================================================
# SECURITY
# ============================================================================

def verify_hmac_signature(payload: bytes, signature: str, secret: str) -> bool:
    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    log_with_context("info", "Application starting...")
    if settings.enable_order_execution and not settings.dry_run:
        await ibkr_client.connect()
    log_with_context("info", "Application ready")
    
    yield
    
    # Shutdown
    log_with_context("info", "Application shutting down...")
    await ibkr_client.disconnect()
    log_with_context("info", "Shutdown complete")

app = FastAPI(
    title="TV-IBKR-v3",
    description="TradingView to Interactive Brokers - M1+M2 Integration",
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
# WEBHOOK ENDPOINT (M1 + M2 INTEGRATED)
# ============================================================================

@app.post("/webhook", response_model=WebhookResponse)
async def webhook_handler(request: Request):
    correlation_id = request.state.correlation_id
    
    try:
        # M1: Security validations
        raw_body = await request.body()
        log_with_context("info", "Webhook received", correlation_id, content_length=len(raw_body))
        
        try:
            payload_dict = json.loads(raw_body)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail="Invalid JSON")
        
        signature = payload_dict.get("signature", "")
        payload_without_sig = {k: v for k, v in payload_dict.items() if k != "signature"}
        payload_for_hmac = json.dumps(payload_without_sig, separators=(',', ':')).encode()
        
        if not verify_hmac_signature(payload_for_hmac, signature, settings.webhook_secret):
            log_with_context("warning", "HMAC verification failed", correlation_id)
            raise HTTPException(status_code=401, detail="Invalid webhook signature")
        
        try:
            payload = WebhookPayload(**payload_dict)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid payload schema: {str(e)}")
        
        replay_guard.validate_timestamp(payload.timestamp)
        idempotency_key = replay_guard.generate_idempotency_key(payload.ticker, payload.action, payload.timestamp)
        
        if replay_guard.check_duplicate(idempotency_key):
            raise HTTPException(status_code=409, detail={"error": "DUPLICATE_WEBHOOK", "idempotency_key": idempotency_key})
        
        log_with_context("info", "Webhook validated (M1)", correlation_id, ticker=payload.ticker, action=payload.action)
        
        # M2: Risk checks
        risk_result = await risk_engine.validate(payload, correlation_id)
        
        if not risk_result.approved:
            log_with_context("warning", "Risk check failed", correlation_id, reason=risk_result.reason)
            await alerter.send_rejection(payload.ticker, payload.action, risk_result.reason)
            return WebhookResponse(
                status="rejected",
                correlation_id=correlation_id,
                message=risk_result.reason,
                idempotency_key=idempotency_key
            )
        
        log_with_context("info", "Risk checks passed", correlation_id)
        
        # M2: Execute order
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
                # Update risk state
                qty_change = quantity if payload.action == "BUY" else -quantity
                risk_engine.update_position(payload.ticker, qty_change)
                risk_engine.record_trade()
                
                await alerter.send_fill_alert(payload.ticker, payload.action, quantity, order_result.fill_price or 0)
                
                log_with_context("info", "Order executed successfully", correlation_id, 
                               order_id=order_result.order_id, fill_price=order_result.fill_price)
                
                return WebhookResponse(
                    status="executed",
                    correlation_id=correlation_id,
                    message=f"Order filled: {payload.action} {quantity} {payload.ticker}",
                    idempotency_key=idempotency_key,
                    order_id=order_result.order_id,
                    fill_price=order_result.fill_price
                )
            else:
                log_with_context("error", "Order execution failed", correlation_id, error=order_result.error)
                await alerter.send(f"❌ Execution failed: {order_result.error}")
                raise HTTPException(status_code=500, detail=f"Order execution failed: {order_result.error}")
        else:
            # Execution disabled - just validate
            return WebhookResponse(
                status="validated",
                correlation_id=correlation_id,
                message=f"Webhook validated (execution disabled): {payload.action} {payload.ticker}",
                idempotency_key=idempotency_key
            )
    
    except HTTPException:
        raise
    except Exception as e:
        log_with_context("error", "Unexpected error", correlation_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

# ============================================================================
# ADMIN ENDPOINTS (M2)
# ============================================================================

@app.post("/admin/kill")
async def activate_kill_switch(req: KillSwitchRequest):
    risk_engine.activate_kill_switch(req.reason, req.actor)
    await alerter.send_kill_switch_alert(True, req.reason, req.actor)
    return {"status": "activated", "reason": req.reason, "actor": req.actor}

@app.post("/admin/resume")
async def deactivate_kill_switch(req: KillSwitchRequest):
    risk_engine.deactivate_kill_switch(req.reason, req.actor)
    await alerter.send_kill_switch_alert(False, req.reason, req.actor)
    return {"status": "deactivated", "reason": req.reason, "actor": req.actor}

@app.get("/admin/status", response_model=SystemStatus)
async def get_system_status():
    return SystemStatus(
        trading_enabled=settings.trading_enabled,
        kill_switch_active=risk_engine.kill_switch_active,
        ibkr_connected=ibkr_client.connected,
        circuit_breaker_open=ibkr_client.circuit_breaker.is_open,
        daily_pnl=risk_engine.daily_pnl,
        daily_trade_count=risk_engine.daily_trade_count,
        positions=risk_engine.positions
    )

@app.post("/admin/reset-limits")
async def reset_daily_limits():
    """Reset daily counters (for testing only)"""
    risk_engine.daily_pnl = 0.0
    risk_engine.daily_trade_count = 0
    return {"status": "reset", "daily_pnl": 0, "daily_trade_count": 0}

# ============================================================================
# HEALTH ENDPOINTS (M1)
# ============================================================================

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc)}

@app.get("/ready")
async def readiness_check():
    checks = {
        "webhook_endpoint": "operational",
        "configuration": "loaded",
        "hmac_secret": "configured" if settings.webhook_secret else "missing",
        "ibkr_connection": "connected" if ibkr_client.connected else "disconnected",
        "risk_engine": "enabled" if settings.enable_risk_engine else "disabled"
    }
    
    ready = all(v not in ["missing", "disconnected"] for k, v in checks.items() if k != "risk_engine")
    
    return {"ready": ready, "checks": checks}

@app.get("/")
async def root():
    return {
        "service": "TV-IBKR-v3",
        "version": "2.0.0",
        "milestones": ["M1: Webhook Security", "M2: Risk Engine & Execution"],
        "features": {
            "m1": ["HMAC auth", "Replay protection", "Schema validation"],
            "m2": ["Risk engine", "IBKR integration", "Telegram alerts", "Kill switch"]
        },
        "status": {
            "trading_enabled": settings.trading_enabled,
            "dry_run": settings.dry_run,
            "ibkr_connected": ibkr_client.connected
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)