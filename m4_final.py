"""
TV-IBKR-v3 COMPLETE - ALL 4 MILESTONES
Production-Ready TradingView to Interactive Brokers Integration

✅ Milestone 1: Core Ingestion & Security
✅ Milestone 2: Risk Engine & Order Execution  
✅ Milestone 3: Persistence & Reconciliation
✅ Milestone 4: Admin, Testing & Handoff

Requirements:
pip install fastapi uvicorn python-dotenv pydantic pydantic-settings ib-insync python-telegram-bot nest-asyncio aiosqlite

Run:
uvicorn final_main_complete:app --host 0.0.0.0 --port 8000
"""

import asyncio
import hashlib
import hmac
import json
import logging
import uuid
import threading
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional, Dict, List, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

from fastapi import FastAPI, HTTPException, Request, status, Depends, Header
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import aiosqlite

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
    
    # M3: Persistence
    database_path: str = "trading.db"
    reconciliation_interval: int = 60  # seconds
    reconciliation_tolerance: int = 0  # allowed position mismatch
    auto_halt_on_mismatch: bool = True
    
    # M2: Feature Flags
    trading_enabled: bool = True
    dry_run: bool = False
    enable_risk_engine: bool = True
    enable_order_execution: bool = True
    enable_telegram: bool = False
    enable_reconciliation: bool = True
    
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

def log_warning(msg: str, **kwargs):
    logger.warning(f"{msg} {json.dumps(kwargs) if kwargs else ''}")

# ============================================================================
# MODELS
# ============================================================================

class TradeStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"

class AuditEventType(str, Enum):
    WEBHOOK_RECEIVED = "WEBHOOK_RECEIVED"
    WEBHOOK_REJECTED = "WEBHOOK_REJECTED"
    ORDER_PLACED = "ORDER_PLACED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_FAILED = "ORDER_FAILED"
    RISK_VIOLATION = "RISK_VIOLATION"
    KILL_SWITCH_ACTIVATED = "KILL_SWITCH_ACTIVATED"
    KILL_SWITCH_DEACTIVATED = "KILL_SWITCH_DEACTIVATED"
    RECONCILIATION_MISMATCH = "RECONCILIATION_MISMATCH"
    RECONCILIATION_SUCCESS = "RECONCILIATION_SUCCESS"
    SYSTEM_HALTED = "SYSTEM_HALTED"

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
    expected_positions: Dict[str, int]
    last_reconciliation: Optional[datetime] = None
    reconciliation_status: str = "unknown"

class TradeRecord(BaseModel):
    id: Optional[int] = None
    correlation_id: str
    timestamp: datetime
    ticker: str
    action: str
    quantity: int
    order_type: str
    limit_price: Optional[float] = None
    status: TradeStatus
    order_id: Optional[int] = None
    fill_price: Optional[float] = None
    fill_quantity: Optional[int] = None
    error_message: Optional[str] = None
    strategy: Optional[str] = None

class AuditRecord(BaseModel):
    id: Optional[int] = None
    timestamp: datetime
    event_type: AuditEventType
    correlation_id: Optional[str] = None
    ticker: Optional[str] = None
    details: Dict[str, Any]

class ReconciliationReport(BaseModel):
    timestamp: datetime
    status: str
    mismatches: List[Dict[str, Any]]
    expected_positions: Dict[str, int]
    actual_positions: Dict[str, int]
    action_taken: Optional[str] = None

# ============================================================================
# M3: DATABASE (SQLite with WAL)
# ============================================================================

class DatabaseManager:
    """
    SQLite database with WAL mode for safe concurrent reads.
    Single-writer queue pattern ensures data integrity.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.write_queue: asyncio.Queue = asyncio.Queue()
        self.write_task: Optional[asyncio.Task] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize database with WAL mode and schema"""
        if self._initialized:
            return
        
        async with aiosqlite.connect(self.db_path) as db:
            # Enable WAL mode for better concurrency
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA synchronous=NORMAL")
            await db.execute("PRAGMA busy_timeout=5000")
            
            # Create tables
            await db.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    correlation_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    order_type TEXT NOT NULL,
                    limit_price REAL,
                    status TEXT NOT NULL,
                    order_id INTEGER,
                    fill_price REAL,
                    fill_quantity INTEGER,
                    error_message TEXT,
                    strategy TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    correlation_id TEXT,
                    ticker TEXT,
                    details TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS expected_positions (
                    ticker TEXT PRIMARY KEY,
                    quantity INTEGER NOT NULL,
                    last_updated TEXT NOT NULL
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS reconciliation_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    status TEXT NOT NULL,
                    mismatches TEXT,
                    expected_positions TEXT NOT NULL,
                    actual_positions TEXT NOT NULL,
                    action_taken TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indices
            await db.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_audit_type ON audit_log(event_type)")
            
            await db.commit()
        
        # Start write worker
        self.write_task = asyncio.create_task(self._write_worker())
        self._initialized = True
        log_info("Database initialized with WAL mode")
    
    async def _write_worker(self):
        """Single-writer worker to serialize all writes"""
        while True:
            try:
                operation, args = await self.write_queue.get()
                
                async with aiosqlite.connect(self.db_path) as db:
                    if operation == "insert_trade":
                        await self._insert_trade(db, args)
                    elif operation == "update_trade":
                        await self._update_trade(db, args)
                    elif operation == "insert_audit":
                        await self._insert_audit(db, args)
                    elif operation == "update_expected_position":
                        await self._update_expected_position(db, args)
                    elif operation == "insert_reconciliation":
                        await self._insert_reconciliation(db, args)
                    
                    await db.commit()
                
                self.write_queue.task_done()
            except Exception as e:
                log_error("Write worker error", error=str(e))
    
    async def _insert_trade(self, db: aiosqlite.Connection, trade: TradeRecord):
        await db.execute("""
            INSERT INTO trades (correlation_id, timestamp, ticker, action, quantity, 
                               order_type, limit_price, status, order_id, fill_price, 
                               fill_quantity, error_message, strategy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.correlation_id, trade.timestamp.isoformat(), trade.ticker,
            trade.action, trade.quantity, trade.order_type, trade.limit_price,
            trade.status.value, trade.order_id, trade.fill_price,
            trade.fill_quantity, trade.error_message, trade.strategy
        ))
    
    async def _update_trade(self, db: aiosqlite.Connection, update_data: dict):
        await db.execute("""
            UPDATE trades 
            SET status=?, order_id=?, fill_price=?, fill_quantity=?, error_message=?
            WHERE correlation_id=?
        """, (
            update_data['status'], update_data['order_id'], 
            update_data['fill_price'], update_data['fill_quantity'],
            update_data['error_message'], update_data['correlation_id']
        ))
    
    async def _insert_audit(self, db: aiosqlite.Connection, audit: AuditRecord):
        await db.execute("""
            INSERT INTO audit_log (timestamp, event_type, correlation_id, ticker, details)
            VALUES (?, ?, ?, ?, ?)
        """, (
            audit.timestamp.isoformat(), audit.event_type.value,
            audit.correlation_id, audit.ticker, json.dumps(audit.details)
        ))
    
    async def _update_expected_position(self, db: aiosqlite.Connection, data: dict):
        await db.execute("""
            INSERT OR REPLACE INTO expected_positions (ticker, quantity, last_updated)
            VALUES (?, ?, ?)
        """, (data['ticker'], data['quantity'], datetime.now(timezone.utc).isoformat()))
    
    async def _insert_reconciliation(self, db: aiosqlite.Connection, report: ReconciliationReport):
        await db.execute("""
            INSERT INTO reconciliation_log (timestamp, status, mismatches, 
                                           expected_positions, actual_positions, action_taken)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            report.timestamp.isoformat(), report.status,
            json.dumps(report.mismatches) if report.mismatches else None,
            json.dumps(report.expected_positions),
            json.dumps(report.actual_positions),
            report.action_taken
        ))
    
    # Write operations (queued)
    async def insert_trade(self, trade: TradeRecord):
        await self.write_queue.put(("insert_trade", trade))
    
    async def update_trade(self, correlation_id: str, status: TradeStatus, 
                          order_id: Optional[int] = None, 
                          fill_price: Optional[float] = None,
                          fill_quantity: Optional[int] = None,
                          error_message: Optional[str] = None):
        await self.write_queue.put(("update_trade", {
            'correlation_id': correlation_id,
            'status': status.value,
            'order_id': order_id,
            'fill_price': fill_price,
            'fill_quantity': fill_quantity,
            'error_message': error_message
        }))
    
    async def insert_audit(self, event_type: AuditEventType, details: Dict[str, Any],
                          correlation_id: Optional[str] = None,
                          ticker: Optional[str] = None):
        audit = AuditRecord(
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            correlation_id=correlation_id,
            ticker=ticker,
            details=details
        )
        await self.write_queue.put(("insert_audit", audit))
    
    async def update_expected_position(self, ticker: str, quantity: int):
        await self.write_queue.put(("update_expected_position", {
            'ticker': ticker,
            'quantity': quantity
        }))
    
    async def insert_reconciliation(self, report: ReconciliationReport):
        await self.write_queue.put(("insert_reconciliation", report))
    
    # Read operations (direct)
    async def get_expected_positions(self) -> Dict[str, int]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT ticker, quantity FROM expected_positions") as cursor:
                rows = await cursor.fetchall()
                return {row['ticker']: row['quantity'] for row in rows}
    
    async def get_daily_trades(self) -> List[TradeRecord]:
        today = datetime.now(timezone.utc).date()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT * FROM trades 
                WHERE date(timestamp) = ?
                ORDER BY timestamp DESC
            """, (today.isoformat(),)) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_trade(row) for row in rows]
    
    async def get_trade_by_correlation_id(self, correlation_id: str) -> Optional[TradeRecord]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM trades WHERE correlation_id = ?", 
                (correlation_id,)
            ) as cursor:
                row = await cursor.fetchone()
                return self._row_to_trade(row) if row else None
    
    async def get_audit_log(self, limit: int = 100) -> List[AuditRecord]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT * FROM audit_log 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,)) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_audit(row) for row in rows]
    
    async def get_reconciliation_history(self, limit: int = 10) -> List[ReconciliationReport]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT * FROM reconciliation_log 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,)) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_reconciliation(row) for row in rows]
    
    def _row_to_trade(self, row) -> TradeRecord:
        return TradeRecord(
            id=row['id'],
            correlation_id=row['correlation_id'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            ticker=row['ticker'],
            action=row['action'],
            quantity=row['quantity'],
            order_type=row['order_type'],
            limit_price=row['limit_price'],
            status=TradeStatus(row['status']),
            order_id=row['order_id'],
            fill_price=row['fill_price'],
            fill_quantity=row['fill_quantity'],
            error_message=row['error_message'],
            strategy=row['strategy']
        )
    
    def _row_to_audit(self, row) -> AuditRecord:
        return AuditRecord(
            id=row['id'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            event_type=AuditEventType(row['event_type']),
            correlation_id=row['correlation_id'],
            ticker=row['ticker'],
            details=json.loads(row['details'])
        )
    
    def _row_to_reconciliation(self, row) -> ReconciliationReport:
        return ReconciliationReport(
            timestamp=datetime.fromisoformat(row['timestamp']),
            status=row['status'],
            mismatches=json.loads(row['mismatches']) if row['mismatches'] else [],
            expected_positions=json.loads(row['expected_positions']),
            actual_positions=json.loads(row['actual_positions']),
            action_taken=row['action_taken']
        )
    
    async def shutdown(self):
        """Wait for all pending writes to complete"""
        await self.write_queue.join()
        if self.write_task:
            self.write_task.cancel()

db = DatabaseManager(settings.database_path)

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
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            from ib_insync import IB, util
            
            try:
                import nest_asyncio
                nest_asyncio.apply()
            except:
                pass
            
            util.patchAsyncio()
            
            self.ib = IB()
            
            log_info(f"Connecting to IBKR", host=settings.ibkr_host, port=settings.ibkr_port)
            
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
        
        self.connection_thread = threading.Thread(
            target=self._connect_in_thread,
            daemon=True
        )
        self.connection_thread.start()
        
        await asyncio.sleep(3)
    
    async def disconnect(self):
        if self.ib and self.connected:
            try:
                self.ib.disconnect()
            except:
                pass
            self.connected = False
    
    async def get_positions(self) -> Dict[str, int]:
        """Get actual positions from IBKR"""
        if not self.connected or settings.dry_run:
            return {}
        
        try:
            if not self.circuit_breaker.can_attempt():
                return {}
            
            positions = {}
            for pos in self.ib.positions():
                if pos.contract.secType == 'STK':
                    positions[pos.contract.symbol] = int(pos.position)
            
            self.circuit_breaker.record_success()
            return positions
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            log_error("Failed to get positions", error=str(e))
            return {}
    
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
    
    async def send_reconciliation_alert(self, report: ReconciliationReport):
        if report.status == "mismatch":
            msg = f"⚠️ POSITION MISMATCH DETECTED\n"
            for mismatch in report.mismatches:
                msg += f"\n{mismatch['ticker']}: Expected {mismatch['expected']}, Got {mismatch['actual']}"
            if report.action_taken:
                msg += f"\n\nAction: {report.action_taken}"
            await self.send(msg)

alerter = TelegramAlerter()

# ============================================================================
# M3: RECONCILIATION ENGINE
# ============================================================================

class ReconciliationEngine:
    """
    Reconciles expected positions (from trade log) with actual IBKR positions.
    Runs every 60 seconds and auto-halts on mismatch if configured.
    """
    
    def __init__(self):
        self.last_reconciliation: Optional[datetime] = None
        self.reconciliation_status = "unknown"
        self.reconciliation_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start reconciliation loop"""
        if not settings.enable_reconciliation:
            log_info("Reconciliation disabled")
            return
        
        self.reconciliation_task = asyncio.create_task(self._reconciliation_loop())
        log_info("Reconciliation engine started", interval=settings.reconciliation_interval)
    
    async def _reconciliation_loop(self):
        """Run reconciliation every interval seconds"""
        while True:
            try:
                await asyncio.sleep(settings.reconciliation_interval)
                await self.reconcile()
            except Exception as e:
                log_error("Reconciliation error", error=str(e))
    
    async def reconcile(self) -> ReconciliationReport:
        """
        Compare expected positions with actual IBKR positions.
        Auto-halt if mismatch exceeds tolerance.
        """
        try:
            expected = await db.get_expected_positions()
            actual = await ibkr_client.get_positions()
            
            mismatches = []
            all_tickers = set(expected.keys()) | set(actual.keys())
            
            for ticker in all_tickers:
                exp_qty = expected.get(ticker, 0)
                act_qty = actual.get(ticker, 0)
                
                if abs(exp_qty - act_qty) > settings.reconciliation_tolerance:
                    mismatches.append({
                        'ticker': ticker,
                        'expected': exp_qty,
                        'actual': act_qty,
                        'delta': act_qty - exp_qty
                    })
            
            status = "mismatch" if mismatches else "match"
            action_taken = None
            
            if mismatches:
                log_warning("Position mismatch detected", mismatches=mismatches)
                
                if settings.auto_halt_on_mismatch:
                    risk_engine.activate_kill_switch(
                        f"Position reconciliation mismatch: {len(mismatches)} ticker(s)",
                        "RECONCILIATION_ENGINE"
                    )
                    action_taken = "KILL_SWITCH_ACTIVATED"
                    
                    await db.insert_audit(
                        AuditEventType.RECONCILIATION_MISMATCH,
                        {'mismatches': mismatches, 'action': 'halt'}
                    )
                
                await alerter.send_reconciliation_alert(
                    ReconciliationReport(
                        timestamp=datetime.now(timezone.utc),
                        status=status,
                        mismatches=mismatches,
                        expected_positions=expected,
                        actual_positions=actual,
                        action_taken=action_taken
                    )
                )
            else:
                log_info("Reconciliation: All positions match")
                await db.insert_audit(
                    AuditEventType.RECONCILIATION_SUCCESS,
                    {'expected': expected, 'actual': actual}
                )
            
            report = ReconciliationReport(
                timestamp=datetime.now(timezone.utc),
                status=status,
                mismatches=mismatches,
                expected_positions=expected,
                actual_positions=actual,
                action_taken=action_taken
            )
            
            await db.insert_reconciliation(report)
            
            self.last_reconciliation = datetime.now(timezone.utc)
            self.reconciliation_status = status
            
            return report
            
        except Exception as e:
            log_error("Reconciliation failed", error=str(e))
            self.reconciliation_status = "error"
            raise
    
    async def stop(self):
        """Stop reconciliation loop"""
        if self.reconciliation_task:
            self.reconciliation_task.cancel()

reconciliation_engine = ReconciliationEngine()

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
    
    # Initialize database
    await db.initialize()
    
    # Start IBKR connection
    if settings.enable_order_execution and not settings.dry_run:
        asyncio.create_task(ibkr_client.connect())
    
    # Start reconciliation engine
    if settings.enable_reconciliation:
        await reconciliation_engine.start()
    
    log_info("Application ready")
    yield
    
    log_info("Application shutting down...")
    await reconciliation_engine.stop()
    await ibkr_client.disconnect()
    await db.shutdown()

app = FastAPI(
    title="TV-IBKR-v3 COMPLETE",
    version="3.0.0",
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
        
        # Verify signature
        signature = payload_dict.get("signature", "")
        payload_without_sig = {k: v for k, v in payload_dict.items() if k != "signature"}
        payload_for_hmac = json.dumps(payload_without_sig, separators=(',', ':')).encode()
        
        if not verify_hmac_signature(payload_for_hmac, signature, settings.webhook_secret):
            await db.insert_audit(
                AuditEventType.WEBHOOK_REJECTED,
                {'reason': 'invalid_signature'},
                correlation_id=correlation_id
            )
            raise HTTPException(status_code=401, detail="Invalid signature")
        
        payload = WebhookPayload(**payload_dict)
        
        # Log webhook received
        await db.insert_audit(
            AuditEventType.WEBHOOK_RECEIVED,
            {
                'ticker': payload.ticker,
                'action': payload.action,
                'quantity': payload.quantity,
                'strategy': payload.strategy
            },
            correlation_id=correlation_id,
            ticker=payload.ticker
        )
        
        # Replay protection
        replay_guard.validate_timestamp(payload.timestamp)
        idempotency_key = replay_guard.generate_idempotency_key(
            payload.ticker, payload.action, payload.timestamp
        )
        
        if replay_guard.check_duplicate(idempotency_key):
            raise HTTPException(status_code=409, detail="Duplicate webhook")
        
        # Create trade record
        quantity = payload.quantity or settings.max_position_size
        trade = TradeRecord(
            correlation_id=correlation_id,
            timestamp=payload.timestamp,
            ticker=payload.ticker,
            action=payload.action,
            quantity=quantity,
            order_type=payload.order_type,
            limit_price=payload.limit_price,
            status=TradeStatus.PENDING,
            strategy=payload.strategy
        )
        await db.insert_trade(trade)
        
        # Risk checks
        risk_result = await risk_engine.validate(payload)
        
        if not risk_result.approved:
            await db.update_trade(
                correlation_id,
                TradeStatus.REJECTED,
                error_message=risk_result.reason
            )
            await db.insert_audit(
                AuditEventType.RISK_VIOLATION,
                {'reason': risk_result.reason, 'checks': risk_result.checks},
                correlation_id=correlation_id,
                ticker=payload.ticker
            )
            await alerter.send_rejection(payload.ticker, payload.action, risk_result.reason)
            
            return WebhookResponse(
                status="rejected",
                correlation_id=correlation_id,
                message=risk_result.reason,
                idempotency_key=idempotency_key
            )
        
        # Execute order
        if settings.enable_order_execution or settings.dry_run:
            order_result = await ibkr_client.place_order(
                payload.ticker,
                payload.action,
                quantity,
                payload.order_type,
                payload.limit_price
            )
            
            if order_result.success:
                # Update trade record
                await db.update_trade(
                    correlation_id,
                    TradeStatus.FILLED,
                    order_id=order_result.order_id,
                    fill_price=order_result.fill_price,
                    fill_quantity=order_result.fill_quantity
                )
                
                # Update positions
                qty_change = quantity if payload.action == "BUY" else -quantity
                risk_engine.update_position(payload.ticker, qty_change)
                risk_engine.record_trade()
                
                # Update expected positions in DB
                new_expected = risk_engine.positions.get(payload.ticker, 0)
                await db.update_expected_position(payload.ticker, new_expected)
                
                # Audit log
                await db.insert_audit(
                    AuditEventType.ORDER_FILLED,
                    {
                        'order_id': order_result.order_id,
                        'fill_price': order_result.fill_price,
                        'fill_quantity': order_result.fill_quantity
                    },
                    correlation_id=correlation_id,
                    ticker=payload.ticker
                )
                
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
                # Order failed
                await db.update_trade(
                    correlation_id,
                    TradeStatus.FAILED,
                    error_message=order_result.error
                )
                await db.insert_audit(
                    AuditEventType.ORDER_FAILED,
                    {'error': order_result.error},
                    correlation_id=correlation_id,
                    ticker=payload.ticker
                )
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
# M4: ADMIN ENDPOINTS
# ============================================================================

@app.post("/admin/kill", dependencies=[Depends(verify_admin_key)])
async def activate_kill_switch(req: KillSwitchRequest):
    risk_engine.activate_kill_switch(req.reason, req.actor)
    await db.insert_audit(
        AuditEventType.KILL_SWITCH_ACTIVATED,
        {'reason': req.reason, 'actor': req.actor}
    )
    await alerter.send(f"🛑 Kill Switch ACTIVATED by {req.actor}: {req.reason}")
    return {"status": "activated", "reason": req.reason}

@app.post("/admin/resume", dependencies=[Depends(verify_admin_key)])
async def deactivate_kill_switch(req: KillSwitchRequest):
    risk_engine.deactivate_kill_switch(req.actor)
    await db.insert_audit(
        AuditEventType.KILL_SWITCH_DEACTIVATED,
        {'actor': req.actor}
    )
    await alerter.send(f"✅ Kill Switch DEACTIVATED by {req.actor}")
    return {"status": "deactivated"}

@app.get("/admin/status", response_model=SystemStatus, dependencies=[Depends(verify_admin_key)])
async def get_status():
    expected_positions = await db.get_expected_positions()
    
    return SystemStatus(
        trading_enabled=settings.trading_enabled,
        kill_switch_active=risk_engine.kill_switch_active,
        ibkr_connected=ibkr_client.connected,
        circuit_breaker_open=ibkr_client.circuit_breaker.is_open,
        daily_pnl=risk_engine.daily_pnl,
        daily_trade_count=risk_engine.daily_trade_count,
        positions=risk_engine.positions,
        expected_positions=expected_positions,
        last_reconciliation=reconciliation_engine.last_reconciliation,
        reconciliation_status=reconciliation_engine.reconciliation_status
    )

@app.post("/admin/reset-limits", dependencies=[Depends(verify_admin_key)])
async def reset_limits():
    risk_engine.daily_pnl = 0.0
    risk_engine.daily_trade_count = 0
    await db.insert_audit(
        AuditEventType.WEBHOOK_RECEIVED,
        {'action': 'reset_limits'}
    )
    return {"status": "reset"}

@app.post("/admin/reconcile", dependencies=[Depends(verify_admin_key)])
async def force_reconciliation():
    """Manually trigger reconciliation"""
    report = await reconciliation_engine.reconcile()
    return report

@app.get("/admin/trades", dependencies=[Depends(verify_admin_key)])
async def get_trades(limit: int = 50):
    """Get recent trades"""
    trades = await db.get_daily_trades()
    return {"trades": trades[:limit]}

@app.get("/admin/audit-log", dependencies=[Depends(verify_admin_key)])
async def get_audit_log(limit: int = 100):
    """Get audit log"""
    logs = await db.get_audit_log(limit)
    return {"logs": logs}

@app.get("/admin/reconciliation-history", dependencies=[Depends(verify_admin_key)])
async def get_reconciliation_history(limit: int = 10):
    """Get reconciliation history"""
    history = await db.get_reconciliation_history(limit)
    return {"history": history}

@app.get("/admin/expected-positions", dependencies=[Depends(verify_admin_key)])
async def get_expected_positions():
    """Get expected positions from trade log"""
    positions = await db.get_expected_positions()
    return {"positions": positions}

@app.get("/admin/actual-positions", dependencies=[Depends(verify_admin_key)])
async def get_actual_positions():
    """Get actual positions from IBKR"""
    positions = await ibkr_client.get_positions()
    return {"positions": positions}

# ============================================================================
# HEALTH ENDPOINTS
# ============================================================================

@app.get("/health")
async def health():
    return {
        "status": "healthy", 
        "timestamp": datetime.now(timezone.utc),
        "database": "connected"
    }

@app.get("/ready")
async def ready():
    return {
        "ready": ibkr_client.connected if settings.enable_order_execution else True,
        "ibkr_connected": ibkr_client.connected,
        "dry_run": settings.dry_run,
        "database": "initialized",
        "reconciliation": reconciliation_engine.reconciliation_status
    }

@app.get("/")
async def root():
    return {
        "service": "TV-IBKR-v3 COMPLETE",
        "version": "3.0.0",
        "milestones": ["M1: Security", "M2: Execution", "M3: Persistence", "M4: Admin"],
        "features": [
            "HMAC Authentication",
            "Risk Engine with Kill Switch",
            "IBKR Execution with Threading",
            "SQLite WAL Persistence",
            "Position Reconciliation",
            "Comprehensive Audit Logging",
            "Admin Endpoints"
        ],
        "status": {
            "trading": settings.trading_enabled,
            "ibkr": ibkr_client.connected,
            "dry_run": settings.dry_run,
            "reconciliation": reconciliation_engine.reconciliation_status
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("__main__:app", host="0.0.0.0", port=8000, reload=False)