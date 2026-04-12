"""SQLite database for trade logging, equity tracking, and candle storage."""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from models import AccountState, Candle, Order, Position, Side, OrderStatus, PositionStatus

logger = logging.getLogger(__name__)


class Database:
    """SQLite database manager for the trading bot."""

    def __init__(self, db_path: str = "db/trades.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS orders (
                id TEXT PRIMARY KEY,
                signal_id TEXT,
                instrument TEXT NOT NULL,
                side TEXT NOT NULL,
                order_type TEXT NOT NULL,
                units REAL NOT NULL,
                price REAL,
                stop_loss REAL,
                take_profit REAL,
                status TEXT NOT NULL DEFAULT 'PENDING',
                broker_order_id TEXT,
                strategy TEXT,
                fill_price REAL,
                fill_time TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS positions (
                id TEXT PRIMARY KEY,
                instrument TEXT NOT NULL,
                side TEXT NOT NULL,
                units REAL NOT NULL,
                entry_price REAL NOT NULL,
                stop_loss REAL,
                take_profit REAL,
                exit_price REAL,
                realized_pnl REAL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'OPEN',
                strategy TEXT,
                broker_trade_id TEXT,
                opened_at TEXT NOT NULL DEFAULT (datetime('now')),
                closed_at TEXT,
                close_reason TEXT
            );

            CREATE TABLE IF NOT EXISTS equity_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                balance REAL NOT NULL,
                equity REAL NOT NULL,
                unrealized_pnl REAL DEFAULT 0,
                open_positions INTEGER DEFAULT 0,
                peak_equity REAL DEFAULT 0,
                daily_pnl REAL DEFAULT 0,
                drawdown_pct REAL DEFAULT 0,
                timestamp TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS candles (
                instrument TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER DEFAULT 0,
                PRIMARY KEY (instrument, timeframe, timestamp)
            );

            CREATE TABLE IF NOT EXISTS daily_pnl (
                date TEXT PRIMARY KEY,
                starting_equity REAL NOT NULL,
                ending_equity REAL,
                realized_pnl REAL DEFAULT 0,
                trades_opened INTEGER DEFAULT 0,
                trades_closed INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
            CREATE INDEX IF NOT EXISTS idx_orders_strategy ON orders(strategy);
            CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
            CREATE INDEX IF NOT EXISTS idx_positions_strategy ON positions(strategy);
            CREATE INDEX IF NOT EXISTS idx_equity_timestamp ON equity_snapshots(timestamp);
            CREATE INDEX IF NOT EXISTS idx_candles_lookup ON candles(instrument, timeframe, timestamp);
        """)
        self.conn.commit()

    # --- Orders ---

    def save_order(self, order: Order):
        self.conn.execute("""
            INSERT OR REPLACE INTO orders
            (id, signal_id, instrument, side, order_type, units, price,
             stop_loss, take_profit, status, broker_order_id, strategy,
             fill_price, fill_time, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            order.id, order.signal_id, order.instrument, order.side.value,
            order.order_type.value, order.units, order.price,
            order.stop_loss, order.take_profit, order.status.value,
            order.broker_order_id, order.strategy,
            order.fill_price,
            order.fill_time.isoformat() if order.fill_time else None,
            order.created_at.isoformat(),
        ))
        self.conn.commit()

    def get_open_orders(self) -> list[dict]:
        cursor = self.conn.execute(
            "SELECT * FROM orders WHERE status IN ('PENDING', 'SUBMITTED')"
        )
        return [dict(row) for row in cursor.fetchall()]

    # --- Positions ---

    def save_position(self, pos: Position):
        self.conn.execute("""
            INSERT OR REPLACE INTO positions
            (id, instrument, side, units, entry_price, stop_loss, take_profit,
             exit_price, realized_pnl, status, strategy, broker_trade_id,
             opened_at, closed_at, close_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pos.id, pos.instrument, pos.side.value, pos.units,
            pos.entry_price, pos.stop_loss, pos.take_profit,
            pos.current_price if pos.status == PositionStatus.CLOSED else None,
            pos.realized_pnl, pos.status.value, pos.strategy,
            pos.broker_trade_id, pos.opened_at.isoformat(),
            pos.closed_at.isoformat() if pos.closed_at else None,
            pos.close_reason,
        ))
        self.conn.commit()

    def get_open_positions(self) -> list[dict]:
        cursor = self.conn.execute("SELECT * FROM positions WHERE status = 'OPEN'")
        return [dict(row) for row in cursor.fetchall()]

    def get_closed_positions(self, limit: int = 100) -> list[dict]:
        cursor = self.conn.execute(
            "SELECT * FROM positions WHERE status = 'CLOSED' ORDER BY closed_at DESC LIMIT ?",
            (limit,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_positions_by_strategy(self, strategy: str) -> list[dict]:
        cursor = self.conn.execute(
            "SELECT * FROM positions WHERE strategy = ? ORDER BY opened_at DESC",
            (strategy,)
        )
        return [dict(row) for row in cursor.fetchall()]

    # --- Equity ---

    def save_equity_snapshot(self, state: AccountState):
        self.conn.execute("""
            INSERT INTO equity_snapshots
            (balance, equity, unrealized_pnl, open_positions, peak_equity,
             daily_pnl, drawdown_pct, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            state.balance, state.equity, state.unrealized_pnl,
            state.open_position_count, state.peak_equity,
            state.daily_pnl, state.drawdown_pct,
            state.timestamp.isoformat(),
        ))
        self.conn.commit()

    def get_equity_history(self, hours: int = 24) -> list[dict]:
        cursor = self.conn.execute("""
            SELECT * FROM equity_snapshots
            WHERE timestamp > datetime('now', ?)
            ORDER BY timestamp ASC
        """, (f"-{hours} hours",))
        return [dict(row) for row in cursor.fetchall()]

    # --- Daily P&L ---

    def update_daily_pnl(self, date: str, **kwargs):
        """Upsert daily P&L record."""
        existing = self.conn.execute(
            "SELECT * FROM daily_pnl WHERE date = ?", (date,)
        ).fetchone()

        if existing:
            sets = ", ".join(f"{k} = ?" for k in kwargs)
            vals = list(kwargs.values()) + [date]
            self.conn.execute(f"UPDATE daily_pnl SET {sets} WHERE date = ?", vals)
        else:
            cols = ["date"] + list(kwargs.keys())
            placeholders = ", ".join(["?"] * len(cols))
            vals = [date] + list(kwargs.values())
            self.conn.execute(
                f"INSERT INTO daily_pnl ({', '.join(cols)}) VALUES ({placeholders})", vals
            )
        self.conn.commit()

    def get_daily_pnl(self, days: int = 30) -> list[dict]:
        cursor = self.conn.execute("""
            SELECT * FROM daily_pnl
            WHERE date > date('now', ?)
            ORDER BY date DESC
        """, (f"-{days} days",))
        return [dict(row) for row in cursor.fetchall()]

    # --- Candles ---

    def save_candles(self, candles: list[Candle]):
        if not candles:
            return
        self.conn.executemany("""
            INSERT OR REPLACE INTO candles
            (instrument, timeframe, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            (c.instrument, c.timeframe, c.timestamp.isoformat(),
             c.open, c.high, c.low, c.close, c.volume)
            for c in candles
        ])
        self.conn.commit()

    def get_candles(
        self, instrument: str, timeframe: str, count: int = 500
    ) -> list[dict]:
        cursor = self.conn.execute("""
            SELECT * FROM candles
            WHERE instrument = ? AND timeframe = ?
            ORDER BY timestamp DESC LIMIT ?
        """, (instrument, timeframe, count))
        rows = [dict(row) for row in cursor.fetchall()]
        rows.reverse()  # chronological order
        return rows

    # --- Stats ---

    def get_strategy_stats(self, strategy: str) -> dict:
        """Get performance stats for a strategy."""
        cursor = self.conn.execute("""
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) as losses,
                SUM(CASE WHEN realized_pnl = 0 THEN 1 ELSE 0 END) as breakeven,
                COALESCE(SUM(realized_pnl), 0) as total_pnl,
                COALESCE(AVG(realized_pnl), 0) as avg_pnl,
                COALESCE(AVG(CASE WHEN realized_pnl > 0 THEN realized_pnl END), 0) as avg_win,
                COALESCE(AVG(CASE WHEN realized_pnl < 0 THEN realized_pnl END), 0) as avg_loss,
                COALESCE(MAX(realized_pnl), 0) as best_trade,
                COALESCE(MIN(realized_pnl), 0) as worst_trade
            FROM positions
            WHERE strategy = ? AND status = 'CLOSED'
        """, (strategy,))
        row = dict(cursor.fetchone())
        total = row["total_trades"]
        row["win_rate"] = (row["wins"] / total * 100) if total > 0 else 0
        if row["avg_loss"] != 0:
            row["profit_factor"] = abs(row["avg_win"] / row["avg_loss"]) if row["avg_loss"] else 0
        else:
            row["profit_factor"] = float("inf") if row["avg_win"] > 0 else 0
        return row

    def get_overall_stats(self) -> dict:
        """Get overall performance stats."""
        cursor = self.conn.execute("""
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) as losses,
                COALESCE(SUM(realized_pnl), 0) as total_pnl,
                COALESCE(AVG(realized_pnl), 0) as avg_pnl
            FROM positions WHERE status = 'CLOSED'
        """)
        return dict(cursor.fetchone())

    def close(self):
        self.conn.close()
