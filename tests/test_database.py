"""Tests for database operations."""

import os
import pytest
from datetime import datetime, timezone

from db.database import Database
from models import (
    AccountState, Candle, Order, OrderStatus, OrderType,
    Position, PositionStatus, Side,
)


@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "test.db")
    database = Database(db_path)
    yield database
    database.close()


class TestDatabase:
    def test_create_tables(self, db):
        """Tables should be created on init."""
        cursor = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        assert "orders" in tables
        assert "positions" in tables
        assert "equity_snapshots" in tables
        assert "candles" in tables
        assert "daily_pnl" in tables

    def test_save_and_get_order(self, db):
        order = Order(
            id="test1", instrument="EUR_USD", side=Side.BUY,
            order_type=OrderType.MARKET, units=1000,
            stop_loss=1.09800, take_profit=1.10400,
            status=OrderStatus.FILLED, strategy="test",
        )
        db.save_order(order)
        # No open orders (it's filled)
        assert len(db.get_open_orders()) == 0

    def test_save_and_get_position(self, db):
        pos = Position(
            id="pos1", instrument="EUR_USD", side=Side.BUY,
            units=1000, entry_price=1.10000,
            stop_loss=1.09800, take_profit=1.10400,
            status=PositionStatus.OPEN, strategy="test",
        )
        db.save_position(pos)
        open_pos = db.get_open_positions()
        assert len(open_pos) == 1
        assert open_pos[0]["instrument"] == "EUR_USD"

    def test_save_equity_snapshot(self, db):
        state = AccountState(
            balance=10000, equity=10050, unrealized_pnl=50,
            peak_equity=10100, daily_pnl=50,
        )
        db.save_equity_snapshot(state)
        history = db.get_equity_history(hours=1)
        assert len(history) == 1
        assert history[0]["balance"] == 10000

    def test_strategy_stats(self, db):
        # Add some closed positions
        for i, pnl in enumerate([10, -5, 20, 15, -8]):
            pos = Position(
                id=f"p{i}", instrument="EUR_USD", side=Side.BUY,
                units=1000, entry_price=1.10000,
                realized_pnl=pnl,
                status=PositionStatus.CLOSED, strategy="test",
            )
            db.save_position(pos)

        stats = db.get_strategy_stats("test")
        assert stats["total_trades"] == 5
        assert stats["wins"] == 3
        assert stats["losses"] == 2
        assert stats["win_rate"] == 60.0
        assert stats["total_pnl"] == 32
