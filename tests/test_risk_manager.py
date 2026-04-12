"""Tests for the risk management engine.

These are the most important tests — risk bugs lose real money.
"""

import pytest
from datetime import datetime, timezone

from config import RiskConfig
from event_bus import bus, Event
from models import (
    AccountState, Position, PositionStatus, Signal, Side,
    SignalStrength, Tick,
)
from risk.risk_manager import RiskManager, RiskDecision


@pytest.fixture
def risk_config():
    return RiskConfig(
        max_risk_per_trade_pct=1.0,
        max_risk_per_trade_usd=100.0,
        max_open_positions=3,
        daily_loss_limit_pct=2.0,
        max_drawdown_halt_pct=10.0,
        require_stop_loss=True,
        min_risk_reward_ratio=1.5,
        max_stop_loss_pips=50.0,
        max_spread_multiplier=2.0,
    )


@pytest.fixture
def risk_manager(risk_config):
    rm = RiskManager(risk_config)
    rm.account = AccountState(
        balance=10000, equity=10000, peak_equity=10000,
    )
    rm.daily_start_equity = 10000
    return rm


@pytest.fixture
def sample_signal():
    return Signal(
        strategy="test",
        instrument="EUR_USD",
        side=Side.BUY,
        strength=SignalStrength.MODERATE,
        entry_price=1.10000,
        stop_loss=1.09800,    # 20 pip SL
        take_profit=1.10400,  # 40 pip TP → 2.0 R:R
    )


@pytest.fixture
def sample_tick():
    return Tick(
        instrument="EUR_USD",
        bid=1.09990,
        ask=1.10010,
        spread=1.2,  # pips
        timestamp=datetime.now(timezone.utc),
    )


class TestRiskManager:
    """Test all risk rules."""

    def test_approved_signal(self, risk_manager, sample_signal, sample_tick):
        """A good signal with proper SL/TP should be approved."""
        decision = risk_manager.evaluate_signal(sample_signal, sample_tick, [])
        assert decision.approved
        assert decision.units > 0

    def test_no_stop_loss_rejected(self, risk_manager, sample_tick):
        """Signal without stop loss MUST be rejected."""
        signal = Signal(
            strategy="test", instrument="EUR_USD", side=Side.BUY,
            entry_price=1.10000, stop_loss=0, take_profit=1.10400,
        )
        decision = risk_manager.evaluate_signal(signal, sample_tick, [])
        assert not decision.approved
        assert "stop loss" in decision.reason.lower()

    def test_low_rr_rejected(self, risk_manager, sample_tick):
        """Signal with R:R below minimum should be rejected."""
        signal = Signal(
            strategy="test", instrument="EUR_USD", side=Side.BUY,
            entry_price=1.10000, stop_loss=1.09800,
            take_profit=1.10100,  # only 10 pip TP vs 20 pip SL = 0.5 R:R
        )
        decision = risk_manager.evaluate_signal(signal, sample_tick, [])
        assert not decision.approved
        assert "R:R" in decision.reason

    def test_max_positions_reached(self, risk_manager, sample_signal, sample_tick):
        """Should reject when max open positions reached."""
        existing = [
            Position(instrument="EUR_USD", side=Side.BUY, units=100),
            Position(instrument="GBP_USD", side=Side.BUY, units=100),
            Position(instrument="USD_JPY", side=Side.SELL, units=100),
        ]
        decision = risk_manager.evaluate_signal(sample_signal, sample_tick, existing)
        assert not decision.approved
        assert "max positions" in decision.reason.lower()

    def test_spread_too_wide(self, risk_manager, sample_signal):
        """Should reject when spread exceeds threshold."""
        wide_tick = Tick(
            instrument="EUR_USD", bid=1.09900, ask=1.10100,
            spread=5.0,  # 5 pips spread, avg is 1.2, 2x max = 2.4
            timestamp=datetime.now(timezone.utc),
        )
        decision = risk_manager.evaluate_signal(sample_signal, wide_tick, [])
        assert not decision.approved
        assert "spread" in decision.reason.lower()

    def test_daily_loss_limit(self, risk_manager, sample_signal, sample_tick):
        """Should halt when daily loss exceeds limit."""
        risk_manager.daily_pnl = -250  # -2.5% of 10000
        decision = risk_manager.evaluate_signal(sample_signal, sample_tick, [])
        assert not decision.approved
        assert risk_manager.trading_halted

    def test_max_drawdown_kills(self, risk_manager, sample_signal, sample_tick):
        """Should trigger kill switch at max drawdown."""
        risk_manager.account.equity = 8900  # 11% drawdown from 10000 peak
        risk_manager.account.peak_equity = 10000
        decision = risk_manager.evaluate_signal(sample_signal, sample_tick, [])
        assert not decision.approved
        assert risk_manager.trading_halted

    def test_position_sizing_1_percent(self, risk_manager, sample_signal, sample_tick):
        """Position size should risk ~1% of equity."""
        decision = risk_manager.evaluate_signal(sample_signal, sample_tick, [])
        assert decision.approved
        # 1% of 10000 = $100 risk
        # SL distance = 0.002 (20 pips)
        # units = 100 / 0.002 = 50,000
        expected_units = 50000
        assert abs(decision.units - expected_units) < 1000  # within 1000 units

    def test_kill_switch_blocks_everything(self, risk_manager, sample_signal, sample_tick):
        """After kill switch, ALL signals must be rejected."""
        risk_manager.manual_kill("test kill")
        decision = risk_manager.evaluate_signal(sample_signal, sample_tick, [])
        assert not decision.approved
        assert "kill switch" in decision.reason.lower()

    def test_stop_loss_too_wide_rejected(self, risk_manager, sample_tick):
        """Signal with SL > max pips should be rejected."""
        signal = Signal(
            strategy="test", instrument="EUR_USD", side=Side.BUY,
            entry_price=1.10000,
            stop_loss=1.09300,    # 70 pips, max is 50
            take_profit=1.11500,  # keeps R:R good
        )
        decision = risk_manager.evaluate_signal(signal, sample_tick, [])
        assert not decision.approved
        assert "stop loss" in decision.reason.lower() and "pips" in decision.reason.lower()

    def test_resume_after_halt(self, risk_manager, sample_signal, sample_tick):
        """Should be able to resume after non-kill halt."""
        risk_manager.daily_pnl = -250
        risk_manager.evaluate_signal(sample_signal, sample_tick, [])
        assert risk_manager.trading_halted

        # Reset for new day
        risk_manager.reset_daily()
        risk_manager.daily_pnl = 0
        risk_manager.account.equity = 10000
        decision = risk_manager.evaluate_signal(sample_signal, sample_tick, [])
        assert decision.approved


class TestPositionSizing:
    """Test position sizing calculations."""

    def test_size_scales_with_equity(self, risk_config):
        """Position size should scale proportionally with equity."""
        rm_small = RiskManager(risk_config)
        rm_small.account = AccountState(balance=5000, equity=5000, peak_equity=5000)

        rm_large = RiskManager(risk_config)
        rm_large.account = AccountState(balance=20000, equity=20000, peak_equity=20000)

        signal = Signal(
            strategy="test", instrument="EUR_USD", side=Side.BUY,
            entry_price=1.10000, stop_loss=1.09800, take_profit=1.10400,
        )
        tick = Tick(instrument="EUR_USD", bid=1.09990, ask=1.10010,
                    spread=1.2, timestamp=datetime.now(timezone.utc))

        d_small = rm_small.evaluate_signal(signal, tick, [])
        d_large = rm_large.evaluate_signal(signal, tick, [])

        # 20k equity should risk $200 (1%), 5k should risk $50 (1%)
        # Both capped at $100 hard cap, so 20k = $100 risk, 5k = $50 risk → 2x
        assert d_large.units >= d_small.units * 1.5
        assert d_large.units <= d_small.units * 2.5

    def test_hard_cap_limits_size(self, risk_config):
        """Hard cap should limit position size even with large equity."""
        risk_config.max_risk_per_trade_usd = 50  # tight cap
        rm = RiskManager(risk_config)
        rm.account = AccountState(balance=100000, equity=100000, peak_equity=100000)

        signal = Signal(
            strategy="test", instrument="EUR_USD", side=Side.BUY,
            entry_price=1.10000, stop_loss=1.09800, take_profit=1.10400,
        )
        tick = Tick(instrument="EUR_USD", bid=1.09990, ask=1.10010,
                    spread=1.2, timestamp=datetime.now(timezone.utc))

        decision = rm.evaluate_signal(signal, tick, [])
        # $50 risk / 0.002 SL = 25000 units max
        assert decision.units <= 25001
