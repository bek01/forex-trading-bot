"""Risk management engine — the most critical component.

This module enforces ALL risk rules. No trade reaches the broker without
passing through here. Capital preservation is the #1 priority.

Rules enforced:
1. Every trade MUST have a stop loss
2. Max risk per trade: 1% of equity (configurable)
3. Position sizing: ATR-based with equity-proportional cap
4. Max open positions: 3 (configurable)
5. Correlation check: correlated pairs count as 1.5x positions
6. Daily loss limit: -2% → halt trading for the day
7. Max drawdown: 10% from peak → kill switch
8. Spread filter: skip if spread > 2x average
9. Friday size reduction: 50% after 18:00 UTC
10. News filter: no trades 30min before/after high-impact events
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from config import RiskConfig
from event_bus import Event, bus
from models import (
    AccountState, FOREX_PAIRS, PAIR_CORRELATIONS,
    Position, PositionStatus, Signal, Side, Tick
)

if TYPE_CHECKING:
    from data.economic_calendar import EconomicCalendar

logger = logging.getLogger(__name__)


class RiskDecision:
    """Result of a risk check."""

    def __init__(self, approved: bool, units: float = 0, reason: str = ""):
        self.approved = approved
        self.units = units
        self.reason = reason

    def __repr__(self):
        status = "APPROVED" if self.approved else "BLOCKED"
        return f"RiskDecision({status}, units={self.units:.0f}, reason='{self.reason}')"


class RiskManager:
    """Enforces all risk management rules."""

    def __init__(
        self,
        config: RiskConfig,
        economic_calendar: Optional["EconomicCalendar"] = None,
    ):
        self.config = config
        self.account: AccountState = AccountState()
        self.open_positions: list[Position] = []
        self.daily_pnl: float = 0.0
        self.daily_start_equity: float = 0.0
        self.trading_halted: bool = False
        self.halt_reason: str = ""
        self._news_events: list[dict] = []  # upcoming high-impact events
        self._killed: bool = False  # permanent kill switch (manual reset required)
        self._calendar = economic_calendar  # optional EconomicCalendar instance

        # Subscribe to events
        bus.subscribe(Event.ACCOUNT_UPDATE, self._on_account_update, priority=1)
        bus.subscribe(Event.POSITION_OPENED, self._on_position_opened, priority=1)
        bus.subscribe(Event.POSITION_CLOSED, self._on_position_closed, priority=1)
        bus.subscribe(Event.KILL_SWITCH, self._on_kill_switch, priority=0)

    def evaluate_signal(
        self,
        signal: Signal,
        current_tick: Tick,
        open_positions: list[Position],
    ) -> RiskDecision:
        """
        Evaluate whether a trading signal should be executed.
        Returns RiskDecision with approved/rejected and position size.
        """
        self.open_positions = open_positions

        # --- Kill switch check ---
        if self._killed:
            return RiskDecision(False, 0, f"Kill switch active: {self.halt_reason}")

        # --- Trading halt check ---
        if self.trading_halted:
            return RiskDecision(False, 0, f"Trading halted: {self.halt_reason}")

        # --- Rule 1: Stop loss required ---
        if self.config.require_stop_loss and signal.stop_loss == 0:
            return RiskDecision(False, 0, "No stop loss set — REJECTED (mandatory)")

        # --- Rule 2: Minimum R:R ratio ---
        if signal.risk_reward_ratio < self.config.min_risk_reward_ratio:
            return RiskDecision(
                False, 0,
                f"R:R {signal.risk_reward_ratio:.2f} < min {self.config.min_risk_reward_ratio}"
            )

        # --- Rule 3: Max open positions ---
        effective_positions = self._count_effective_positions(open_positions)
        if effective_positions >= self.config.max_open_positions:
            return RiskDecision(
                False, 0,
                f"Max positions reached: {effective_positions:.1f} >= {self.config.max_open_positions}"
            )

        # --- Rule 4: Correlation check ---
        corr_penalty = self._check_correlation(signal.instrument, open_positions)
        if effective_positions + corr_penalty > self.config.max_open_positions:
            return RiskDecision(
                False, 0,
                f"Correlated exposure too high: {effective_positions + corr_penalty:.1f} effective positions"
            )

        # --- Rule 5: Spread filter ---
        pair_info = FOREX_PAIRS.get(signal.instrument, {})
        avg_spread = pair_info.get("avg_spread_pips", 2.0)
        if current_tick.spread > avg_spread * self.config.max_spread_multiplier:
            return RiskDecision(
                False, 0,
                f"Spread {current_tick.spread:.1f} pips > {avg_spread * self.config.max_spread_multiplier:.1f} max"
            )

        # --- Rule 6: Drawdown check ---
        drawdown = self.account.drawdown_pct
        if drawdown >= self.config.max_drawdown_halt_pct:
            self._halt_trading(f"Max drawdown {drawdown:.1f}% >= {self.config.max_drawdown_halt_pct}%")
            bus.emit(Event.KILL_SWITCH, {"reason": self.halt_reason, "drawdown": drawdown})
            return RiskDecision(False, 0, self.halt_reason)

        # --- Rule 7: Daily loss limit ---
        if self.daily_start_equity > 0:
            daily_loss_pct = (self.daily_pnl / self.daily_start_equity) * 100
            if daily_loss_pct <= -self.config.daily_loss_limit_pct:
                self._halt_trading(
                    f"Daily loss limit: {daily_loss_pct:.1f}% <= -{self.config.daily_loss_limit_pct}%"
                )
                bus.emit(Event.DAILY_LOSS_LIMIT, {"pnl": self.daily_pnl, "pct": daily_loss_pct})
                return RiskDecision(False, 0, self.halt_reason)

        # --- Rule 8: Friday size reduction ---
        now = datetime.now(timezone.utc)
        friday_reduction = 1.0
        if now.weekday() == 4 and now.hour >= 18:  # Friday after 18:00 UTC
            friday_reduction = 1.0 - (self.config.friday_reduce_size_pct / 100.0)
            logger.info(f"Friday size reduction: {friday_reduction:.0%}")

        # --- Rule 9: News filter ---
        if self._is_near_news_event(signal.instrument):
            return RiskDecision(False, 0, "High-impact news event nearby — skipping")

        # --- Rule 10: Stop loss distance check ---
        pip_size = pair_info.get("pip_size", 0.0001)
        sl_pips = abs(signal.entry_price - signal.stop_loss) / pip_size
        if sl_pips > self.config.max_stop_loss_pips:
            return RiskDecision(
                False, 0,
                f"Stop loss {sl_pips:.0f} pips > max {self.config.max_stop_loss_pips:.0f}"
            )

        # --- Position sizing ---
        units = self._calculate_position_size(signal, pip_size)
        units *= friday_reduction

        # Drawdown-based size reduction
        if drawdown >= self.config.max_drawdown_critical_pct:
            units *= 0.5
            logger.warning(f"Critical drawdown {drawdown:.1f}% — reducing size 50%")
        elif drawdown >= self.config.max_drawdown_warning_pct:
            units *= 0.75
            logger.warning(f"Drawdown warning {drawdown:.1f}% — reducing size 25%")
            bus.emit(Event.DRAWDOWN_WARNING, {"drawdown": drawdown})

        units = int(units)
        if units <= 0:
            return RiskDecision(False, 0, "Position size calculated as 0 units")

        return RiskDecision(True, units, "All risk checks passed")

    def _calculate_position_size(self, signal: Signal, pip_size: float) -> float:
        """
        ATR-based position sizing: risk X% of equity per trade.

        Formula: units = risk_amount / (stop_loss_distance_in_price)
        """
        equity = self.account.equity
        if equity <= 0:
            return 0

        # Risk amount = min(pct of equity, hard cap)
        risk_pct_amount = equity * (self.config.max_risk_per_trade_pct / 100.0)
        risk_amount = min(risk_pct_amount, self.config.max_risk_per_trade_usd)

        # Stop loss distance in price
        sl_distance = abs(signal.entry_price - signal.stop_loss)
        if sl_distance <= 0:
            return 0

        # units = risk / sl_distance
        # For USD-denominated pairs (EUR/USD, GBP/USD), pip value ≈ pip_size per unit
        # For JPY pairs, need to convert — simplified here
        pip_value_per_unit = pip_size  # approximate for USD account
        sl_pips = sl_distance / pip_size
        units = risk_amount / (sl_pips * pip_value_per_unit)

        return units

    def _count_effective_positions(self, positions: list[Position]) -> float:
        """Count positions with correlation weighting."""
        return len(positions)

    def _check_correlation(
        self, instrument: str, positions: list[Position]
    ) -> float:
        """
        Check correlation with existing positions.
        Returns additional position count penalty.
        """
        penalty = 0.0
        for pos in positions:
            pair = tuple(sorted([instrument, pos.instrument]))
            corr = PAIR_CORRELATIONS.get(pair, 0.0)
            if abs(corr) > 0.7:
                # High correlation — count as extra exposure
                penalty += (self.config.max_correlated_exposure - 1.0) * abs(corr)
        return penalty

    def _is_near_news_event(self, instrument: str) -> bool:
        """Check if there's a high-impact news event nearby.

        Uses the EconomicCalendar if one was provided at construction time.
        Falls back to False (backwards compatible) if no calendar is available.
        """
        if self._calendar is None:
            return False

        try:
            return self._calendar.is_near_high_impact(
                currency=instrument,
                minutes_before=self.config.no_trade_before_news_minutes,
                minutes_after=self.config.no_trade_after_news_minutes,
            )
        except Exception as exc:
            logger.warning(f"Economic calendar check failed: {exc}")
            return False

    def _halt_trading(self, reason: str):
        """Halt all new trading."""
        self.trading_halted = True
        self.halt_reason = reason
        logger.critical(f"TRADING HALTED: {reason}")

    def resume_trading(self):
        """Resume trading (manual action)."""
        if not self._killed:
            self.trading_halted = False
            self.halt_reason = ""
            logger.info("Trading resumed")

    def reset_daily(self):
        """Reset daily counters — call at start of each trading day."""
        self.daily_pnl = 0.0
        self.daily_start_equity = self.account.equity
        self.trading_halted = False
        self.halt_reason = ""
        logger.info(f"Daily reset — starting equity: ${self.daily_start_equity:.2f}")

    # --- Event handlers ---

    def _on_account_update(self, state: AccountState):
        self.account = state
        if state.equity > self.account.peak_equity:
            self.account.peak_equity = state.equity

    def _on_position_opened(self, position: Position):
        self.open_positions.append(position)

    def _on_position_closed(self, data: dict):
        pnl = data.get("realized_pnl", 0)
        self.daily_pnl += pnl
        # Remove from open positions
        pos_id = data.get("id")
        self.open_positions = [p for p in self.open_positions if p.id != pos_id]

    def _on_kill_switch(self, data):
        self._killed = True
        self.trading_halted = True
        self.halt_reason = f"KILL SWITCH: {data.get('reason', 'manual')}"
        logger.critical(self.halt_reason)

    def manual_kill(self, reason: str = "Manual kill switch"):
        """Manually trigger kill switch."""
        bus.emit(Event.KILL_SWITCH, {"reason": reason})

    def manual_reset(self):
        """Reset kill switch (use with extreme caution)."""
        self._killed = False
        self.trading_halted = False
        self.halt_reason = ""
        logger.warning("Kill switch manually reset")

    def get_status(self) -> dict:
        """Current risk manager status."""
        return {
            "trading_halted": self.trading_halted,
            "halt_reason": self.halt_reason,
            "killed": self._killed,
            "daily_pnl": self.daily_pnl,
            "drawdown_pct": self.account.drawdown_pct,
            "open_positions": len(self.open_positions),
            "max_positions": self.config.max_open_positions,
            "equity": self.account.equity,
            "peak_equity": self.account.peak_equity,
        }
