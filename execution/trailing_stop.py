"""Trailing Stop Manager — automatically protects profits on open trades.

This replaces the need for manual closes. It monitors all open trades
and progressively moves the stop loss as the trade moves in your favor.

How it works:
1. Trade opens with original SL (e.g., -15 pips)
2. Price moves +10 pips → SL moved to breakeven (entry price)
3. Price moves +20 pips → SL trails at +10 pips behind
4. Price hits TP → full target closed automatically
5. Price reverses → trailing SL catches it with locked profit

Settings (all in pips, configurable):
- breakeven_trigger: Move SL to breakeven after this many pips profit (default: 10)
- trail_trigger: Start trailing after this many pips profit (default: 20)
- trail_distance: How far behind price the SL trails (default: 12 pips)
- min_trail_step: Minimum pip movement before updating SL (default: 3 pips)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

from execution.broker import OandaBroker

logger = logging.getLogger(__name__)


class TrailingStopManager:
    """Monitors open trades and manages trailing stops via OANDA API."""

    def __init__(
        self,
        broker: OandaBroker,
        breakeven_trigger_pips: float = 10.0,
        trail_trigger_pips: float = 20.0,
        trail_distance_pips: float = 12.0,
        min_trail_step_pips: float = 3.0,
    ):
        self.broker = broker
        self.breakeven_trigger = breakeven_trigger_pips
        self.trail_trigger = trail_trigger_pips
        self.trail_distance = trail_distance_pips
        self.min_trail_step = min_trail_step_pips

        # Track which trades we've already moved to breakeven
        self._breakeven_set: set[str] = set()  # trade IDs at breakeven
        self._last_sl: dict[str, float] = {}   # trade_id → last SL price we set
        self._last_check = 0.0

    def check_and_update(self, check_interval_sec: float = 15.0):
        """
        Check all open trades and update trailing stops.
        Call this from the main loop. Throttled by check_interval_sec.
        """
        now = time.time()
        if now - self._last_check < check_interval_sec:
            return
        self._last_check = now

        try:
            self._update_all_trades()
        except Exception as e:
            logger.error(f"Trailing stop check failed: {e}")

    def _update_all_trades(self):
        """Fetch open trades from broker and update SLs."""
        resp = self.broker._get(
            f"/v3/accounts/{self.broker.account_id}/openTrades"
        )
        trades = resp.get("trades", [])

        for trade in trades:
            try:
                self._process_trade(trade)
            except Exception as e:
                logger.debug(f"Trail update error for trade {trade.get('id')}: {e}")

    def _process_trade(self, trade: dict):
        """Process a single open trade for trailing stop logic."""
        trade_id = trade["id"]
        instrument = trade["instrument"]
        units = int(trade["currentUnits"])
        is_buy = units > 0
        entry_price = float(trade["price"])
        current_price = float(trade.get("unrealizedPL", 0))  # not useful directly

        # Get current SL
        sl_order = trade.get("stopLossOrder")
        if not sl_order:
            return  # no SL set — shouldn't happen but skip

        current_sl = float(sl_order["price"])

        # Determine pip size
        pip_size = 0.01 if "JPY" in instrument else 0.0001

        # Get current market price
        # We use the trade's currentUnits and unrealizedPL to infer current price
        # But more reliable: use the pricing endpoint
        try:
            tick = self.broker.get_price(instrument)
        except Exception:
            return

        if is_buy:
            current_market = tick.bid  # exit price for longs
            profit_pips = (current_market - entry_price) / pip_size
        else:
            current_market = tick.ask  # exit price for shorts
            profit_pips = (entry_price - current_market) / pip_size

        # === STAGE 1: Move to breakeven ===
        if (profit_pips >= self.breakeven_trigger and
                trade_id not in self._breakeven_set):
            # Move SL to entry price (breakeven)
            new_sl = entry_price
            if is_buy and new_sl > current_sl:
                if self._update_sl(trade_id, new_sl, instrument):
                    self._breakeven_set.add(trade_id)
                    logger.info(
                        f"BREAKEVEN: {instrument} trade {trade_id} — "
                        f"SL moved to {new_sl:.5f} (+{profit_pips:.0f} pips profit)"
                    )
            elif not is_buy and new_sl < current_sl:
                if self._update_sl(trade_id, new_sl, instrument):
                    self._breakeven_set.add(trade_id)
                    logger.info(
                        f"BREAKEVEN: {instrument} trade {trade_id} — "
                        f"SL moved to {new_sl:.5f} (+{profit_pips:.0f} pips profit)"
                    )

        # === STAGE 2: Trail the stop ===
        if profit_pips >= self.trail_trigger:
            if is_buy:
                new_sl = current_market - (self.trail_distance * pip_size)
                # Only move SL up, never down
                if new_sl <= current_sl:
                    return
            else:
                new_sl = current_market + (self.trail_distance * pip_size)
                # Only move SL down, never up
                if new_sl >= current_sl:
                    return

            # Check minimum step
            sl_move_pips = abs(new_sl - current_sl) / pip_size
            if sl_move_pips < self.min_trail_step:
                return

            new_sl = round(new_sl, 5 if pip_size == 0.0001 else 3)

            if self._update_sl(trade_id, new_sl, instrument):
                locked_pips = abs(new_sl - entry_price) / pip_size
                logger.info(
                    f"TRAILING: {instrument} trade {trade_id} — "
                    f"SL → {new_sl:.5f} (profit locked: +{locked_pips:.0f} pips, "
                    f"current: +{profit_pips:.0f} pips)"
                )

    def _update_sl(self, trade_id: str, new_sl: float, instrument: str) -> bool:
        """Update the stop loss on a trade via OANDA API."""
        try:
            # Determine decimal places
            decimals = 3 if "JPY" in instrument else 5
            sl_str = f"{new_sl:.{decimals}f}"

            self.broker._put(
                f"/v3/accounts/{self.broker.account_id}/trades/{trade_id}/orders",
                json={
                    "stopLoss": {
                        "price": sl_str,
                        "timeInForce": "GTC",
                    }
                },
            )
            self._last_sl[trade_id] = new_sl
            return True
        except Exception as e:
            logger.error(f"Failed to update SL on trade {trade_id}: {e}")
            return False

    def on_trade_closed(self, trade_id: str):
        """Clean up state when a trade closes."""
        self._breakeven_set.discard(trade_id)
        self._last_sl.pop(trade_id, None)

    def get_status(self) -> dict:
        """Return current trailing stop state."""
        return {
            "trades_at_breakeven": len(self._breakeven_set),
            "trades_trailing": len(self._last_sl),
            "breakeven_ids": list(self._breakeven_set),
        }
