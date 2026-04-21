"""Profit Manager — partial profit scaling + trailing stops.

Replaces manual closes with automated 4-stage profit taking:

  STAGE 1: +10 pips → Close 25% of position + SL to breakeven
  STAGE 2: +20 pips → Close 25% more + start trailing SL
  STAGE 3: +30 pips → Close 25% more + tighten trail
  STAGE 4: Let final 25% run with tight trailing stop

This captures profit incrementally instead of all-or-nothing TP.
Our data shows TPs only hit 39% of the time — partial scaling
captures profit on the other 61% of trades.

Also manages:
- Breakeven SL moves
- Trailing stop progression
- Minimum position size checks (OANDA requires >= 1 unit)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from execution.broker import OandaBroker

logger = logging.getLogger(__name__)


class ProfitManager:
    """Manages partial profit taking + trailing stops on open trades."""

    def __init__(
        self,
        broker: OandaBroker,
        # Partial close stages (in pips of profit)
        stage1_pips: float = 10.0,   # close 25%, lock +N pips (not breakeven)
        stage2_pips: float = 20.0,   # close 25% more, wider trailing
        stage3_pips: float = 30.0,   # close 25% more, tighten trail
        # Partial close percentages (must sum to < 1.0, remainder runs)
        stage1_close_pct: float = 0.25,
        stage2_close_pct: float = 0.25,
        stage3_close_pct: float = 0.25,
        # Stage 1 locked profit — prevents breakeven give-back after +10 pips
        stage1_lock_pips: float = 3.0,
        # Trailing stop settings — separate per stage
        trail_stage1_pips: float = 5.0,   # 2026-04-17 tightened 10→5
        trail_distance_pips: float = 10.0,  # used at stage 2
        trail_tight_pips: float = 7.0,   # tighter trail after stage 3
        min_trail_step_pips: float = 3.0,
    ):
        self.broker = broker

        # Stage thresholds
        self.stage1_pips = stage1_pips
        self.stage2_pips = stage2_pips
        self.stage3_pips = stage3_pips
        self.stage1_pct = stage1_close_pct
        self.stage2_pct = stage2_close_pct
        self.stage3_pct = stage3_close_pct

        # Stage 1 profit lock
        self.stage1_lock = stage1_lock_pips

        # Trail settings
        self.trail_stage1 = trail_stage1_pips
        self.trail_distance = trail_distance_pips
        self.trail_tight = trail_tight_pips
        self.min_trail_step = min_trail_step_pips

        # State tracking per trade
        self._trade_state: dict[str, dict] = {}
        # Each trade_id -> {
        #   "initial_units": int,
        #   "stage": 0-4,
        #   "last_sl": float,
        # }
        self._last_check = 0.0
        self._price_cache: dict[str, tuple] = {}  # instrument -> (bid, ask, timestamp)
        self._price_cache_ttl = 5.0  # seconds

    def check_and_update(self, check_interval_sec: float = 10.0):
        """Check all open trades. Call from main loop."""
        now = time.time()
        if now - self._last_check < check_interval_sec:
            return
        self._last_check = now

        try:
            self._update_all_trades()
        except Exception as e:
            logger.error(f"Profit manager check failed: {e}")

    def _update_all_trades(self):
        """Fetch open trades and process each one."""
        try:
            resp = self.broker._get(
                f"/v3/accounts/{self.broker.account_id}/openTrades"
            )
        except Exception as e:
            logger.debug(f"Failed to fetch open trades: {e}")
            return

        trades = resp.get("trades", [])
        active_ids = set()

        for trade in trades:
            try:
                trade_id = trade["id"]
                active_ids.add(trade_id)
                self._process_trade(trade)
            except Exception as e:
                logger.debug(f"Profit manager error trade {trade.get('id')}: {e}")

        # Clean up closed trades
        closed = [tid for tid in self._trade_state if tid not in active_ids]
        for tid in closed:
            del self._trade_state[tid]

    def _get_price(self, instrument: str) -> tuple[float, float]:
        """Get bid/ask with short cache to avoid API spam."""
        now = time.time()
        cached = self._price_cache.get(instrument)
        if cached and (now - cached[2]) < self._price_cache_ttl:
            return cached[0], cached[1]

        tick = self.broker.get_price(instrument)
        self._price_cache[instrument] = (tick.bid, tick.ask, now)
        return tick.bid, tick.ask

    def _process_trade(self, trade: dict):
        """Process a single open trade for partial profit + trailing."""
        trade_id = trade["id"]
        instrument = trade["instrument"]
        current_units = int(trade["currentUnits"])
        is_buy = current_units > 0
        entry_price = float(trade["price"])
        abs_units = abs(current_units)

        # Get or create state
        if trade_id not in self._trade_state:
            self._trade_state[trade_id] = {
                "initial_units": abs_units,
                "stage": 0,
                "last_sl": 0.0,
            }

        state = self._trade_state[trade_id]

        # Get current SL
        sl_order = trade.get("stopLossOrder")
        if not sl_order:
            return
        current_sl = float(sl_order["price"])

        # Pip size
        pip_size = 0.01 if "JPY" in instrument else 0.0001

        # Get market price
        try:
            bid, ask = self._get_price(instrument)
        except Exception:
            return

        if is_buy:
            market_price = bid  # exit price for longs
            profit_pips = (market_price - entry_price) / pip_size
        else:
            market_price = ask  # exit price for shorts
            profit_pips = (entry_price - market_price) / pip_size

        # Skip if not in profit
        if profit_pips < self.stage1_pips and state["stage"] == 0:
            return

        initial_units = state["initial_units"]

        # === STAGE 1: +10 pips → close 25%, lock +N pips (not breakeven) ===
        # 2026-04-17: changed from breakeven → +3 pip lock. EUR_GBP gave back
        # all gains after hitting +10 pips because SL at entry = too loose.
        if profit_pips >= self.stage1_pips and state["stage"] < 1:
            lock_price = (
                entry_price + (self.stage1_lock * pip_size) if is_buy
                else entry_price - (self.stage1_lock * pip_size)
            )
            close_units = int(initial_units * self.stage1_pct)
            if close_units >= 1 and abs_units > close_units:
                if self._partial_close(trade_id, close_units, instrument, is_buy):
                    logger.info(
                        f"PARTIAL 1/4: {instrument} trade {trade_id} — "
                        f"closed {close_units} units at +{profit_pips:.0f} pips"
                    )
                self._update_sl(trade_id, lock_price, instrument)
                logger.info(
                    f"LOCK +{self.stage1_lock:.0f}: {instrument} trade {trade_id} — "
                    f"SL → {lock_price:.5f}"
                )
            elif close_units < 1:
                # Position too small to split — just lock profit on the whole lot
                self._update_sl(trade_id, lock_price, instrument)
                logger.info(
                    f"LOCK +{self.stage1_lock:.0f} (no partial — units too small): "
                    f"{instrument} trade {trade_id}"
                )
            state["stage"] = 1

        # === STAGE 2: +20 pips → close 25% more, start trailing ===
        if profit_pips >= self.stage2_pips and state["stage"] < 2:
            # Recalculate remaining units
            remaining = abs_units - int(initial_units * self.stage1_pct)
            close_units = int(initial_units * self.stage2_pct)
            if close_units >= 1 and abs_units > close_units:
                if self._partial_close(trade_id, close_units, instrument, is_buy):
                    logger.info(
                        f"PARTIAL 2/4: {instrument} trade {trade_id} — "
                        f"closed {close_units} units at +{profit_pips:.0f} pips"
                    )
            state["stage"] = 2

        # === STAGE 3: +30 pips → close 25% more, tighten trail ===
        if profit_pips >= self.stage3_pips and state["stage"] < 3:
            close_units = int(initial_units * self.stage3_pct)
            if close_units >= 1 and abs_units > close_units:
                if self._partial_close(trade_id, close_units, instrument, is_buy):
                    logger.info(
                        f"PARTIAL 3/4: {instrument} trade {trade_id} — "
                        f"closed {close_units} units at +{profit_pips:.0f} pips"
                    )
            state["stage"] = 3

        # === TRAILING STOP (active after stage 1) ===
        if state["stage"] >= 1 and profit_pips >= self.stage1_pips:
            # Stage-specific trail distance:
            #   Stage 1: 5 pips (tight — protect +3 pip lock)
            #   Stage 2: 10 pips (wider — let trend breathe)
            #   Stage 3+: 7 pips (tight final)
            if state["stage"] >= 3:
                trail = self.trail_tight
            elif state["stage"] >= 2:
                trail = self.trail_distance
            else:
                trail = self.trail_stage1

            if is_buy:
                new_sl = market_price - (trail * pip_size)
                if new_sl <= current_sl:
                    return  # only move up
            else:
                new_sl = market_price + (trail * pip_size)
                if new_sl >= current_sl:
                    return  # only move down

            # Minimum step check
            sl_move = abs(new_sl - current_sl) / pip_size
            if sl_move < self.min_trail_step:
                return

            new_sl = round(new_sl, 3 if "JPY" in instrument else 5)

            if self._update_sl(trade_id, new_sl, instrument):
                locked = abs(new_sl - entry_price) / pip_size
                logger.info(
                    f"TRAILING: {instrument} trade {trade_id} — "
                    f"SL → {new_sl} (locked: +{locked:.0f} pips, "
                    f"current: +{profit_pips:.0f} pips, "
                    f"stage: {state['stage']}/4, trail: {trail} pips)"
                )

    def _partial_close(self, trade_id: str, units: int, instrument: str, is_buy: bool) -> bool:
        """Close a portion of a trade."""
        try:
            # OANDA close uses positive units always — the side is implicit from the trade
            self.broker._put(
                f"/v3/accounts/{self.broker.account_id}/trades/{trade_id}/close",
                json={"units": str(units)},
            )
            return True
        except Exception as e:
            logger.error(f"Partial close failed trade {trade_id}: {e}")
            return False

    def _update_sl(self, trade_id: str, new_sl: float, instrument: str) -> bool:
        """Update stop loss on a trade."""
        try:
            decimals = 3 if "JPY" in instrument else 5
            self.broker._put(
                f"/v3/accounts/{self.broker.account_id}/trades/{trade_id}/orders",
                json={
                    "stopLoss": {
                        "price": f"{new_sl:.{decimals}f}",
                        "timeInForce": "GTC",
                    }
                },
            )
            return True
        except Exception as e:
            logger.error(f"SL update failed trade {trade_id}: {e}")
            return False

    def get_status(self) -> dict:
        """Current state for monitoring."""
        stages = {0: 0, 1: 0, 2: 0, 3: 0}
        for state in self._trade_state.values():
            s = state["stage"]
            stages[s] = stages.get(s, 0) + 1
        return {
            "tracked_trades": len(self._trade_state),
            "stage_0_waiting": stages[0],
            "stage_1_breakeven": stages[1],
            "stage_2_trailing": stages[2],
            "stage_3_tight_trail": stages[3],
        }
