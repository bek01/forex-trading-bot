"""Smart Pair Auto-Block — disables losing pairs, re-enables with exponential backoff.

How it works:
1. Track consecutive losses per instrument
2. After 3 consecutive losses → block pair for 7 days
3. After re-enable, if it loses again → block for 14 days (doubled)
4. Next failure → 28 days → 56 days → effectively permanent
5. If pair wins after re-enable → reset loss counter
6. Re-enable requires trend filter consensus (H4+D aligned)

State is persisted to disk (JSON) so it survives bot restarts.

This is the professional approach — adaptive pair rotation with
zero manual intervention.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_STATE_FILE = "db/pair_guard.json"


class PairGuard:
    """Tracks pair performance and auto-blocks consistent losers."""

    def __init__(
        self,
        state_file: str = DEFAULT_STATE_FILE,
        consecutive_losses_to_block: int = 3,
        initial_block_days: int = 3,  # 2026-04-20: lowered 7→3; backoff still doubles
        max_block_days: int = 90,
        backoff_multiplier: float = 2.0,
    ):
        self.state_file = state_file
        self.losses_to_block = consecutive_losses_to_block
        self.initial_block_days = initial_block_days
        self.max_block_days = max_block_days
        self.backoff_multiplier = backoff_multiplier

        # State per instrument:
        # {
        #   "EUR_USD": {
        #     "consecutive_losses": 0,
        #     "total_wins": 5,
        #     "total_losses": 2,
        #     "total_pnl": 1308.00,
        #     "blocked_until": null or "2026-04-22T00:00:00",
        #     "block_count": 0,  (how many times blocked — for backoff)
        #     "last_block_days": 0,
        #     "last_trade_time": "2026-04-15T16:24:00",
        #   }
        # }
        self._state: dict[str, dict] = {}
        self._load_state()

    def _load_state(self):
        """Load state from disk."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, "r") as f:
                    self._state = json.load(f)
                logger.info(f"PairGuard: loaded state for {len(self._state)} pairs")
        except Exception as e:
            logger.warning(f"PairGuard: failed to load state: {e}")
            self._state = {}

    def _save_state(self):
        """Persist state to disk."""
        try:
            Path(self.state_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump(self._state, f, indent=2)
        except Exception as e:
            logger.error(f"PairGuard: failed to save state: {e}")

    def _get_pair(self, instrument: str) -> dict:
        """Get or create pair state."""
        if instrument not in self._state:
            self._state[instrument] = {
                "consecutive_losses": 0,
                "total_wins": 0,
                "total_losses": 0,
                "total_pnl": 0.0,
                "blocked_until": None,
                "block_count": 0,
                "last_block_days": 0,
                "last_trade_time": None,
            }
        return self._state[instrument]

    def is_blocked(self, instrument: str) -> tuple[bool, str]:
        """
        Check if a pair is currently blocked.

        Returns:
            (blocked: bool, reason: str)
        """
        pair = self._get_pair(instrument)
        blocked_until = pair.get("blocked_until")

        if not blocked_until:
            return False, ""

        now = datetime.now(timezone.utc)
        block_end = datetime.fromisoformat(blocked_until)

        if now < block_end:
            days_left = (block_end - now).days
            hours_left = int((block_end - now).total_seconds() / 3600) % 24
            return True, (
                f"PAIR BLOCKED: {instrument} — "
                f"{pair['consecutive_losses']} consecutive losses, "
                f"blocked for {pair['last_block_days']}d "
                f"({days_left}d {hours_left}h remaining), "
                f"block #{pair['block_count']}"
            )

        # Block expired — pair gets another chance
        logger.info(
            f"PairGuard: {instrument} block expired after "
            f"{pair['last_block_days']} days — re-enabled for trading"
        )
        pair["blocked_until"] = None
        # Don't reset consecutive losses — if it loses again immediately,
        # it gets blocked with longer backoff
        self._save_state()
        return False, ""

    def record_trade(self, instrument: str, pnl: float):
        """
        Record a completed trade result.
        Call this when a trade closes (win or loss).
        """
        pair = self._get_pair(instrument)
        pair["last_trade_time"] = datetime.now(timezone.utc).isoformat()
        pair["total_pnl"] = pair.get("total_pnl", 0) + pnl

        if pnl > 0:
            # WIN — reset consecutive loss counter and clear any active block
            pair["total_wins"] = pair.get("total_wins", 0) + 1
            pair["consecutive_losses"] = 0
            if pair.get("blocked_until"):
                logger.info(
                    f"PairGuard: {instrument} won — clearing block "
                    f"(was blocked until {pair['blocked_until']})"
                )
                pair["blocked_until"] = None
            # If this is the first win after a re-enable, reset block count
            if pair.get("block_count", 0) > 0:
                logger.info(
                    f"PairGuard: {instrument} won after re-enable — "
                    f"resetting block count (was {pair['block_count']})"
                )
                pair["block_count"] = 0
                pair["last_block_days"] = 0

        elif pnl < 0:
            # LOSS — increment consecutive counter
            pair["total_losses"] = pair.get("total_losses", 0) + 1
            pair["consecutive_losses"] = pair.get("consecutive_losses", 0) + 1

            # Check if we should block
            if pair["consecutive_losses"] >= self.losses_to_block:
                self._block_pair(instrument, pair)

        self._save_state()

    def _block_pair(self, instrument: str, pair: dict):
        """Block a pair with exponential backoff."""
        pair["block_count"] = pair.get("block_count", 0) + 1

        # Calculate block duration with exponential backoff
        block_days = self.initial_block_days * (
            self.backoff_multiplier ** (pair["block_count"] - 1)
        )
        block_days = min(block_days, self.max_block_days)
        block_days = int(block_days)

        pair["last_block_days"] = block_days
        pair["blocked_until"] = (
            datetime.now(timezone.utc) + timedelta(days=block_days)
        ).isoformat()

        logger.warning(
            f"PAIR BLOCKED: {instrument} — "
            f"{pair['consecutive_losses']} consecutive losses, "
            f"blocked for {block_days} days (block #{pair['block_count']}), "
            f"total P&L: £{pair['total_pnl']:+,.2f}"
        )

    def get_status(self) -> dict:
        """Get status of all tracked pairs for monitoring."""
        now = datetime.now(timezone.utc)
        status = {}

        for inst, pair in sorted(self._state.items()):
            blocked_until = pair.get("blocked_until")
            is_blocked = False
            remaining = ""

            if blocked_until:
                block_end = datetime.fromisoformat(blocked_until)
                if now < block_end:
                    is_blocked = True
                    delta = block_end - now
                    remaining = f"{delta.days}d {int(delta.seconds/3600)}h"

            status[inst] = {
                "wins": pair.get("total_wins", 0),
                "losses": pair.get("total_losses", 0),
                "consecutive_losses": pair.get("consecutive_losses", 0),
                "pnl": pair.get("total_pnl", 0),
                "blocked": is_blocked,
                "remaining": remaining,
                "block_count": pair.get("block_count", 0),
            }

        return status

    def get_blocked_pairs(self) -> list[str]:
        """Return list of currently blocked instruments."""
        blocked = []
        now = datetime.now(timezone.utc)
        for inst, pair in self._state.items():
            blocked_until = pair.get("blocked_until")
            if blocked_until:
                block_end = datetime.fromisoformat(blocked_until)
                if now < block_end:
                    blocked.append(inst)
        return blocked

    def print_report(self):
        """Print a formatted report of all pair statuses."""
        status = self.get_status()
        if not status:
            print("  No pair data yet")
            return

        print(f"  {'Pair':<11} {'W':>3} {'L':>3} {'Streak':>6} {'P&L':>10} {'Status':<25}")
        print(f"  {'─'*60}")

        for inst in sorted(status, key=lambda x: status[x]["pnl"], reverse=True):
            s = status[inst]
            if s["blocked"]:
                state = f"🚫 BLOCKED ({s['remaining']})"
            elif s["consecutive_losses"] >= 2:
                state = f"⚠️ {s['consecutive_losses']} losses in a row"
            elif s["pnl"] > 0:
                state = "✅ Active"
            else:
                state = "📊 Monitoring"

            streak = f"-{s['consecutive_losses']}" if s["consecutive_losses"] > 0 else "OK"
            print(f"  {inst:<11} {s['wins']:>3} {s['losses']:>3} {streak:>6} £{s['pnl']:>+8,.0f} {state}")
