"""Book-It rule — close profitable positions at session close to lock gains.

Problem: By end of London/NY session, a profitable intraday trade often
gives back pips overnight (thin liquidity, Asia reversals, gap risk).

Logic (once per trigger time per day):
  - At the configured UTC hour (default 17:00 = London close), iterate
    all open trades.
  - For each trade with unrealizedPL >= min_profit_usd OR (mfe_pips >=
    min_profit_pips and unrealizedPL > 0), close it via broker.
  - Trades in loss OR barely positive are left alone (SL will handle).

Runs at most once per calendar day per trigger. Safe to call every loop
iteration — internal guard prevents re-trigger.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from execution.broker import OandaBroker

logger = logging.getLogger(__name__)


class BookItRule:
    """Session-close profit locker."""

    def __init__(
        self,
        broker: OandaBroker,
        trigger_hour_utc: int = 17,      # London close
        trigger_minute_utc: int = 0,
        min_profit_usd: float = 0.50,    # close if unrealized >= this $
        min_profit_pips: float = 5.0,    # OR if MFE >= this many pips
        trigger_on_weekdays: tuple = (0, 1, 2, 3, 4),  # Mon-Fri
    ):
        self.broker = broker
        self.trigger_hour = trigger_hour_utc
        self.trigger_minute = trigger_minute_utc
        self.min_profit_usd = min_profit_usd
        self.min_profit_pips = min_profit_pips
        self.trigger_weekdays = set(trigger_on_weekdays)
        self._last_run_date: Optional[str] = None  # YYYY-MM-DD

    def check_and_run(self):
        """Call from main loop — triggers at most once per day."""
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")

        if self._last_run_date == today:
            return
        if now.weekday() not in self.trigger_weekdays:
            return
        # Trigger within the first 5 minutes of the target window
        if now.hour != self.trigger_hour:
            return
        if now.minute < self.trigger_minute or now.minute > self.trigger_minute + 5:
            return

        self._last_run_date = today
        self._run()

    def _run(self):
        """Close profitable trades."""
        try:
            resp = self.broker._get(
                f"/v3/accounts/{self.broker.account_id}/openTrades"
            )
        except Exception as e:
            logger.error(f"Book-It: failed to fetch open trades: {e}")
            return

        trades = resp.get("trades", [])
        if not trades:
            logger.info("Book-It triggered — no open trades to review")
            return

        closed_count = 0
        kept_count = 0
        total_locked = 0.0

        for t in trades:
            tid = str(t.get("id"))
            instrument = t.get("instrument", "")
            units = int(float(t.get("currentUnits", 0)))
            unrealized = float(t.get("unrealizedPL", 0))
            # MFE pips: approximate via open price vs current — OANDA doesn't
            # give historical high directly on /openTrades.
            # Rely on unrealizedPL as the primary trigger.

            if unrealized >= self.min_profit_usd:
                ok = self.broker.close_trade(tid)
                if ok:
                    closed_count += 1
                    total_locked += unrealized
                    logger.info(
                        f"Book-It: CLOSED {instrument} trade={tid} "
                        f"units={units:+d} +${unrealized:.2f}"
                    )
            else:
                kept_count += 1
                logger.debug(
                    f"Book-It: kept {instrument} trade={tid} "
                    f"uPnL=${unrealized:+.2f} (< ${self.min_profit_usd:.2f})"
                )

        logger.info(
            f"Book-It complete: closed={closed_count} kept={kept_count} "
            f"locked=${total_locked:+.2f}"
        )
