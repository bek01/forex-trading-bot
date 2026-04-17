"""Weekend Flatten rule — close ALL open trades before the weekend.

FX markets close ~22:00 UTC Friday and reopen ~22:00 UTC Sunday. During
that window:
  - Positions are unhedgeable (market is shut)
  - News can hit (geopolitical, central bank surprises, weekend data)
  - Sunday gaps can blow through stop losses

Book-It closes only profitable trades. This rule takes it one step
further for Fridays only: close everything, realized or not, and be
flat for the weekend.

Runs at most once per ISO week (Friday), at configured UTC time.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable, Optional

from execution.broker import OandaBroker

logger = logging.getLogger(__name__)


class WeekendFlattenRule:
    """Flatten all positions before FX weekend close."""

    def __init__(
        self,
        broker: OandaBroker,
        trigger_hour_utc: int = 20,          # 20:00 UTC Fri ≈ 1h before NY close
        trigger_minute_utc: int = 0,
        close_all: bool = True,              # True = close losers too
        telegram_notifier: Optional[Callable[[str], None]] = None,
    ):
        self.broker = broker
        self.trigger_hour = trigger_hour_utc
        self.trigger_minute = trigger_minute_utc
        self.close_all = close_all
        self.telegram = telegram_notifier
        self._last_run_week: Optional[str] = None  # ISO "YYYY-Www"

    def check_and_run(self):
        """Call from main loop; fires at most once per Friday."""
        now = datetime.now(timezone.utc)
        # Friday = 4
        if now.weekday() != 4:
            return
        if now.hour != self.trigger_hour:
            return
        if now.minute < self.trigger_minute or now.minute > self.trigger_minute + 5:
            return

        iso = now.isocalendar()
        week_key = f"{iso.year}-W{iso.week:02d}"
        if self._last_run_week == week_key:
            return

        self._last_run_week = week_key
        self._run(now)

    def _run(self, now: datetime):
        try:
            resp = self.broker._get(
                f"/v3/accounts/{self.broker.account_id}/openTrades"
            )
        except Exception as e:
            logger.error(f"Weekend-Flatten: failed to list open trades: {e}")
            return

        trades = resp.get("trades", [])
        if not trades:
            logger.info("Weekend-Flatten triggered — no open trades to close")
            if self.telegram:
                try:
                    self.telegram(
                        f"🏁 <b>WEEKEND FLATTEN</b>\n"
                        f"  {now.strftime('%Y-%m-%d %H:%M UTC')}\n"
                        f"  Already flat. No action needed."
                    )
                except Exception:
                    pass
            return

        closed_count = 0
        skipped_count = 0
        total_pnl = 0.0
        winners = 0
        losers = 0
        details: list[str] = []

        for t in trades:
            tid = str(t.get("id"))
            instrument = t.get("instrument", "")
            units = int(float(t.get("currentUnits", 0)))
            unrealized = float(t.get("unrealizedPL", 0))

            if not self.close_all and unrealized <= 0:
                skipped_count += 1
                continue

            ok = self.broker.close_trade(tid)
            if ok:
                closed_count += 1
                total_pnl += unrealized
                if unrealized > 0:
                    winners += 1
                elif unrealized < 0:
                    losers += 1
                sign = "+" if unrealized >= 0 else ""
                details.append(
                    f"    {instrument} {units:+d} {sign}${unrealized:.2f}"
                )
                logger.info(
                    f"Weekend-Flatten: CLOSED {instrument} trade={tid} "
                    f"units={units:+d} uPnL=${unrealized:+.2f}"
                )
            else:
                logger.warning(
                    f"Weekend-Flatten: failed to close {instrument} trade={tid}"
                )

        # Cancel any leftover pending orders (SL/TP stubs without parents)
        try:
            po = self.broker._get(
                f"/v3/accounts/{self.broker.account_id}/pendingOrders"
            )
            for o in po.get("orders", []):
                # Skip if still has a live parent trade
                oid = o.get("id")
                trade_id = o.get("tradeID")
                if trade_id:
                    # This is a SL/TP — OANDA auto-cancels when parent closes,
                    # but the close-trade response isn't instant, so leave it.
                    continue
                # Only cancel standalone limit/stop orders
                try:
                    self.broker._put(
                        f"/v3/accounts/{self.broker.account_id}/orders/{oid}/cancel"
                    )
                except Exception as e:
                    logger.debug(f"Weekend-Flatten: could not cancel order {oid}: {e}")
        except Exception as e:
            logger.debug(f"Weekend-Flatten: pending-order cleanup skipped: {e}")

        summary = (
            f"🏁 <b>WEEKEND FLATTEN</b>\n"
            f"  {now.strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"  Closed: {closed_count}  Skipped: {skipped_count}\n"
            f"  Winners: {winners}  Losers: {losers}\n"
            f"  Realized: ${total_pnl:+.2f}\n"
        )
        if details:
            summary += "  Trades:\n" + "\n".join(details)

        logger.info(
            f"Weekend-Flatten complete: closed={closed_count} "
            f"skipped={skipped_count} total=${total_pnl:+.2f}"
        )
        if self.telegram:
            try:
                self.telegram(summary)
            except Exception as e:
                logger.error(f"Weekend-Flatten telegram failed: {e}")
