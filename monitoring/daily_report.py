"""Daily P&L report — fires once per day to Telegram.

Pulls today's closed trades from the DB, aggregates stats, and sends
a formatted summary. Triggers at configured UTC hour (default 21:05
— five minutes after NY close).
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class DailyReport:
    """Emit a once-per-day P&L summary to Telegram."""

    def __init__(
        self,
        db_path: str,
        telegram_notifier,
        account_label: str = "DEMO",        # prefix for multi-account clarity
        trigger_hour_utc: int = 21,
        trigger_minute_utc: int = 5,
        skip_weekends: bool = True,
    ):
        self.db_path = db_path
        self.telegram = telegram_notifier
        self.label = account_label
        self.trigger_hour = trigger_hour_utc
        self.trigger_minute = trigger_minute_utc
        self.skip_weekends = skip_weekends
        self._last_run_date: Optional[str] = None

    def check_and_run(self, current_equity: float, current_balance: float):
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")

        if self._last_run_date == today:
            return
        if self.skip_weekends and now.weekday() >= 5:  # Sat=5, Sun=6
            return
        if now.hour != self.trigger_hour:
            return
        if now.minute < self.trigger_minute:
            return

        self._last_run_date = today
        try:
            self._fire(today, current_equity, current_balance)
        except Exception as e:
            logger.error(f"Daily report failed: {e}")

    def _fire(self, date: str, equity: float, balance: float):
        stats = self._compute_stats(date)
        self._send(date, stats, equity, balance)

    def _compute_stats(self, date: str) -> dict:
        """Query DB for today's closed trades."""
        start = f"{date}T00:00:00"
        end = f"{date}T23:59:59.999999"

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        rows = c.execute(
            """SELECT instrument, side, realized_pnl, close_reason, strategy,
                      closed_at, opened_at
                 FROM positions
                WHERE status='CLOSED' AND closed_at BETWEEN ? AND ?""",
            (start, end),
        ).fetchall()

        conn.close()

        if not rows:
            return {"count": 0}

        trades = [dict(r) for r in rows]
        pnls = [float(t["realized_pnl"] or 0) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        breakeven = [p for p in pnls if p == 0]

        # By close reason
        reason_counts = {}
        for t in trades:
            r = t.get("close_reason") or "unknown"
            reason_counts[r] = reason_counts.get(r, 0) + 1

        # Per instrument
        per_inst = {}
        for t in trades:
            ins = t["instrument"]
            p = float(t.get("realized_pnl") or 0)
            if ins not in per_inst:
                per_inst[ins] = {"cnt": 0, "pnl": 0.0}
            per_inst[ins]["cnt"] += 1
            per_inst[ins]["pnl"] += p

        top = sorted(per_inst.items(), key=lambda x: x[1]["pnl"], reverse=True)

        return {
            "count": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "breakeven": len(breakeven),
            "total_pnl": sum(pnls),
            "avg_win": sum(wins) / len(wins) if wins else 0.0,
            "avg_loss": sum(losses) / len(losses) if losses else 0.0,
            "best": max(pnls),
            "worst": min(pnls),
            "reasons": reason_counts,
            "top_3_pairs": top[:3],
            "bottom_3_pairs": top[-3:][::-1] if len(top) >= 3 else [],
        }

    def _send(self, date: str, s: dict, equity: float, balance: float):
        if s.get("count", 0) == 0:
            msg = (
                f"📊 <b>{self.label} DAILY REPORT — {date}</b>\n"
                f"  No closed trades today\n"
                f"  Equity: ${equity:,.2f}\n"
                f"  Balance: ${balance:,.2f}"
            )
            self.telegram.send(msg)
            return

        total = s["total_pnl"]
        emoji = "📊" if total >= 0 else "📉"
        wr = s["wins"] / s["count"] * 100 if s["count"] else 0
        rr = (
            abs(s["avg_win"] / s["avg_loss"])
            if s["avg_loss"] != 0
            else float("inf")
        )

        reasons_str = ", ".join(f"{k}:{v}" for k, v in s["reasons"].items())

        top_lines = "\n".join(
            f"    {ins}: ${d['pnl']:+.2f} ({d['cnt']})"
            for ins, d in s["top_3_pairs"]
        )
        bot_lines = "\n".join(
            f"    {ins}: ${d['pnl']:+.2f} ({d['cnt']})"
            for ins, d in s["bottom_3_pairs"]
            if d["pnl"] < 0
        )

        msg = (
            f"{emoji} <b>{self.label} DAILY REPORT — {date}</b>\n"
            f"  <b>P&amp;L: ${total:+.2f}</b>  ({total/balance*100:+.2f}% of bal)\n"
            f"  Trades: {s['count']}  W{s['wins']}/L{s['losses']}/BE{s['breakeven']}  WR {wr:.0f}%\n"
            f"  Avg W: ${s['avg_win']:+.2f}  Avg L: ${s['avg_loss']:+.2f}  R:R {rr:.2f}\n"
            f"  Best: ${s['best']:+.2f}  Worst: ${s['worst']:+.2f}\n"
            f"  Close reasons: {reasons_str}\n"
            f"  <b>Top pairs:</b>\n{top_lines}"
        )
        if bot_lines:
            msg += f"\n  <b>Losing pairs:</b>\n{bot_lines}"
        msg += (
            f"\n  Equity: ${equity:,.2f}  Balance: ${balance:,.2f}"
        )
        self.telegram.send(msg)
