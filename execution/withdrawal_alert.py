"""Weekly withdrawal alert — OANDA does not expose a programmatic
cash-transfer endpoint (v20 REST only records TRANSFER_FUNDS, never
initiates them). Workflow is Hub-UI only.

This module instead fires a clear actionable alert every Friday near FX
week close with:
  - Current balance
  - Target (keep) balance
  - Recommended transfer amount to the savings sub-account
  - A log line written to logs/withdrawal_alerts.log

User follows the link to OANDA Hub → Transfer → target sub-account and
moves the amount manually (30 seconds).

Triggers at most once per calendar week.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class WithdrawalAlert:
    """Friday-close alert to manually withdraw profits."""

    def __init__(
        self,
        target_balance: float,           # keep at this level after withdrawal
        savings_account_id: str,         # e.g. "5302313"
        min_transfer_amount: float = 5.0,  # don't bother below this
        trigger_hour_utc: int = 20,      # 20:00 UTC Fri ≈ NY close
        trigger_minute_utc: int = 55,
        telegram_notifier: Optional[Callable[[str], None]] = None,
        alert_log_path: str = "logs/withdrawal_alerts.log",
    ):
        self.target_balance = target_balance
        self.savings_account_id = savings_account_id
        self.min_transfer = min_transfer_amount
        self.trigger_hour = trigger_hour_utc
        self.trigger_minute = trigger_minute_utc
        self.telegram = telegram_notifier
        self.alert_log_path = alert_log_path
        self._last_run_week: Optional[str] = None  # ISO year-week

    def check_and_run(self, balance: float):
        """Call from main loop. Triggers Fri around configured time, once/week."""
        now = datetime.now(timezone.utc)
        # Friday = 4
        if now.weekday() != 4:
            return
        if now.hour != self.trigger_hour:
            return
        if now.minute < self.trigger_minute:
            return

        iso_week = f"{now.isocalendar().year}-W{now.isocalendar().week:02d}"
        if self._last_run_week == iso_week:
            return

        self._last_run_week = iso_week
        self._fire_alert(balance, now)

    def _fire_alert(self, balance: float, now: datetime):
        excess = balance - self.target_balance
        if excess < self.min_transfer:
            logger.info(
                f"Weekly withdrawal check: balance ${balance:.2f} <= target "
                f"${self.target_balance:.2f} + min_transfer ${self.min_transfer:.2f} "
                f"— no transfer needed"
            )
            return

        msg = (
            f"\n{'='*60}\n"
            f"WEEKLY WITHDRAWAL ALERT — {now.strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"{'='*60}\n"
            f"  Balance:             ${balance:.2f}\n"
            f"  Target (keep):       ${self.target_balance:.2f}\n"
            f"  Suggested transfer:  ${excess:.2f}\n"
            f"  To sub-account:      {self.savings_account_id}\n\n"
            f"  ACTION: Log into OANDA Hub → Funds → Transfer\n"
            f"          (OANDA REST API does not support programmatic transfers)\n"
            f"{'='*60}\n"
        )
        logger.warning(msg)

        # Append to persistent alert log
        try:
            Path(self.alert_log_path).parent.mkdir(exist_ok=True, parents=True)
            with open(self.alert_log_path, "a") as f:
                f.write(
                    f"{now.isoformat()}\tbalance={balance:.2f}\t"
                    f"target={self.target_balance:.2f}\ttransfer={excess:.2f}\t"
                    f"to={self.savings_account_id}\n"
                )
        except Exception as e:
            logger.error(f"Failed to write withdrawal alert log: {e}")

        # Telegram (best-effort)
        if self.telegram:
            try:
                self.telegram(msg)
            except Exception as e:
                logger.error(f"Telegram withdrawal alert failed: {e}")
