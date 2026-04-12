"""Telegram bot for monitoring, alerts, and kill switch.

Commands:
    /status  — Current bot status, positions, P&L
    /equity  — Equity curve summary
    /trades  — Recent closed trades
    /kill    — Emergency kill switch (close all, halt)
    /resume  — Resume trading after halt
    /help    — Show commands

Events pushed automatically:
    - Trade opened/closed
    - Daily P&L report (configurable time)
    - Error alerts
    - Drawdown warnings
    - Kill switch activation
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional, Callable

import httpx

from config import TelegramConfig

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Simple Telegram notification sender (no polling — just push)."""

    def __init__(self, config: TelegramConfig):
        self.config = config
        self.enabled = config.enabled and config.bot_token and config.chat_id
        self.base_url = f"https://api.telegram.org/bot{config.bot_token}"
        self._client = httpx.Client(timeout=10.0) if self.enabled else None

        # Callbacks for commands (set by main.py)
        self._command_handlers: dict[str, Callable] = {}

    def send(self, message: str, parse_mode: str = "HTML"):
        """Send a message to the configured chat."""
        if not self.enabled:
            logger.debug(f"Telegram disabled, would send: {message[:100]}...")
            return

        try:
            self._client.post(
                f"{self.base_url}/sendMessage",
                json={
                    "chat_id": self.config.chat_id,
                    "text": message,
                    "parse_mode": parse_mode,
                    "disable_web_page_preview": True,
                },
            )
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")

    # --- Formatted messages ---

    def notify_trade_opened(self, instrument: str, side: str, units: float,
                            entry: float, sl: float, tp: float, strategy: str):
        self.send(
            f"📈 <b>TRADE OPENED</b>\n"
            f"  {side} {units:.0f} {instrument}\n"
            f"  Entry: {entry:.5f}\n"
            f"  SL: {sl:.5f} | TP: {tp:.5f}\n"
            f"  Strategy: {strategy}"
        )

    def notify_trade_closed(self, instrument: str, side: str, pnl: float,
                            reason: str, strategy: str):
        emoji = "✅" if pnl >= 0 else "❌"
        self.send(
            f"{emoji} <b>TRADE CLOSED</b>\n"
            f"  {side} {instrument}\n"
            f"  P&L: ${pnl:+.2f}\n"
            f"  Reason: {reason}\n"
            f"  Strategy: {strategy}"
        )

    def notify_daily_report(self, date: str, pnl: float, trades: int,
                            wins: int, equity: float):
        emoji = "📊" if pnl >= 0 else "📉"
        wr = f"{wins/trades*100:.0f}%" if trades > 0 else "N/A"
        self.send(
            f"{emoji} <b>DAILY REPORT — {date}</b>\n"
            f"  P&L: ${pnl:+.2f}\n"
            f"  Trades: {trades} (WR: {wr})\n"
            f"  Equity: ${equity:,.2f}"
        )

    def notify_error(self, error: str):
        self.send(f"🚨 <b>ERROR</b>\n{error}")

    def notify_drawdown(self, pct: float, level: str):
        self.send(
            f"⚠️ <b>DRAWDOWN {level}</b>\n"
            f"  Current: {pct:.1f}% from peak"
        )

    def notify_kill_switch(self, reason: str):
        self.send(
            f"🛑 <b>KILL SWITCH ACTIVATED</b>\n"
            f"  {reason}\n"
            f"  All trading halted. Use /resume to restart."
        )

    def notify_startup(self, mode: str, balance: float, strategies: list[str]):
        self.send(
            f"🤖 <b>BOT STARTED</b>\n"
            f"  Mode: {mode}\n"
            f"  Balance: ${balance:,.2f}\n"
            f"  Strategies: {', '.join(strategies)}"
        )

    def close(self):
        if self._client:
            self._client.close()


class TelegramPoller:
    """
    Polls for incoming Telegram commands (kill switch, status requests).
    Run in a separate thread.
    """

    def __init__(self, notifier: TelegramNotifier):
        self.notifier = notifier
        self._offset = 0
        self._handlers: dict[str, Callable] = {}

    def register_command(self, command: str, handler: Callable):
        """Register a /command handler."""
        self._handlers[command.lstrip("/")] = handler

    def poll_once(self):
        """Poll for new messages and dispatch commands."""
        if not self.notifier.enabled:
            return

        try:
            resp = self.notifier._client.get(
                f"{self.notifier.base_url}/getUpdates",
                params={"offset": self._offset, "timeout": 5},
            )
            data = resp.json()

            for update in data.get("result", []):
                self._offset = update["update_id"] + 1
                message = update.get("message", {})
                text = message.get("text", "")
                chat_id = str(message.get("chat", {}).get("id", ""))

                # Security: only respond to configured chat
                if chat_id != self.notifier.config.chat_id:
                    continue

                if text.startswith("/"):
                    cmd = text.split()[0].lstrip("/").lower()
                    handler = self._handlers.get(cmd)
                    if handler:
                        try:
                            result = handler()
                            if isinstance(result, str):
                                self.notifier.send(result)
                        except Exception as e:
                            self.notifier.send(f"Command error: {e}")
                    else:
                        self.notifier.send(f"Unknown command: /{cmd}\nUse /help for available commands.")

        except Exception as e:
            logger.error(f"Telegram poll error: {e}")
