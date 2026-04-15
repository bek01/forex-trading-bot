#!/usr/bin/env python3
"""
Trading Bot — Main Entry Point

Runs the main event loop:
1. Initialize broker connection, data feeds, strategies, risk manager
2. Load historical candles
3. Enter main loop: poll candles → feed strategies → risk check → execute
4. Monitor account state, emit heartbeats, handle graceful shutdown

Usage:
    python main.py                    # Run bot (uses .env config)
    python main.py --paper            # Force paper trading mode
    python main.py --backtest         # Run backtest instead of live
    python main.py --status           # Show current status and exit
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from config import get_config, AppConfig
from data.candle_manager import CandleManager
from data.trend_filter import GlobalTrendFilter
from data.sentiment import SentimentData
from db.database import Database
from event_bus import Event, bus
from execution.trailing_stop import ProfitManager
from execution.broker import OandaBroker
from execution.order_executor import OrderExecutor
from models import AccountState, Position, Signal, Tick
from monitoring.telegram_bot import TelegramNotifier, TelegramPoller
from risk.risk_manager import RiskManager
from strategies.base import Strategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.trend_following import TrendFollowingStrategy
from strategies.london_breakout import LondonBreakoutStrategy
from strategies.confluence import ConfluenceStrategy
from strategies.session_momentum import SessionMomentumStrategy
from strategies.stat_arb import StatArbStrategy

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO"):
    """Configure logging with rotation."""
    Path("logs").mkdir(exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)

    # File handler with rotation (2MB, 3 backups)
    file_handler = RotatingFileHandler(
        "logs/trading_bot.log",
        maxBytes=2 * 1024 * 1024,
        backupCount=3,
    )
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper()))
    root.addHandler(console)
    root.addHandler(file_handler)

    # Reduce noise from HTTP libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


class TradingBot:
    """Main trading bot orchestrator."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.running = False
        self._shutdown_requested = False

        # Components (initialized in start())
        self.db: Optional[Database] = None
        self.broker: Optional[OandaBroker] = None
        self.candle_manager: Optional[CandleManager] = None
        self.risk_manager: Optional[RiskManager] = None
        self.executor: Optional[OrderExecutor] = None
        self.telegram: Optional[TelegramNotifier] = None
        self.telegram_poller: Optional[TelegramPoller] = None
        self.trend_filter: Optional[GlobalTrendFilter] = None
        self.sentiment: Optional[SentimentData] = None
        self.trailing_stop: Optional[ProfitManager] = None

        # Strategies
        self.strategies: list[Strategy] = []

        # Timing
        self._last_heartbeat = 0.0
        self._last_account_sync = 0.0
        self._last_equity_snapshot = 0.0
        self._last_daily_reset: Optional[str] = None

        # Account state
        self.account = AccountState()

    def start(self):
        """Initialize all components and start the main loop."""
        logger.info("=" * 60)
        logger.info(f"Trading Bot starting — mode: {self.config.trading_mode}")
        logger.info("=" * 60)

        # Initialize components
        self.db = Database(self.config.db_path)
        self.broker = OandaBroker(self.config.broker)
        self.risk_manager = RiskManager(self.config.risk)
        self.executor = OrderExecutor(self.broker, self.config)
        self.telegram = TelegramNotifier(self.config.telegram)

        # Trend filter + sentiment — prevents counter-trend trading
        self.trend_filter = GlobalTrendFilter()
        try:
            self.sentiment = SentimentData(
                self.config.broker.api_token,
                self.config.broker.rest_url,
            )
        except Exception as e:
            logger.warning(f"Sentiment data init failed: {e}")
            self.sentiment = None

        # Profit manager — partial profit scaling + trailing stops
        self.trailing_stop = ProfitManager(
            broker=self.broker,
            stage1_pips=10.0,        # +10 pips → close 25%, SL to breakeven
            stage2_pips=20.0,        # +20 pips → close 25% more, start trailing
            stage3_pips=30.0,        # +30 pips → close 25% more, tighten trail
            stage1_close_pct=0.25,   # close 25% at each stage
            stage2_close_pct=0.25,
            stage3_close_pct=0.25,
            trail_distance_pips=10.0,  # trail 10 pips behind after stage 2
            trail_tight_pips=7.0,      # tighter 7 pip trail after stage 3
            min_trail_step_pips=3.0,
        )

        # Load strategies
        self._load_strategies()

        # Set up candle manager
        self.candle_manager = CandleManager(
            broker=self.broker,
            instruments=self.config.strategy.instruments,
            timeframes=self.config.strategy.timeframes,
            buffer_size=self.config.strategy.candle_buffer_size,
        )

        # Subscribe to events
        self._setup_event_handlers()

        # Setup Telegram commands
        if self.config.telegram.enabled:
            self.telegram_poller = TelegramPoller(self.telegram)
            self._setup_telegram_commands()

        # Initial account sync
        try:
            self.account = self.broker.get_account_state()
            self.account.peak_equity = self.account.equity
            self.risk_manager.reset_daily()
            logger.info(f"Account: balance=${self.account.balance:.2f}, equity=${self.account.equity:.2f}")
        except Exception as e:
            logger.error(f"Failed to connect to broker: {e}")
            logger.error("Check OANDA_ACCOUNT_ID and OANDA_API_TOKEN in .env")
            sys.exit(1)

        # Load candle history
        logger.info("Loading candle history...")
        self.candle_manager.initialize()

        # Notify
        strategy_names = [s.name for s in self.strategies if s.enabled]
        self.telegram.notify_startup(
            self.config.trading_mode, self.account.balance, strategy_names
        )
        bus.emit(Event.BOT_STARTED)

        # Enter main loop
        self.running = True
        self._main_loop()

    def _load_strategies(self):
        """Initialize and register trading strategies."""
        cfg = self.config.strategy

        if cfg.mean_reversion_enabled:
            self.strategies.append(MeanReversionStrategy())
        if cfg.trend_following_enabled:
            self.strategies.append(TrendFollowingStrategy())
        if cfg.london_breakout_enabled:
            self.strategies.append(LondonBreakoutStrategy())
        if cfg.confluence_enabled:
            self.strategies.append(ConfluenceStrategy())
        if cfg.session_momentum_enabled:
            self.strategies.append(SessionMomentumStrategy())
        if cfg.stat_arb_enabled:
            self.strategies.append(StatArbStrategy())

        enabled = [s.name for s in self.strategies if s.enabled]
        logger.info(f"Strategies loaded: {enabled}")

    def _setup_event_handlers(self):
        """Wire up event handlers."""
        bus.subscribe(Event.CANDLE_CLOSE, self._on_candle_close)
        bus.subscribe(Event.SIGNAL, self._on_signal)
        bus.subscribe(Event.ORDER_FILLED, self._on_order_filled)
        bus.subscribe(Event.POSITION_OPENED, self._on_position_opened)
        bus.subscribe(Event.POSITION_CLOSED, self._on_position_closed)
        bus.subscribe(Event.KILL_SWITCH, self._on_kill_switch)
        bus.subscribe(Event.DAILY_LOSS_LIMIT, self._on_daily_loss_limit)
        bus.subscribe(Event.DRAWDOWN_WARNING, self._on_drawdown_warning)

    def _setup_telegram_commands(self):
        """Register Telegram command handlers."""
        self.telegram_poller.register_command("status", self._cmd_status)
        self.telegram_poller.register_command("kill", self._cmd_kill)
        self.telegram_poller.register_command("resume", self._cmd_resume)
        self.telegram_poller.register_command("trades", self._cmd_trades)
        self.telegram_poller.register_command("help", self._cmd_help)

    def _main_loop(self):
        """Main bot loop — runs until shutdown."""
        logger.info("Entering main loop")

        while self.running and not self._shutdown_requested:
            try:
                now = time.time()

                # --- Daily reset check ---
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                if today != self._last_daily_reset:
                    self.risk_manager.reset_daily()
                    self._last_daily_reset = today
                    logger.info(f"New trading day: {today}")

                # --- Poll candles for new data ---
                self.candle_manager.poll()

                # --- Profit manager: partial closes + trailing stops (every 10s) ---
                if self.trailing_stop:
                    self.trailing_stop.check_and_update(check_interval_sec=10.0)

                # --- Sync account state ---
                if now - self._last_account_sync >= self.config.account_sync_interval_sec:
                    self._sync_account()
                    self._last_account_sync = now

                # --- Equity snapshot ---
                if now - self._last_equity_snapshot >= self.config.equity_snapshot_interval_sec:
                    self._save_equity_snapshot()
                    self._last_equity_snapshot = now

                # --- Poll Telegram commands ---
                if self.telegram_poller:
                    self.telegram_poller.poll_once()

                # --- Heartbeat ---
                if now - self._last_heartbeat >= self.config.heartbeat_interval_sec:
                    self._heartbeat()
                    self._last_heartbeat = now

                # Sleep to avoid hammering APIs
                time.sleep(self.config.candle_poll_interval_sec)

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}", exc_info=True)
                self.telegram.notify_error(f"Main loop error: {e}")
                time.sleep(10)  # back off on error

        self._shutdown()

    def _on_candle_close(self, data: dict):
        """Feed new candle to all strategies + update trend filter."""
        instrument = data["instrument"]
        timeframe = data["timeframe"]

        # --- Update global trend filter on H4/D/H1 candle closes ---
        if timeframe in ("H4", "D", "H1") and self.trend_filter:
            try:
                h4 = self.candle_manager.get_candles(instrument, "H4")
                daily = self.candle_manager.get_candles(instrument, "D")
                h1 = self.candle_manager.get_candles(instrument, "H1")
                self.trend_filter.update_trend(instrument, h4, daily, h1)
            except Exception as e:
                logger.debug(f"Trend filter update error {instrument}: {e}")

        # --- Update DXY + positioning periodically ---
        if timeframe == "H1" and instrument == "EUR_USD" and self.sentiment:
            try:
                dxy = self.sentiment.get_dxy_trend()
                self.trend_filter.update_dxy(dxy)
            except Exception:
                pass
            try:
                pos = self.sentiment.get_oanda_positioning(instrument)
                if pos:
                    self.trend_filter.update_retail_positioning(
                        instrument, pos.get("pct_long", 50)
                    )
            except Exception:
                pass

        # --- Feed to strategies ---
        for strategy in self.strategies:
            if not strategy.enabled:
                continue
            if timeframe not in strategy.timeframes:
                continue
            if strategy.instruments and instrument not in strategy.instruments:
                continue

            try:
                candles = self.candle_manager.get_candles(instrument, timeframe)
                signal = strategy.on_candle(instrument, timeframe, candles)
                if signal:
                    bus.emit(Event.SIGNAL, signal)
            except Exception as e:
                logger.error(
                    f"Strategy {strategy.name} error on {instrument}/{timeframe}: {e}",
                    exc_info=True,
                )

    def _on_signal(self, signal: Signal):
        """Process a trading signal through trend filter → risk manager → executor."""
        logger.info(
            f"Signal received: {signal.side.value} {signal.instrument} "
            f"[{signal.strategy}] — {signal.reason}"
        )

        # === TREND FILTER (prevents counter-trend trades) ===
        if self.trend_filter:
            allowed, reason = self.trend_filter.filter_signal(signal)
            if not allowed:
                logger.info(f"Trend filter: {reason}")
                bus.emit(Event.RISK_BLOCKED, {
                    "signal": signal,
                    "reason": f"TREND_FILTER: {reason}",
                })
                return

        # Get current price for spread check
        try:
            tick = self.broker.get_price(signal.instrument)
        except Exception as e:
            logger.error(f"Failed to get price for {signal.instrument}: {e}")
            return

        # Get open positions
        broker_positions = self.broker.get_open_positions()

        # Risk check
        decision = self.risk_manager.evaluate_signal(signal, tick, broker_positions)
        logger.info(f"Risk decision: {decision}")

        if not decision.approved:
            bus.emit(Event.RISK_BLOCKED, {
                "signal": signal,
                "reason": decision.reason,
            })
            return

        # Execute
        order = self.executor.execute_signal(signal, decision.units)
        if order:
            self.db.save_order(order)

    def _on_order_filled(self, order):
        """Log filled order."""
        self.db.save_order(order)
        logger.info(f"Order filled: {order.instrument} @ {order.fill_price}")

    def _on_position_opened(self, position: Position):
        """Log new position and notify."""
        self.db.save_position(position)
        self.telegram.notify_trade_opened(
            position.instrument, position.side.value,
            position.units, position.entry_price,
            position.stop_loss, position.take_profit,
            position.strategy,
        )

    def _on_position_closed(self, data: dict):
        """Log closed position and notify."""
        self.telegram.notify_trade_closed(
            data.get("instrument", ""),
            data.get("side", ""),
            data.get("realized_pnl", 0),
            data.get("close_reason", ""),
            data.get("strategy", ""),
        )

    def _on_kill_switch(self, data):
        """Handle kill switch — close everything."""
        reason = data.get("reason", "unknown")
        logger.critical(f"KILL SWITCH: {reason}")
        self.telegram.notify_kill_switch(reason)

        # Close all positions
        if self.broker:
            closed = self.broker.close_all_positions()
            logger.critical(f"Emergency: closed {closed} positions")

        # Cancel all pending orders
        if self.executor:
            cancelled = self.executor.cancel_all_orders()
            logger.critical(f"Emergency: cancelled {cancelled} orders")

    def _on_daily_loss_limit(self, data):
        """Handle daily loss limit hit."""
        logger.warning(f"Daily loss limit: ${data.get('pnl', 0):.2f}")
        self.telegram.send(
            f"⚠️ Daily loss limit hit: ${data.get('pnl', 0):.2f} "
            f"({data.get('pct', 0):.1f}%)\nTrading halted for today."
        )

    def _on_drawdown_warning(self, data):
        """Handle drawdown warning."""
        self.telegram.notify_drawdown(data.get("drawdown", 0), "WARNING")

    def _sync_account(self):
        """Sync account state from broker."""
        try:
            self.account = self.broker.get_account_state()
            if self.account.equity > self.account.peak_equity:
                self.account.peak_equity = self.account.equity
            bus.emit(Event.ACCOUNT_UPDATE, self.account)
        except Exception as e:
            logger.error(f"Account sync failed: {e}")

    def _save_equity_snapshot(self):
        """Save periodic equity snapshot to DB."""
        if self.db and self.account.equity > 0:
            self.db.save_equity_snapshot(self.account)

    def _heartbeat(self):
        """Periodic health check."""
        risk_status = self.risk_manager.get_status()
        candle_stats = self.candle_manager.get_stats() if self.candle_manager else {}
        exec_stats = self.executor.get_stats() if self.executor else {}

        logger.info(
            f"♥ Equity=${self.account.equity:.2f} "
            f"DD={self.account.drawdown_pct:.1f}% "
            f"Positions={risk_status['open_positions']}/{risk_status['max_positions']} "
            f"Halted={risk_status['trading_halted']} "
            f"Daily PnL=${risk_status['daily_pnl']:.2f}"
        )
        bus.emit(Event.HEARTBEAT)

    def _shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down...")
        self.running = False
        bus.emit(Event.BOT_STOPPED)

        # Note: positions are LEFT OPEN (they have stop losses)
        # Kill switch explicitly closes them

        if self.telegram:
            self.telegram.send("🔴 Bot shutting down. Positions left open with SL/TP.")
            self.telegram.close()
        if self.broker:
            self.broker.close()
        if self.db:
            self.db.close()

        logger.info("Shutdown complete")

    def request_shutdown(self):
        """Request graceful shutdown from outside."""
        self._shutdown_requested = True

    # --- Telegram command handlers ---

    def _cmd_status(self) -> str:
        risk = self.risk_manager.get_status()
        positions = self.broker.get_open_positions() if self.broker else []
        return (
            f"📊 <b>BOT STATUS</b>\n"
            f"Mode: {self.config.trading_mode}\n"
            f"Equity: ${self.account.equity:,.2f}\n"
            f"Balance: ${self.account.balance:,.2f}\n"
            f"Drawdown: {self.account.drawdown_pct:.1f}%\n"
            f"Positions: {len(positions)}/{risk['max_positions']}\n"
            f"Daily P&L: ${risk['daily_pnl']:+.2f}\n"
            f"Halted: {risk['trading_halted']}\n"
            f"Strategies: {[s.name for s in self.strategies if s.enabled]}"
        )

    def _cmd_kill(self) -> str:
        self.risk_manager.manual_kill("Telegram /kill command")
        return "🛑 Kill switch activated. All positions being closed."

    def _cmd_resume(self) -> str:
        self.risk_manager.manual_reset()
        return "▶️ Trading resumed. Kill switch cleared."

    def _cmd_trades(self) -> str:
        if not self.db:
            return "No database connection"
        trades = self.db.get_closed_positions(limit=5)
        if not trades:
            return "No closed trades yet."
        lines = ["📋 <b>RECENT TRADES</b>"]
        for t in trades:
            emoji = "✅" if t["realized_pnl"] > 0 else "❌"
            lines.append(
                f"{emoji} {t['instrument']} {t['side']} "
                f"${t['realized_pnl']:+.2f} ({t['close_reason']})"
            )
        return "\n".join(lines)

    def _cmd_help(self) -> str:
        return (
            "🤖 <b>COMMANDS</b>\n"
            "/status — Bot status & P&L\n"
            "/trades — Recent closed trades\n"
            "/kill — Emergency stop (closes all)\n"
            "/resume — Resume after kill\n"
            "/help — This message"
        )


def main():
    parser = argparse.ArgumentParser(description="Forex/Stock Trading Bot")
    parser.add_argument("--paper", action="store_true", help="Force paper trading")
    parser.add_argument("--backtest", action="store_true", help="Run backtest mode")
    parser.add_argument("--status", action="store_true", help="Show status and exit")
    args = parser.parse_args()

    config = get_config()

    if args.paper:
        config.trading_mode = "paper"
        config.broker.environment = "practice"

    setup_logging(config.log_level)

    if args.backtest:
        logger.info("Backtest mode — see backtesting/run_backtest.py")
        return

    if args.status:
        bot = TradingBot(config)
        try:
            bot.broker = OandaBroker(config.broker)
            state = bot.broker.get_account_state()
            print(f"Balance: ${state.balance:,.2f}")
            print(f"Equity:  ${state.equity:,.2f}")
            print(f"Margin:  ${state.margin_used:,.2f}")
            print(f"Open:    {state.open_position_count} positions")
        except Exception as e:
            print(f"Error connecting to broker: {e}")
        return

    # Set up signal handlers for graceful shutdown
    bot = TradingBot(config)

    def handle_signal(signum, frame):
        logger.info(f"Signal {signum} received, requesting shutdown...")
        bot.request_shutdown()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Run
    bot.start()


if __name__ == "__main__":
    main()
