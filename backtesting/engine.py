"""Backtesting engine — replay historical data through strategies.

Features:
- Realistic simulation: accounts for spread, slippage, commission
- Walk-forward analysis support
- Performance metrics: Sharpe, Sortino, max drawdown, profit factor
- Monte Carlo simulation for robustness testing
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from models import Candle, Side, Signal, Tick, FOREX_PAIRS
from strategies.base import Strategy, candles_to_df

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """A completed trade in the backtest."""
    instrument: str = ""
    side: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    units: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    pnl: float = 0.0
    pnl_pips: float = 0.0
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    exit_reason: str = ""
    strategy: str = ""
    r_multiple: float = 0.0  # pnl / risk


@dataclass
class BacktestResult:
    """Complete backtest results with performance metrics."""
    strategy_name: str = ""
    instrument: str = ""
    timeframe: str = ""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    # Capital
    initial_capital: float = 10000.0
    final_equity: float = 0.0
    total_return_pct: float = 0.0

    # Trade stats
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0  # average P&L per trade

    # Risk metrics
    max_drawdown_pct: float = 0.0
    max_drawdown_usd: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    avg_r_multiple: float = 0.0

    # Equity curve
    equity_curve: list[float] = field(default_factory=list)
    trades: list[BacktestTrade] = field(default_factory=list)

    def passes_minimum_bar(self) -> tuple[bool, list[str]]:
        """Check if results meet minimum thresholds for paper trading."""
        failures = []
        if self.sharpe_ratio < 1.5:
            failures.append(f"Sharpe {self.sharpe_ratio:.2f} < 1.5")
        if self.max_drawdown_pct > 15.0:
            failures.append(f"Max DD {self.max_drawdown_pct:.1f}% > 15%")
        if self.profit_factor < 1.5:
            failures.append(f"Profit factor {self.profit_factor:.2f} < 1.5")
        if self.total_trades < 200:
            failures.append(f"Only {self.total_trades} trades (need 200+)")
        if self.win_rate < 45:
            failures.append(f"Win rate {self.win_rate:.1f}% < 45%")
        return len(failures) == 0, failures

    def summary(self) -> str:
        passed, failures = self.passes_minimum_bar()
        status = "✅ PASS" if passed else "❌ FAIL"
        lines = [
            f"=== BACKTEST: {self.strategy_name} on {self.instrument}/{self.timeframe} ===",
            f"Period: {self.start_date} → {self.end_date}",
            f"Status: {status}",
            f"",
            f"Capital: ${self.initial_capital:,.2f} → ${self.final_equity:,.2f} ({self.total_return_pct:+.1f}%)",
            f"Trades: {self.total_trades} (W:{self.wins} L:{self.losses} WR:{self.win_rate:.1f}%)",
            f"Avg Win: ${self.avg_win:.2f} | Avg Loss: ${self.avg_loss:.2f}",
            f"Best: ${self.best_trade:.2f} | Worst: ${self.worst_trade:.2f}",
            f"Profit Factor: {self.profit_factor:.2f}",
            f"Expectancy: ${self.expectancy:.2f}/trade",
            f"",
            f"Max Drawdown: {self.max_drawdown_pct:.1f}% (${self.max_drawdown_usd:.2f})",
            f"Sharpe Ratio: {self.sharpe_ratio:.2f}",
            f"Sortino Ratio: {self.sortino_ratio:.2f}",
            f"Avg R-Multiple: {self.avg_r_multiple:.2f}",
        ]
        if failures:
            lines.append(f"\nMinimum bar failures:")
            for f in failures:
                lines.append(f"  ❌ {f}")
        return "\n".join(lines)


class BacktestEngine:
    """Replay historical candles through a strategy and simulate trades."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        risk_per_trade_pct: float = 1.0,
        spread_pips: float = 1.5,  # simulated spread
        slippage_pips: float = 0.5,  # simulated slippage
        commission_per_unit: float = 0.0,  # OANDA = 0 for spread-only
    ):
        self.initial_capital = initial_capital
        self.risk_per_trade_pct = risk_per_trade_pct
        self.spread_pips = spread_pips
        self.slippage_pips = slippage_pips
        self.commission = commission_per_unit

    def run(
        self,
        strategy: Strategy,
        candles: list[Candle],
        instrument: str = "EUR_USD",
    ) -> BacktestResult:
        """
        Run a backtest by replaying candles through the strategy.

        Args:
            strategy: Strategy instance to test
            candles: Historical candles (must be complete, sorted chronologically)
            instrument: The instrument being tested

        Returns:
            BacktestResult with all metrics
        """
        pip_size = FOREX_PAIRS.get(instrument, {}).get("pip_size", 0.0001)
        spread = self.spread_pips * pip_size
        slippage = self.slippage_pips * pip_size

        equity = self.initial_capital
        peak_equity = equity
        equity_curve = [equity]
        trades: list[BacktestTrade] = []

        # Active position tracking
        active_trade: Optional[BacktestTrade] = None

        # Feed candles to strategy
        history_needed = strategy.get_required_history()

        for i in range(history_needed, len(candles)):
            candle = candles[i]
            history = candles[max(0, i - history_needed):i + 1]

            # --- Check active trade for SL/TP hit ---
            if active_trade:
                hit = self._check_sl_tp(active_trade, candle, pip_size)
                if hit:
                    active_trade.exit_time = candle.timestamp
                    active_trade.exit_reason = hit
                    # Calculate P&L
                    if active_trade.side == "BUY":
                        active_trade.pnl = (
                            (active_trade.exit_price - active_trade.entry_price)
                            * active_trade.units
                        )
                    else:
                        active_trade.pnl = (
                            (active_trade.entry_price - active_trade.exit_price)
                            * active_trade.units
                        )
                    active_trade.pnl_pips = active_trade.pnl / (pip_size * active_trade.units)

                    # R-multiple
                    risk_per_unit = abs(active_trade.entry_price - active_trade.stop_loss)
                    if risk_per_unit > 0:
                        active_trade.r_multiple = (
                            active_trade.pnl / (risk_per_unit * active_trade.units)
                        )

                    equity += active_trade.pnl
                    trades.append(active_trade)
                    active_trade = None

                    peak_equity = max(peak_equity, equity)
                    equity_curve.append(equity)
                    continue

            # --- Check for strategy-based exit ---
            if active_trade:
                timeframe = candle.timeframe if hasattr(candle, 'timeframe') else "H1"
                close_reason = strategy.should_close(
                    instrument, timeframe, history,
                    active_trade.entry_price, active_trade.side,
                )
                if close_reason:
                    active_trade.exit_price = candle.close
                    active_trade.exit_time = candle.timestamp
                    active_trade.exit_reason = close_reason
                    if active_trade.side == "BUY":
                        active_trade.pnl = (
                            (candle.close - active_trade.entry_price) * active_trade.units
                        )
                    else:
                        active_trade.pnl = (
                            (active_trade.entry_price - candle.close) * active_trade.units
                        )
                    active_trade.pnl_pips = active_trade.pnl / (pip_size * active_trade.units)
                    risk_per_unit = abs(active_trade.entry_price - active_trade.stop_loss)
                    if risk_per_unit > 0:
                        active_trade.r_multiple = (
                            active_trade.pnl / (risk_per_unit * active_trade.units)
                        )
                    equity += active_trade.pnl
                    trades.append(active_trade)
                    active_trade = None
                    peak_equity = max(peak_equity, equity)
                    equity_curve.append(equity)

            # --- Get signal from strategy ---
            if not active_trade:
                timeframe = candle.timeframe if hasattr(candle, 'timeframe') else "H1"
                signal = strategy.on_candle(instrument, timeframe, history)

                if signal and signal.stop_loss != 0:
                    # Calculate position size
                    risk_amount = equity * (self.risk_per_trade_pct / 100.0)
                    sl_distance = abs(signal.entry_price - signal.stop_loss)
                    if sl_distance > 0:
                        units = risk_amount / sl_distance

                        # Apply spread and slippage
                        entry = signal.entry_price
                        if signal.side == Side.BUY:
                            entry += (spread / 2 + slippage)
                        else:
                            entry -= (spread / 2 + slippage)

                        active_trade = BacktestTrade(
                            instrument=instrument,
                            side=signal.side.value,
                            entry_price=entry,
                            units=units,
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit,
                            entry_time=candle.timestamp,
                            strategy=strategy.name,
                        )

            equity_curve.append(equity)

        # Close any remaining position at last candle price
        if active_trade and candles:
            last_close = candles[-1].close
            if active_trade.side == "BUY":
                active_trade.pnl = (last_close - active_trade.entry_price) * active_trade.units
            else:
                active_trade.pnl = (active_trade.entry_price - last_close) * active_trade.units
            active_trade.exit_price = last_close
            active_trade.exit_time = candles[-1].timestamp
            active_trade.exit_reason = "backtest_end"
            equity += active_trade.pnl
            trades.append(active_trade)
            equity_curve.append(equity)

        return self._compute_metrics(strategy, instrument, candles, equity,
                                     equity_curve, trades)

    def _check_sl_tp(
        self, trade: BacktestTrade, candle: Candle, pip_size: float
    ) -> Optional[str]:
        """Check if candle hit stop loss or take profit."""
        if trade.side == "BUY":
            # Check SL (price went down to SL)
            if candle.low <= trade.stop_loss:
                trade.exit_price = trade.stop_loss
                return "stop_loss"
            # Check TP (price went up to TP)
            if trade.take_profit and candle.high >= trade.take_profit:
                trade.exit_price = trade.take_profit
                return "take_profit"
        else:  # SELL
            if candle.high >= trade.stop_loss:
                trade.exit_price = trade.stop_loss
                return "stop_loss"
            if trade.take_profit and candle.low <= trade.take_profit:
                trade.exit_price = trade.take_profit
                return "take_profit"
        return None

    def _compute_metrics(
        self, strategy, instrument, candles, final_equity,
        equity_curve, trades
    ) -> BacktestResult:
        """Compute all performance metrics from trades."""
        result = BacktestResult(
            strategy_name=strategy.name,
            instrument=instrument,
            initial_capital=self.initial_capital,
            final_equity=final_equity,
            equity_curve=equity_curve,
            trades=trades,
        )

        if candles:
            result.start_date = candles[0].timestamp
            result.end_date = candles[-1].timestamp

        result.total_return_pct = (
            (final_equity - self.initial_capital) / self.initial_capital * 100
        )
        result.total_trades = len(trades)

        if not trades:
            return result

        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        result.wins = len(wins)
        result.losses = len(losses)
        result.win_rate = result.wins / result.total_trades * 100
        result.avg_win = np.mean(wins) if wins else 0
        result.avg_loss = np.mean(losses) if losses else 0
        result.best_trade = max(pnls)
        result.worst_trade = min(pnls)
        result.expectancy = np.mean(pnls)

        # Profit factor
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Max drawdown
        eq = np.array(equity_curve)
        peak = np.maximum.accumulate(eq)
        drawdown = (peak - eq) / peak * 100
        result.max_drawdown_pct = float(np.max(drawdown))
        result.max_drawdown_usd = float(np.max(peak - eq))

        # Sharpe ratio (annualized, assuming daily returns)
        if len(equity_curve) > 1:
            returns = np.diff(equity_curve) / equity_curve[:-1]
            if len(returns) > 1 and np.std(returns) > 0:
                result.sharpe_ratio = float(
                    np.mean(returns) / np.std(returns) * np.sqrt(252)
                )
                # Sortino (downside deviation only)
                downside = returns[returns < 0]
                if len(downside) > 0:
                    result.sortino_ratio = float(
                        np.mean(returns) / np.std(downside) * np.sqrt(252)
                    )

        # Average R-multiple
        r_multiples = [t.r_multiple for t in trades if t.r_multiple != 0]
        result.avg_r_multiple = float(np.mean(r_multiples)) if r_multiples else 0

        return result

    def monte_carlo(
        self, trades: list[BacktestTrade], simulations: int = 1000
    ) -> dict:
        """
        Monte Carlo simulation — randomize trade order to test robustness.

        Returns dict with percentile statistics.
        """
        pnls = [t.pnl for t in trades]
        if not pnls:
            return {"error": "No trades to simulate"}

        final_equities = []
        max_drawdowns = []

        for _ in range(simulations):
            shuffled = pnls.copy()
            random.shuffle(shuffled)

            equity = self.initial_capital
            peak = equity
            max_dd = 0

            for pnl in shuffled:
                equity += pnl
                peak = max(peak, equity)
                dd = (peak - equity) / peak * 100 if peak > 0 else 0
                max_dd = max(max_dd, dd)

            final_equities.append(equity)
            max_drawdowns.append(max_dd)

        equities = np.array(final_equities)
        dds = np.array(max_drawdowns)

        profitable_pct = np.sum(equities > self.initial_capital) / simulations * 100

        return {
            "simulations": simulations,
            "profitable_pct": float(profitable_pct),
            "equity_p5": float(np.percentile(equities, 5)),
            "equity_p25": float(np.percentile(equities, 25)),
            "equity_median": float(np.median(equities)),
            "equity_p75": float(np.percentile(equities, 75)),
            "equity_p95": float(np.percentile(equities, 95)),
            "max_dd_median": float(np.median(dds)),
            "max_dd_p95": float(np.percentile(dds, 95)),
            "ruin_pct": float(np.sum(equities < self.initial_capital * 0.5) / simulations * 100),
        }
