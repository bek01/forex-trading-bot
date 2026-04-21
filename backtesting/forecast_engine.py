"""Forecast-based backtest engine — Carver-style EWMAC + portfolio vol targeting.

Vectorized pandas/numpy implementation (no vectorbt dependency). Simulates:
  1. EWMAC forecasts per day per instrument
  2. Portfolio-level forecast combination + vol-targeted sizing
  3. Daily position rebalancing
  4. Spread + slippage costs
  5. Equity curve tracking

Outputs: equity curve, trade list, summary metrics.

This engine is deliberately SEPARATE from `backtesting/engine.py` (which is
Signal-based for the current architecture). We don't mix the two paradigms.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# --- Forecast rules (vectorized versions of strategies/ewmac.py logic) ---

def ewmac_forecast_vectorized(
    close: pd.Series,
    fast: int,
    slow: int,
    vol_lookback: int = 35,
    forecast_scalar: float = 2.65,
    cap: float = 20.0,
) -> pd.Series:
    """Compute vol-normalised EWMAC forecast for a price series.

    Must match the live logic in strategies/ewmac.py.
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    raw = ema_fast - ema_slow
    # Daily absolute returns as vol proxy (matches live)
    vol = close.diff().abs().rolling(vol_lookback).mean()
    normalised = raw / vol.replace(0, np.nan)
    scaled = normalised * forecast_scalar
    return scaled.clip(-cap, cap)


def combined_ewmac_forecast(
    close: pd.Series,
    variants: list[tuple[int, int]] = [(16, 64), (32, 128)],
    # Carver's canonical scalars (2.65, 1.55) are calibrated against futures
    # universes with higher vol. Our FX-only universe produces forecasts
    # averaging ~6 instead of ~10, so we apply a ~1.55× FX-specific boost.
    scalars: list[float] = [4.10, 2.40],
    cap: float = 20.0,
) -> pd.Series:
    """Combine multiple EWMAC variants into one forecast (average)."""
    forecasts = []
    for (fast, slow), scalar in zip(variants, scalars):
        f = ewmac_forecast_vectorized(close, fast, slow, forecast_scalar=scalar, cap=cap)
        forecasts.append(f)
    combined = pd.concat(forecasts, axis=1).mean(axis=1)
    return combined.clip(-cap, cap)


# --- Portfolio simulation ---

@dataclass
class BacktestTrade:
    timestamp: pd.Timestamp
    instrument: str
    units_delta: float  # + = buy, - = sell
    price: float
    cost: float  # spread + slippage
    note: str = ""


@dataclass
class PortfolioBacktestResult:
    start: pd.Timestamp
    end: pd.Timestamp
    initial_capital: float
    final_equity: float
    equity_curve: pd.Series
    positions: pd.DataFrame  # index=date, columns=instruments, values=units
    forecasts: pd.DataFrame  # index=date, columns=instruments, values=combined forecast
    trades: list[BacktestTrade] = field(default_factory=list)

    @property
    def total_return_pct(self) -> float:
        return (self.final_equity / self.initial_capital - 1) * 100

    @property
    def n_trades(self) -> int:
        # Count non-zero position changes
        return len(self.trades)

    @property
    def sharpe_ratio(self) -> float:
        daily_ret = self.equity_curve.pct_change().dropna()
        if daily_ret.std() == 0:
            return 0.0
        return float((daily_ret.mean() / daily_ret.std()) * np.sqrt(252))

    @property
    def sortino_ratio(self) -> float:
        daily_ret = self.equity_curve.pct_change().dropna()
        downside = daily_ret[daily_ret < 0]
        if len(downside) == 0 or downside.std() == 0:
            return float("inf") if daily_ret.mean() > 0 else 0.0
        return float((daily_ret.mean() / downside.std()) * np.sqrt(252))

    @property
    def max_drawdown_pct(self) -> float:
        cummax = self.equity_curve.cummax()
        dd = (self.equity_curve - cummax) / cummax
        return float(dd.min() * 100)

    @property
    def annualised_vol(self) -> float:
        daily_ret = self.equity_curve.pct_change().dropna()
        return float(daily_ret.std() * np.sqrt(252) * 100)

    def summary(self) -> str:
        return (
            f"Backtest {self.start.date()} → {self.end.date()}  "
            f"({(self.end - self.start).days} days)\n"
            f"  Initial:   £{self.initial_capital:,.0f}\n"
            f"  Final:     £{self.final_equity:,.0f}  ({self.total_return_pct:+.1f}%)\n"
            f"  Sharpe:    {self.sharpe_ratio:.2f}\n"
            f"  Sortino:   {self.sortino_ratio:.2f}\n"
            f"  Max DD:    {self.max_drawdown_pct:.1f}%\n"
            f"  Ann. vol:  {self.annualised_vol:.1f}%\n"
            f"  Trades:    {self.n_trades}\n"
        )


class ForecastPortfolioBacktester:
    """Run a multi-instrument forecast-based backtest."""

    def __init__(
        self,
        annual_vol_target: float = 0.20,
        idm: float = 1.2,
        fdm: float = 1.0,  # only one rule (ewmac) here — no FDM boost
        forecast_avg_magnitude: float = 10.0,
        trading_days: int = 252,
        # Costs (per round-trip, in fraction of notional)
        spread_pips: dict[str, float] = None,
        slippage_pips: float = 0.5,
        # Position change threshold — don't trade tiny deltas (costs dominate)
        # Carver's "position buffering" concept: only rebalance when the
        # target position differs from current by more than `buffer_frac` of
        # the average absolute position. Absolute min is a floor.
        min_position_delta_units: float = 500.0,
        buffer_frac: float = 0.10,  # 10% position buffer
    ):
        self.annual_vol_target = annual_vol_target
        self.idm = idm
        self.fdm = fdm
        self.forecast_avg_magnitude = forecast_avg_magnitude
        self.trading_days = trading_days
        self.slippage_pips = slippage_pips
        self.min_position_delta_units = min_position_delta_units
        self.buffer_frac = buffer_frac
        # Default realistic OANDA spreads in pips for majors
        self.spread_pips = spread_pips or {
            "EUR_USD": 0.8, "GBP_USD": 1.2, "USD_JPY": 0.8, "AUD_USD": 1.0,
            "USD_CAD": 1.4, "NZD_USD": 1.6, "USD_CHF": 1.2, "EUR_JPY": 1.2,
        }

    @staticmethod
    def _pip_size(instrument: str) -> float:
        return 0.01 if "JPY" in instrument else 0.0001

    def run(
        self,
        price_data: dict[str, pd.DataFrame],
        instrument_weights: Optional[dict[str, float]] = None,
        initial_capital: float = 100_000.0,
        vol_lookback: int = 35,
    ) -> PortfolioBacktestResult:
        """Run a multi-instrument backtest.

        Args:
            price_data: {instrument: DataFrame with columns [open,high,low,close,volume]}
            instrument_weights: per-instrument weight (defaults to equal)
            initial_capital: account starting equity
            vol_lookback: for daily price vol (matches live strategy default)
        """
        instruments = list(price_data.keys())
        n = len(instruments)
        if n == 0:
            raise ValueError("No price data provided")

        if instrument_weights is None:
            instrument_weights = {inst: 1.0 / n for inst in instruments}

        # Build a common date index (intersection of all instruments)
        date_index = None
        for inst, df in price_data.items():
            if date_index is None:
                date_index = df.index
            else:
                date_index = date_index.intersection(df.index)
        date_index = date_index.sort_values()
        if len(date_index) < 250:
            raise ValueError(f"Only {len(date_index)} common dates — need ≥250")

        # Compute forecasts per instrument (vectorised)
        forecasts = pd.DataFrame(index=date_index, columns=instruments, dtype=float)
        closes = pd.DataFrame(index=date_index, columns=instruments, dtype=float)
        daily_price_vol = pd.DataFrame(index=date_index, columns=instruments, dtype=float)
        for inst in instruments:
            df = price_data[inst].loc[date_index]
            closes[inst] = df["close"]
            forecasts[inst] = combined_ewmac_forecast(df["close"])
            daily_price_vol[inst] = df["close"].diff().abs().rolling(vol_lookback).mean()

        # Walk forward, compute target units, simulate P&L
        equity_curve = pd.Series(index=date_index, dtype=float)
        positions = pd.DataFrame(index=date_index, columns=instruments, dtype=float).fillna(0.0)

        equity = initial_capital
        prev_positions = {inst: 0.0 for inst in instruments}
        trades: list[BacktestTrade] = []

        for i, dt in enumerate(date_index):
            # Skip warmup period
            if i < max(200, vol_lookback):
                equity_curve.iloc[i] = equity
                for inst in instruments:
                    positions.loc[dt, inst] = prev_positions[inst]
                continue

            # P&L from previous day's positions: position × (today_close − yesterday_close)
            if i > 0:
                prev_dt = date_index[i - 1]
                for inst in instruments:
                    pos = prev_positions[inst]
                    if pos == 0:
                        continue
                    price_change = closes.loc[dt, inst] - closes.loc[prev_dt, inst]
                    equity += pos * price_change

            # Target position per instrument
            daily_vol_budget = equity * self.annual_vol_target / np.sqrt(self.trading_days)
            new_positions = dict(prev_positions)
            for inst in instruments:
                fc = forecasts.loc[dt, inst]
                dpv = daily_price_vol.loc[dt, inst]
                if pd.isna(fc) or pd.isna(dpv) or dpv <= 0:
                    continue
                combined = fc * self.fdm
                combined = max(-20.0, min(combined, 20.0))
                w = instrument_weights.get(inst, 0.0)
                if w <= 0:
                    continue
                notional = daily_vol_budget * self.idm * w * (combined / self.forecast_avg_magnitude)
                target_units = notional / dpv if dpv > 0 else 0.0
                # Position buffering per Carver: only rebalance when the
                # delta exceeds the larger of (absolute floor, buffer_frac *
                # |target|). This is the primary turnover control.
                delta = target_units - prev_positions[inst]
                buffer = max(
                    self.min_position_delta_units,
                    self.buffer_frac * abs(target_units),
                )
                if abs(delta) < buffer:
                    continue
                # Apply cost: spread + slippage, in quote ccy per unit
                pip_size = self._pip_size(inst)
                cost_pips = self.spread_pips.get(inst, 1.0) + self.slippage_pips
                cost_ccy = abs(delta) * pip_size * cost_pips
                equity -= cost_ccy
                new_positions[inst] = target_units
                trades.append(BacktestTrade(
                    timestamp=dt,
                    instrument=inst,
                    units_delta=delta,
                    price=closes.loc[dt, inst],
                    cost=cost_ccy,
                ))

            prev_positions = new_positions
            for inst in instruments:
                positions.loc[dt, inst] = prev_positions[inst]
            equity_curve.iloc[i] = equity

        # Forward-fill any NaN equity (shouldn't happen if logic correct)
        equity_curve = equity_curve.ffill().fillna(initial_capital)

        return PortfolioBacktestResult(
            start=date_index[0],
            end=date_index[-1],
            initial_capital=initial_capital,
            final_equity=equity,
            equity_curve=equity_curve,
            positions=positions,
            forecasts=forecasts,
            trades=trades,
        )
