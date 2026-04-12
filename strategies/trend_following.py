"""Trend Following Strategy — Dual EMA crossover with ADX filter.

Logic:
- Entry BUY: EMA(20) crosses above EMA(50) + ADX > 25 + daily EMA(200) uptrend
- Entry SELL: EMA(20) crosses below EMA(50) + ADX > 25 + daily EMA(200) downtrend
- Stop loss: 2 × ATR from entry
- Take profit: 3 × ATR from entry (1.5 R:R)
- Filter: ADX > 25 (only trade strong trends)

Best session: London + New York overlap (highest volatility)
Best pairs: GBP_USD, EUR_USD (strong trending behavior)
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

try:
    import pandas_ta as ta
except ImportError:
    ta = None

from models import Candle, Signal, Side, SignalStrength, FOREX_PAIRS
from strategies.base import Strategy, candles_to_df


class TrendFollowingStrategy(Strategy):
    name = "trend_following"
    timeframes = ["M15", "H1", "D"]

    # Parameters
    fast_ema: int = 20
    slow_ema: int = 50
    trend_ema: int = 200  # for daily trend filter
    adx_period: int = 14
    adx_min: float = 25.0  # only trade when ADX > this (trending market)
    atr_period: int = 14
    sl_atr_multiplier: float = 2.0
    tp_atr_multiplier: float = 3.0
    entry_timeframe: str = "H1"
    trend_timeframe: str = "D"

    # State: daily trend direction (updated when D candle arrives)
    _daily_trend: dict[str, str] = {}  # instrument -> "UP" / "DOWN" / "FLAT"

    def on_candle(
        self,
        instrument: str,
        timeframe: str,
        candles: list[Candle],
    ) -> Optional[Signal]:
        df = candles_to_df(candles)
        if df.empty or ta is None:
            return None

        # Update daily trend when daily candle arrives
        if timeframe == self.trend_timeframe:
            self._update_daily_trend(instrument, df)
            return None

        # Only generate signals on entry timeframe
        if timeframe != self.entry_timeframe:
            return None

        if len(candles) < self.slow_ema + 10:
            return None

        # Calculate indicators
        ema_fast = ta.ema(df["close"], length=self.fast_ema)
        ema_slow = ta.ema(df["close"], length=self.slow_ema)
        adx_df = ta.adx(df["high"], df["low"], df["close"], length=self.adx_period)
        atr = ta.atr(df["high"], df["low"], df["close"], length=self.atr_period)

        if ema_fast is None or ema_slow is None or adx_df is None or atr is None:
            return None

        # Latest values
        fast_now = ema_fast.iloc[-1]
        fast_prev = ema_fast.iloc[-2]
        slow_now = ema_slow.iloc[-1]
        slow_prev = ema_slow.iloc[-2]
        close = df["close"].iloc[-1]
        current_adx = adx_df.iloc[-1, 0]
        current_atr = atr.iloc[-1]

        # ADX filter: must be trending
        if current_adx < self.adx_min:
            return None

        # Check for crossover
        bullish_cross = fast_prev <= slow_prev and fast_now > slow_now
        bearish_cross = fast_prev >= slow_prev and fast_now < slow_now

        if not bullish_cross and not bearish_cross:
            return None

        # Daily trend filter
        daily_trend = self._daily_trend.get(instrument, "FLAT")

        signal = None

        if bullish_cross and daily_trend in ("UP", "FLAT"):
            stop_loss = close - (current_atr * self.sl_atr_multiplier)
            take_profit = close + (current_atr * self.tp_atr_multiplier)

            strength = SignalStrength.STRONG if daily_trend == "UP" else SignalStrength.MODERATE
            signal = Signal(
                strategy=self.name,
                instrument=instrument,
                side=Side.BUY,
                strength=strength,
                entry_price=close,
                stop_loss=round(stop_loss, 5),
                take_profit=round(take_profit, 5),
                reason=f"EMA cross BUY: ADX={current_adx:.1f}, daily={daily_trend}",
                metadata={
                    "adx": current_adx,
                    "atr": current_atr,
                    "ema_fast": fast_now,
                    "ema_slow": slow_now,
                    "daily_trend": daily_trend,
                    "timeframe": timeframe,
                },
            )

        elif bearish_cross and daily_trend in ("DOWN", "FLAT"):
            stop_loss = close + (current_atr * self.sl_atr_multiplier)
            take_profit = close - (current_atr * self.tp_atr_multiplier)

            strength = SignalStrength.STRONG if daily_trend == "DOWN" else SignalStrength.MODERATE
            signal = Signal(
                strategy=self.name,
                instrument=instrument,
                side=Side.SELL,
                strength=strength,
                entry_price=close,
                stop_loss=round(stop_loss, 5),
                take_profit=round(take_profit, 5),
                reason=f"EMA cross SELL: ADX={current_adx:.1f}, daily={daily_trend}",
                metadata={
                    "adx": current_adx,
                    "atr": current_atr,
                    "ema_fast": fast_now,
                    "ema_slow": slow_now,
                    "daily_trend": daily_trend,
                    "timeframe": timeframe,
                },
            )

        if signal:
            self.trade_count += 1
            self.logger.info(f"Signal: {signal.side.value} {instrument} — {signal.reason}")

        return signal

    def _update_daily_trend(self, instrument: str, df: pd.DataFrame):
        """Determine daily trend using EMA(200)."""
        if len(df) < self.trend_ema + 5:
            self._daily_trend[instrument] = "FLAT"
            return

        ema_200 = ta.ema(df["close"], length=self.trend_ema)
        if ema_200 is None:
            self._daily_trend[instrument] = "FLAT"
            return

        close = df["close"].iloc[-1]
        ema_val = ema_200.iloc[-1]

        # 0.5% threshold to avoid whipsaws around the EMA
        threshold = ema_val * 0.005
        if close > ema_val + threshold:
            self._daily_trend[instrument] = "UP"
        elif close < ema_val - threshold:
            self._daily_trend[instrument] = "DOWN"
        else:
            self._daily_trend[instrument] = "FLAT"

        self.logger.debug(f"{instrument} daily trend: {self._daily_trend[instrument]}")

    def should_close(self, instrument, timeframe, candles, entry_price, side):
        """Close if EMA crossover reverses (trend exhaustion)."""
        if timeframe != self.entry_timeframe or len(candles) < self.slow_ema + 5:
            return None

        df = candles_to_df(candles)
        ema_fast = ta.ema(df["close"], length=self.fast_ema)
        ema_slow = ta.ema(df["close"], length=self.slow_ema)

        if ema_fast is None or ema_slow is None:
            return None

        fast_now = ema_fast.iloc[-1]
        slow_now = ema_slow.iloc[-1]

        if side == "BUY" and fast_now < slow_now:
            return "ema_reverse_cross"
        if side == "SELL" and fast_now > slow_now:
            return "ema_reverse_cross"

        return None
