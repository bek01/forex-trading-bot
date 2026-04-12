"""Mean Reversion Strategy — Bollinger Band bounce with RSI confirmation.

Logic:
- Entry BUY: Price touches lower BB + RSI < 30 + price within ATR range (not trending)
- Entry SELL: Price touches upper BB + RSI > 70 + price within ATR range
- Stop loss: 1.5 × ATR beyond entry
- Take profit: Middle BB (mean reversion target)
- Filter: ADX < 25 (ranging market, not trending)

Best session: Asian session (lower volatility, more ranging)
Best pairs: EUR_USD, GBP_USD, EUR_GBP in ranging conditions
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


class MeanReversionStrategy(Strategy):
    name = "mean_reversion"
    timeframes = ["M15", "H1"]

    # Parameters (tunable via config or optimization)
    bb_period: int = 20
    bb_std: float = 2.0
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    atr_period: int = 14
    adx_period: int = 14
    adx_max: float = 25.0  # only trade when ADX < this (ranging market)
    sl_atr_multiplier: float = 1.5
    primary_timeframe: str = "M15"

    def on_candle(
        self,
        instrument: str,
        timeframe: str,
        candles: list[Candle],
    ) -> Optional[Signal]:
        if timeframe != self.primary_timeframe:
            return None

        if len(candles) < max(self.bb_period, self.rsi_period, self.adx_period) + 10:
            return None

        df = candles_to_df(candles)
        if df.empty or ta is None:
            return None

        # Calculate indicators
        bb = ta.bbands(df["close"], length=self.bb_period, std=self.bb_std)
        if bb is None or bb.empty:
            return None

        rsi = ta.rsi(df["close"], length=self.rsi_period)
        atr = ta.atr(df["high"], df["low"], df["close"], length=self.atr_period)
        adx_df = ta.adx(df["high"], df["low"], df["close"], length=self.adx_period)

        if rsi is None or atr is None or adx_df is None:
            return None

        # Get latest values
        close = df["close"].iloc[-1]
        prev_close = df["close"].iloc[-2]
        lower_bb = bb.iloc[-1, 0]  # BBL
        mid_bb = bb.iloc[-1, 1]    # BBM
        upper_bb = bb.iloc[-1, 2]  # BBU
        current_rsi = rsi.iloc[-1]
        current_atr = atr.iloc[-1]
        current_adx = adx_df.iloc[-1, 0] if not adx_df.empty else 30  # ADX column

        pip_size = FOREX_PAIRS.get(instrument, {}).get("pip_size", 0.0001)

        # Filter: only trade in ranging markets
        if current_adx > self.adx_max:
            return None

        # Filter: ATR must be reasonable (not during news spikes)
        if current_atr <= 0:
            return None

        signal = None

        # --- BUY signal: price at/below lower BB + RSI oversold ---
        if close <= lower_bb and current_rsi < self.rsi_oversold:
            stop_loss = close - (current_atr * self.sl_atr_multiplier)
            take_profit = mid_bb  # target the mean

            # Only take if R:R is decent
            risk = close - stop_loss
            reward = take_profit - close
            if risk > 0 and reward / risk >= 1.5:
                strength = SignalStrength.STRONG if current_rsi < 25 else SignalStrength.MODERATE
                signal = Signal(
                    strategy=self.name,
                    instrument=instrument,
                    side=Side.BUY,
                    strength=strength,
                    entry_price=close,
                    stop_loss=round(stop_loss, 5),
                    take_profit=round(take_profit, 5),
                    reason=f"BB bounce BUY: RSI={current_rsi:.1f}, ADX={current_adx:.1f}",
                    metadata={
                        "rsi": current_rsi,
                        "adx": current_adx,
                        "atr": current_atr,
                        "bb_lower": lower_bb,
                        "bb_mid": mid_bb,
                        "timeframe": timeframe,
                    },
                )

        # --- SELL signal: price at/above upper BB + RSI overbought ---
        elif close >= upper_bb and current_rsi > self.rsi_overbought:
            stop_loss = close + (current_atr * self.sl_atr_multiplier)
            take_profit = mid_bb

            risk = stop_loss - close
            reward = close - take_profit
            if risk > 0 and reward / risk >= 1.5:
                strength = SignalStrength.STRONG if current_rsi > 75 else SignalStrength.MODERATE
                signal = Signal(
                    strategy=self.name,
                    instrument=instrument,
                    side=Side.SELL,
                    strength=strength,
                    entry_price=close,
                    stop_loss=round(stop_loss, 5),
                    take_profit=round(take_profit, 5),
                    reason=f"BB bounce SELL: RSI={current_rsi:.1f}, ADX={current_adx:.1f}",
                    metadata={
                        "rsi": current_rsi,
                        "adx": current_adx,
                        "atr": current_atr,
                        "bb_upper": upper_bb,
                        "bb_mid": mid_bb,
                        "timeframe": timeframe,
                    },
                )

        if signal:
            self.trade_count += 1
            self.logger.info(f"Signal: {signal.side.value} {instrument} — {signal.reason}")

        return signal

    def should_close(self, instrument, timeframe, candles, entry_price, side):
        """Close if RSI reverses past midpoint (mean reversion complete)."""
        if timeframe != self.primary_timeframe or len(candles) < self.rsi_period + 5:
            return None

        df = candles_to_df(candles)
        rsi = ta.rsi(df["close"], length=self.rsi_period)
        if rsi is None:
            return None

        current_rsi = rsi.iloc[-1]

        # If we're long and RSI is now overbought, take profit
        if side == "BUY" and current_rsi > 65:
            return "rsi_reversal_exit"
        # If we're short and RSI is now oversold, take profit
        if side == "SELL" and current_rsi < 35:
            return "rsi_reversal_exit"

        return None
