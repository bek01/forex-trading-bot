"""Range Scalp Strategy — fills the ADX 15-25 FLAT-consensus gap.

Logic:
- Entry BUY: price touches lower BB + RSI < 35 + ADX between 15-25
- Entry SELL: price touches upper BB + RSI > 65 + ADX between 15-25
- Stop loss: 1.0x ATR beyond entry
- Take profit: 1.5R fixed (conservative scalp target)
- Filter: ADX window 15-25 — only fires when market is mildly ranging
- Max 2 trades per pair per session to avoid churn

Designed to sit between `mean_reversion` (extreme RSI, ADX < 35) and
`trend_following` (ADX > 25). It's the piece that fires when nothing else
can — quiet, mildly-ranging markets with a FLAT daily trend.

Size: quarter risk of normal strategies. Conservative scalp.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd

try:
    import pandas_ta as ta
except ImportError:
    ta = None

from models import Candle, Signal, Side, SignalStrength, FOREX_PAIRS
from strategies.base import Strategy, candles_to_df


class RangeScalpStrategy(Strategy):
    name = "range_scalp"
    timeframes = ["M15"]
    instruments = [
        "EUR_USD", "GBP_USD", "EUR_GBP",
        "USD_CHF", "AUD_USD", "USD_CAD",
    ]

    # Bollinger Band + RSI — moderate extremes (less strict than mean_reversion)
    bb_period: int = 20
    bb_std: float = 2.0
    rsi_period: int = 14
    rsi_oversold: float = 35.0
    rsi_overbought: float = 65.0

    # Narrow ADX window — this is the regime the strategy targets
    adx_period: int = 14
    adx_min: float = 15.0
    adx_max: float = 25.0

    # Fixed-R exits
    atr_period: int = 14
    sl_atr_multiplier: float = 1.0
    tp_risk_multiplier: float = 1.5

    # Risk sizing hint — executor halves position when reading this flag
    risk_multiplier: float = 0.25  # quarter of normal risk

    # Churn cap: 2 entries per pair per UTC day
    max_trades_per_day: int = 2

    _daily_count: dict[str, dict] = {}

    def __init__(self):
        super().__init__()
        self._daily_count = {}

    def on_candle(
        self,
        instrument: str,
        timeframe: str,
        candles: list[Candle],
    ) -> Optional[Signal]:
        if timeframe != "M15":
            return None

        if len(candles) < max(self.bb_period, self.rsi_period, self.adx_period, self.atr_period) + 10:
            return None

        if ta is None:
            return None

        df = candles_to_df(candles)
        if df.empty:
            return None

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        count = self._daily_count.get(instrument, {"date": today, "count": 0})
        if count["date"] != today:
            count = {"date": today, "count": 0}
            self._daily_count[instrument] = count
        if count["count"] >= self.max_trades_per_day:
            return None

        bb = ta.bbands(df["close"], length=self.bb_period, std=self.bb_std)
        rsi = ta.rsi(df["close"], length=self.rsi_period)
        adx_df = ta.adx(df["high"], df["low"], df["close"], length=self.adx_period)
        atr = ta.atr(df["high"], df["low"], df["close"], length=self.atr_period)

        if any(x is None or (hasattr(x, "empty") and x.empty) for x in (bb, rsi, adx_df, atr)):
            return None

        close = df["close"].iloc[-1]
        lower_bb = bb.iloc[-1, 0]
        upper_bb = bb.iloc[-1, 2]
        current_rsi = rsi.iloc[-1]
        current_adx = adx_df.iloc[-1, 0]
        current_atr = atr.iloc[-1]

        if current_atr <= 0:
            return None

        # Narrow ADX filter — the whole point of this strategy
        if not (self.adx_min <= current_adx <= self.adx_max):
            return None

        signal = None

        if close <= lower_bb and current_rsi < self.rsi_oversold:
            stop_loss = close - (current_atr * self.sl_atr_multiplier)
            risk = close - stop_loss
            if risk > 0:
                take_profit = close + (risk * self.tp_risk_multiplier)
                signal = Signal(
                    strategy=self.name,
                    instrument=instrument,
                    side=Side.BUY,
                    strength=SignalStrength.MODERATE,
                    entry_price=close,
                    stop_loss=round(stop_loss, 5),
                    take_profit=round(take_profit, 5),
                    reason=f"Range scalp BUY: RSI={current_rsi:.0f}, ADX={current_adx:.0f}",
                    metadata={
                        "rsi": current_rsi,
                        "adx": current_adx,
                        "atr": current_atr,
                        "risk_multiplier": self.risk_multiplier,
                        "timeframe": timeframe,
                    },
                )

        elif close >= upper_bb and current_rsi > self.rsi_overbought:
            stop_loss = close + (current_atr * self.sl_atr_multiplier)
            risk = stop_loss - close
            if risk > 0:
                take_profit = close - (risk * self.tp_risk_multiplier)
                signal = Signal(
                    strategy=self.name,
                    instrument=instrument,
                    side=Side.SELL,
                    strength=SignalStrength.MODERATE,
                    entry_price=close,
                    stop_loss=round(stop_loss, 5),
                    take_profit=round(take_profit, 5),
                    reason=f"Range scalp SELL: RSI={current_rsi:.0f}, ADX={current_adx:.0f}",
                    metadata={
                        "rsi": current_rsi,
                        "adx": current_adx,
                        "atr": current_atr,
                        "risk_multiplier": self.risk_multiplier,
                        "timeframe": timeframe,
                    },
                )

        if signal:
            count["count"] += 1
            self.trade_count += 1
            self.logger.info(f"Signal: {signal.side.value} {instrument} — {signal.reason}")

        return signal
