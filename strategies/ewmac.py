"""EWMAC — Exponentially Weighted Moving Average Crossover, Carver-style.

Reference: Robert Carver's pysystemtrade (github.com/robcarver17/pysystemtrade),
"Systematic Trading" book, and Moskowitz/Ooi/Pedersen 2012 time-series momentum
(JFE). One of only two retail FX strategy families with peer-reviewed multi-
decade out-of-sample evidence.

What this rule does:
- Compute fast EMA minus slow EMA, normalised by recent price volatility
- Convert to a "forecast" — bounded scalar where +10 is neutral-strong-buy,
  -10 neutral-strong-sell, capped at ±20
- Multiple EWMAC pairs (e.g. 16/64 and 32/128) are combined into one forecast
- Forecasts aggregate at portfolio level with vol targeting (see portfolio/)

Timeframe: **daily bars only**. This is position trading. Signals update once
per day at the D1 close; holding periods are weeks to months.

This strategy does NOT emit traditional stop/target Signals. It produces a
Forecast — a continuous directional score. The portfolio sizer turns forecasts
into desired positions, and ProfitManager handles trailing exits once in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import pandas_ta as ta
except ImportError:
    ta = None

from models import Candle
from strategies.base import Strategy, candles_to_df


@dataclass
class Forecast:
    """A continuous directional score for an instrument.

    Positive = long bias, negative = short bias. ±20 cap prevents single-rule
    outliers dominating the portfolio.
    """
    strategy: str
    instrument: str
    value: float  # bounded in [-20, +20]
    metadata: dict


class EWMACStrategy(Strategy):
    """EWMAC(fast, slow) — trend rule.

    Combines two EWMAC crossover variants into a single forecast. Parameter
    choice (16/64 and 32/128) spans roughly weeks-to-quarters trend horizons
    per Carver's pysystemtrade defaults.
    """

    name = "ewmac"
    timeframes = ["D"]  # daily only
    instruments = [
        # Majors + EURJPY per reference architecture. Limited set = fewer
        # correlated positions and cleaner cost profile.
        "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD",
        "USD_CAD", "NZD_USD", "USD_CHF", "EUR_JPY",
    ]

    # Two crossover horizons combined. More variants help in theory but
    # diminishing returns past 3-4.
    variants: list[tuple[int, int]] = [(16, 64), (32, 128)]

    # Volatility lookback for normalisation (daily returns, 35-day window is
    # pysystemtrade default).
    vol_lookback: int = 35

    # Forecast scalar — Carver tunes this so average absolute forecast ≈ 10
    # across history. Calibrated against historical FX data; revisit in
    # backtest if the realised average drifts.
    forecast_scalar: float = 2.65  # 16/64
    forecast_scalar_slow: float = 1.55  # 32/128

    forecast_cap: float = 20.0

    # Minimum bars needed before emitting anything
    _min_bars_required = 200

    def on_candle(
        self,
        instrument: str,
        timeframe: str,
        candles: list[Candle],
    ) -> Optional[Forecast]:
        if timeframe != "D":
            return None
        if ta is None:
            return None
        if len(candles) < self._min_bars_required:
            return None

        df = candles_to_df(candles)
        if df.empty:
            return None

        # Daily returns for vol normalisation
        returns = df["close"].diff().abs()
        price_vol = returns.rolling(self.vol_lookback).mean().iloc[-1]
        if not price_vol or price_vol <= 0:
            return None

        close = df["close"].iloc[-1]

        # Compute raw EWMAC for each variant, normalise, scale, combine
        forecasts = []
        for idx, (fast, slow) in enumerate(self.variants):
            ema_fast = ta.ema(df["close"], length=fast)
            ema_slow = ta.ema(df["close"], length=slow)
            if ema_fast is None or ema_slow is None:
                continue
            raw = ema_fast.iloc[-1] - ema_slow.iloc[-1]
            normalised = raw / price_vol
            scalar = self.forecast_scalar if idx == 0 else self.forecast_scalar_slow
            scaled = normalised * scalar
            capped = max(-self.forecast_cap, min(scaled, self.forecast_cap))
            forecasts.append(capped)

        if not forecasts:
            return None

        combined = sum(forecasts) / len(forecasts)
        # Cap again after combination
        combined = max(-self.forecast_cap, min(combined, self.forecast_cap))

        forecast = Forecast(
            strategy=self.name,
            instrument=instrument,
            value=combined,
            metadata={
                "close": close,
                "price_vol": price_vol,
                "variant_forecasts": forecasts,
                "variants": self.variants,
            },
        )
        self.trade_count += 1
        self.logger.info(
            f"Forecast: {instrument} EWMAC={combined:+.1f} "
            f"(variants={[round(f, 1) for f in forecasts]})"
        )
        return forecast

    def get_required_history(self) -> int:
        return self._min_bars_required
