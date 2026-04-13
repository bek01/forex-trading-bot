"""Global Trend Direction Filter — prevents counter-trend trades.

This is the #1 fix for the Apr 13 losses. The bot sold into a raging
uptrend because individual strategies (mean reversion) didn't check
higher timeframe direction.

THIS MODULE SITS BETWEEN STRATEGIES AND ORDER EXECUTION.
Any signal must pass the trend filter before reaching the risk manager.

Rules:
1. Check H4 and Daily EMA(50) direction for the instrument
2. If BOTH H4 + Daily are bullish → BLOCK all SELL signals
3. If BOTH H4 + Daily are bearish → BLOCK all BUY signals
4. If conflicting (H4 up, Daily down) → allow both directions
5. For USD pairs: also check DXY direction as confirmation
6. If ADX(14) on H1 > 30 → market is trending, disable mean reversion
7. If OANDA positioning shows >65% retail on one side → contrarian signal

Regime detection:
- TRENDING: ADX > 30 on H1 → only allow trend-following strategies
- RANGING:  ADX < 20 on H1 → only allow mean-reversion strategies
- MIXED:    ADX 20-30 → allow all strategies
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

try:
    import pandas_ta as ta
except ImportError:
    ta = None

from models import Signal, Side, Candle, FOREX_PAIRS
from strategies.base import candles_to_df

logger = logging.getLogger(__name__)


class TrendDirection:
    """Trend direction for an instrument."""

    def __init__(self):
        self.h4_trend: str = "FLAT"      # UP / DOWN / FLAT
        self.daily_trend: str = "FLAT"    # UP / DOWN / FLAT
        self.h1_adx: float = 0.0         # ADX value on H1
        self.regime: str = "MIXED"        # TRENDING / RANGING / MIXED
        self.dxy_trend: str = "FLAT"      # UP / DOWN / FLAT (USD pairs only)
        self.retail_bias: str = "NEUTRAL" # LONG / SHORT / NEUTRAL
        self.updated_at: Optional[datetime] = None

    @property
    def consensus(self) -> str:
        """Overall direction: UP, DOWN, or FLAT."""
        if self.h4_trend == "UP" and self.daily_trend == "UP":
            return "UP"
        if self.h4_trend == "DOWN" and self.daily_trend == "DOWN":
            return "DOWN"
        return "FLAT"

    def __repr__(self):
        return (
            f"Trend(H4={self.h4_trend}, D={self.daily_trend}, "
            f"ADX={self.h1_adx:.1f}, regime={self.regime}, "
            f"consensus={self.consensus})"
        )


# Strategies that should be BLOCKED in strong trends
MEAN_REVERSION_STRATEGIES = {"mean_reversion"}

# Strategies that should be BLOCKED in ranging markets
TREND_STRATEGIES = {"trend_following"}


class GlobalTrendFilter:
    """
    Checks higher-timeframe trend before allowing any trade.
    Call filter_signal() on every signal before sending to risk manager.
    """

    def __init__(self):
        self._trends: dict[str, TrendDirection] = {}
        self._ema_period = 50
        self._adx_trending_threshold = 30.0
        self._adx_ranging_threshold = 20.0
        self._retail_contrarian_threshold = 65.0  # >65% on one side

    def update_trend(
        self,
        instrument: str,
        h4_candles: list[Candle],
        daily_candles: list[Candle],
        h1_candles: list[Candle],
    ):
        """
        Update trend direction for an instrument.
        Call this when H4 or Daily candles close, or periodically.
        """
        td = self._trends.get(instrument, TrendDirection())

        # H4 trend from EMA(50)
        if h4_candles and len(h4_candles) > self._ema_period + 5:
            td.h4_trend = self._compute_ema_trend(h4_candles)

        # Daily trend from EMA(50)
        if daily_candles and len(daily_candles) > self._ema_period + 5:
            td.daily_trend = self._compute_ema_trend(daily_candles)

        # H1 ADX for regime detection
        if h1_candles and len(h1_candles) > 20:
            td.h1_adx = self._compute_adx(h1_candles)
            if td.h1_adx > self._adx_trending_threshold:
                td.regime = "TRENDING"
            elif td.h1_adx < self._adx_ranging_threshold:
                td.regime = "RANGING"
            else:
                td.regime = "MIXED"

        td.updated_at = datetime.now(timezone.utc)
        self._trends[instrument] = td

    def update_dxy(self, dxy_trend: str):
        """Update DXY trend for all USD pairs."""
        for instrument, td in self._trends.items():
            if "USD" in instrument:
                td.dxy_trend = dxy_trend

    def update_retail_positioning(self, instrument: str, pct_long: float):
        """Update retail positioning (contrarian indicator)."""
        td = self._trends.get(instrument, TrendDirection())
        if pct_long > self._retail_contrarian_threshold:
            td.retail_bias = "LONG"  # crowd is long → contrarian short
        elif pct_long < (100 - self._retail_contrarian_threshold):
            td.retail_bias = "SHORT"  # crowd is short → contrarian long
        else:
            td.retail_bias = "NEUTRAL"
        self._trends[instrument] = td

    def filter_signal(self, signal: Signal) -> tuple[bool, str]:
        """
        Filter a trading signal against the global trend.

        Returns:
            (allowed: bool, reason: str)
        """
        instrument = signal.instrument
        td = self._trends.get(instrument)

        if not td or not td.updated_at:
            # No trend data yet — allow (don't block during startup)
            return True, "no trend data yet"

        consensus = td.consensus

        # === RULE 1: Block counter-trend trades ===
        if consensus == "UP" and signal.side == Side.SELL:
            return False, (
                f"BLOCKED: SELL against uptrend "
                f"(H4={td.h4_trend}, D={td.daily_trend}, ADX={td.h1_adx:.0f})"
            )

        if consensus == "DOWN" and signal.side == Side.BUY:
            return False, (
                f"BLOCKED: BUY against downtrend "
                f"(H4={td.h4_trend}, D={td.daily_trend}, ADX={td.h1_adx:.0f})"
            )

        # === RULE 2: Regime-based strategy filtering ===
        if td.regime == "TRENDING" and signal.strategy in MEAN_REVERSION_STRATEGIES:
            return False, (
                f"BLOCKED: mean reversion in TRENDING regime "
                f"(ADX={td.h1_adx:.0f} > {self._adx_trending_threshold})"
            )

        if td.regime == "RANGING" and signal.strategy in TREND_STRATEGIES:
            return False, (
                f"BLOCKED: trend following in RANGING regime "
                f"(ADX={td.h1_adx:.0f} < {self._adx_ranging_threshold})"
            )

        # === RULE 3: DXY confirmation for USD pairs ===
        if "USD" in instrument and td.dxy_trend != "FLAT":
            # If DXY is UP (USD strong), selling USD pairs is risky
            base, quote = instrument.split("_")
            if base == "USD":
                # USD is base — DXY UP means pair goes UP
                if td.dxy_trend == "UP" and signal.side == Side.SELL:
                    return False, f"BLOCKED: SELL {instrument} while DXY trending UP (USD strong)"
                if td.dxy_trend == "DOWN" and signal.side == Side.BUY:
                    return False, f"BLOCKED: BUY {instrument} while DXY trending DOWN (USD weak)"
            else:
                # USD is quote — DXY UP means pair goes DOWN
                if td.dxy_trend == "UP" and signal.side == Side.BUY:
                    return False, f"BLOCKED: BUY {instrument} while DXY trending UP (USD strong)"
                if td.dxy_trend == "DOWN" and signal.side == Side.SELL:
                    return False, f"BLOCKED: SELL {instrument} while DXY trending DOWN (USD weak)"

        # === RULE 4: Contrarian retail positioning ===
        if td.retail_bias != "NEUTRAL":
            # If crowd is heavily LONG, contrarian says sell (or at least don't buy)
            if td.retail_bias == "LONG" and signal.side == Side.BUY:
                # Don't hard block, but log warning
                logger.warning(
                    f"WARNING: BUY {instrument} but {self._retail_contrarian_threshold}%+ "
                    f"retail is already LONG (contrarian bearish)"
                )
            elif td.retail_bias == "SHORT" and signal.side == Side.SELL:
                logger.warning(
                    f"WARNING: SELL {instrument} but {self._retail_contrarian_threshold}%+ "
                    f"retail is already SHORT (contrarian bullish)"
                )

        return True, f"ALLOWED: trend={consensus}, regime={td.regime}, ADX={td.h1_adx:.0f}"

    def _compute_ema_trend(self, candles: list[Candle]) -> str:
        """Compute trend direction from EMA(50)."""
        df = candles_to_df(candles)
        if df.empty or ta is None:
            return "FLAT"

        ema = ta.ema(df["close"], length=self._ema_period)
        if ema is None or len(ema) < 3:
            return "FLAT"

        close = df["close"].iloc[-1]
        ema_val = ema.iloc[-1]
        ema_prev = ema.iloc[-3]  # 3 bars ago for slope

        if close > ema_val and ema_val > ema_prev:
            return "UP"
        elif close < ema_val and ema_val < ema_prev:
            return "DOWN"
        return "FLAT"

    def _compute_adx(self, candles: list[Candle]) -> float:
        """Compute ADX(14) from H1 candles."""
        df = candles_to_df(candles)
        if df.empty or ta is None:
            return 0.0

        adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
        if adx_df is None or adx_df.empty:
            return 0.0

        return float(adx_df.iloc[-1, 0])

    def get_all_trends(self) -> dict[str, dict]:
        """Get trend info for all instruments (for monitoring)."""
        result = {}
        for inst, td in self._trends.items():
            result[inst] = {
                "h4": td.h4_trend,
                "daily": td.daily_trend,
                "adx": td.h1_adx,
                "regime": td.regime,
                "consensus": td.consensus,
                "dxy": td.dxy_trend,
                "retail": td.retail_bias,
            }
        return result

    def get_trend(self, instrument: str) -> Optional[TrendDirection]:
        return self._trends.get(instrument)
