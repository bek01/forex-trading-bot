"""Statistical Arbitrage Strategy — Mean reversion of correlated pair spreads.

Logic:
- Track the price ratio (spread) between correlated pairs: EUR_USD / GBP_USD
- Calculate z-score of spread using rolling window (50 periods)
- Entry: when z-score > 2.0, sell the overpriced pair and buy the underpriced
- Exit: when z-score returns to 0.5 (or below)
- Emergency exit: z-score > 3.5 (spread diverging further — cut losses)
- Creates 2 positions (1 long, 1 short) that are market-neutral
- Timeframe: H1
- Pairs: EUR_USD vs GBP_USD (correlation ~0.87)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except ImportError:
    ta = None

from models import Candle, Signal, Side, SignalStrength, FOREX_PAIRS
from strategies.base import Strategy, candles_to_df


class StatArbStrategy(Strategy):
    name = "stat_arb"
    timeframes = ["H1"]
    instruments = ["EUR_USD", "GBP_USD"]

    # --- Tunable parameters ---

    # Pair definition
    pair_a: str = "EUR_USD"  # numerator in spread ratio
    pair_b: str = "GBP_USD"  # denominator in spread ratio

    # Z-score parameters
    lookback_period: int = 50       # rolling window for mean/std
    entry_z: float = 2.0           # enter when |z| > this
    exit_z: float = 0.5            # exit when |z| < this
    emergency_z: float = 3.5       # emergency exit when |z| > this

    # ATR for SL/TP (fallback, primary exits are z-score based)
    atr_period: int = 14
    sl_atr_multiplier: float = 3.0  # wide SL — z-score exit is primary
    tp_atr_multiplier: float = 2.0

    # Timeframe
    entry_timeframe: str = "H1"

    # State: store candle data for both pairs
    _pair_closes: dict[str, list[float]] = {}
    _pair_timestamps: dict[str, list] = {}
    _last_z_score: float = 0.0
    _in_position: bool = False
    _position_side: str = ""  # "LONG_A_SHORT_B" or "SHORT_A_LONG_B"

    def __init__(self):
        super().__init__()
        self._pair_closes = {self.pair_a: [], self.pair_b: []}
        self._pair_timestamps = {self.pair_a: [], self.pair_b: []}
        self._last_z_score = 0.0
        self._in_position = False
        self._position_side = ""

    def on_candle(
        self,
        instrument: str,
        timeframe: str,
        candles: list[Candle],
    ) -> Optional[Signal]:
        if timeframe != self.entry_timeframe:
            return None

        if instrument not in (self.pair_a, self.pair_b):
            return None

        if len(candles) < self.lookback_period + 10:
            return None

        df = candles_to_df(candles)
        if df.empty:
            return None

        # Store latest closes for this instrument
        closes = df["close"].tolist()
        timestamps = df.index.tolist()
        self._pair_closes[instrument] = closes
        self._pair_timestamps[instrument] = timestamps

        # Only compute z-score when we have data for BOTH pairs
        if not self._pair_closes.get(self.pair_a) or not self._pair_closes.get(self.pair_b):
            return None

        # Align the two series by length (use the shorter)
        closes_a = self._pair_closes[self.pair_a]
        closes_b = self._pair_closes[self.pair_b]
        min_len = min(len(closes_a), len(closes_b))

        if min_len < self.lookback_period + 5:
            return None

        # Use the most recent data, aligned by position
        arr_a = np.array(closes_a[-min_len:])
        arr_b = np.array(closes_b[-min_len:])

        # Compute spread ratio
        with np.errstate(divide="ignore", invalid="ignore"):
            spread = arr_a / arr_b

        if np.any(np.isnan(spread)) or np.any(np.isinf(spread)):
            return None

        # Rolling z-score of spread
        spread_series = pd.Series(spread)
        rolling_mean = spread_series.rolling(self.lookback_period).mean()
        rolling_std = spread_series.rolling(self.lookback_period).std()

        if rolling_std.iloc[-1] is None or rolling_std.iloc[-1] == 0:
            return None

        z_score = (spread_series.iloc[-1] - rolling_mean.iloc[-1]) / rolling_std.iloc[-1]

        if np.isnan(z_score) or np.isinf(z_score):
            return None

        self._last_z_score = z_score

        # Don't open new position if already in one (managed externally)
        if self._in_position:
            return None

        # Calculate ATR for SL sizing
        atr_val = 0.0
        if ta is not None:
            atr_series = ta.atr(df["high"], df["low"], df["close"], length=self.atr_period)
            if atr_series is not None:
                atr_val = atr_series.iloc[-1]

        if atr_val <= 0:
            atr_val = df["close"].iloc[-1] * 0.005  # fallback: 0.5% of price

        signal = None
        close = df["close"].iloc[-1]

        # --- Z-score > entry threshold: spread too wide, sell overpriced ---
        if z_score > self.entry_z:
            # Pair A is overpriced relative to Pair B
            # Strategy: SELL Pair A (this signal), BUY Pair B (separate signal)
            # We emit a signal for the CURRENT instrument being processed
            if instrument == self.pair_a:
                # Sell A (overpriced)
                stop_loss = close + (atr_val * self.sl_atr_multiplier)
                take_profit = close - (atr_val * self.tp_atr_multiplier)

                signal = Signal(
                    strategy=self.name,
                    instrument=instrument,
                    side=Side.SELL,
                    strength=SignalStrength.STRONG if z_score > 2.5 else SignalStrength.MODERATE,
                    entry_price=close,
                    stop_loss=round(stop_loss, 5),
                    take_profit=round(take_profit, 5),
                    reason=f"StatArb SELL {self.pair_a}: z={z_score:.2f}, spread overextended",
                    metadata={
                        "z_score": z_score,
                        "spread": float(spread_series.iloc[-1]),
                        "spread_mean": float(rolling_mean.iloc[-1]),
                        "spread_std": float(rolling_std.iloc[-1]),
                        "pair_role": "overpriced",
                        "hedge_pair": self.pair_b,
                        "hedge_side": "BUY",
                        "atr": atr_val,
                        "timeframe": timeframe,
                    },
                )
            elif instrument == self.pair_b:
                # Buy B (underpriced)
                stop_loss = close - (atr_val * self.sl_atr_multiplier)
                take_profit = close + (atr_val * self.tp_atr_multiplier)

                signal = Signal(
                    strategy=self.name,
                    instrument=instrument,
                    side=Side.BUY,
                    strength=SignalStrength.STRONG if z_score > 2.5 else SignalStrength.MODERATE,
                    entry_price=close,
                    stop_loss=round(stop_loss, 5),
                    take_profit=round(take_profit, 5),
                    reason=f"StatArb BUY {self.pair_b}: z={z_score:.2f}, spread overextended",
                    metadata={
                        "z_score": z_score,
                        "spread": float(spread_series.iloc[-1]),
                        "spread_mean": float(rolling_mean.iloc[-1]),
                        "spread_std": float(rolling_std.iloc[-1]),
                        "pair_role": "underpriced",
                        "hedge_pair": self.pair_a,
                        "hedge_side": "SELL",
                        "atr": atr_val,
                        "timeframe": timeframe,
                    },
                )

        # --- Z-score < -entry threshold: spread compressed, buy overpriced ---
        elif z_score < -self.entry_z:
            # Pair B is overpriced relative to Pair A
            if instrument == self.pair_a:
                # Buy A (underpriced)
                stop_loss = close - (atr_val * self.sl_atr_multiplier)
                take_profit = close + (atr_val * self.tp_atr_multiplier)

                signal = Signal(
                    strategy=self.name,
                    instrument=instrument,
                    side=Side.BUY,
                    strength=SignalStrength.STRONG if z_score < -2.5 else SignalStrength.MODERATE,
                    entry_price=close,
                    stop_loss=round(stop_loss, 5),
                    take_profit=round(take_profit, 5),
                    reason=f"StatArb BUY {self.pair_a}: z={z_score:.2f}, spread compressed",
                    metadata={
                        "z_score": z_score,
                        "spread": float(spread_series.iloc[-1]),
                        "spread_mean": float(rolling_mean.iloc[-1]),
                        "spread_std": float(rolling_std.iloc[-1]),
                        "pair_role": "underpriced",
                        "hedge_pair": self.pair_b,
                        "hedge_side": "SELL",
                        "atr": atr_val,
                        "timeframe": timeframe,
                    },
                )
            elif instrument == self.pair_b:
                # Sell B (overpriced)
                stop_loss = close + (atr_val * self.sl_atr_multiplier)
                take_profit = close - (atr_val * self.tp_atr_multiplier)

                signal = Signal(
                    strategy=self.name,
                    instrument=instrument,
                    side=Side.SELL,
                    strength=SignalStrength.STRONG if z_score < -2.5 else SignalStrength.MODERATE,
                    entry_price=close,
                    stop_loss=round(stop_loss, 5),
                    take_profit=round(take_profit, 5),
                    reason=f"StatArb SELL {self.pair_b}: z={z_score:.2f}, spread compressed",
                    metadata={
                        "z_score": z_score,
                        "spread": float(spread_series.iloc[-1]),
                        "spread_mean": float(rolling_mean.iloc[-1]),
                        "spread_std": float(rolling_std.iloc[-1]),
                        "pair_role": "overpriced",
                        "hedge_pair": self.pair_a,
                        "hedge_side": "BUY",
                        "atr": atr_val,
                        "timeframe": timeframe,
                    },
                )

        if signal:
            self._in_position = True
            self._position_side = (
                "SHORT_A_LONG_B" if z_score > 0 else "LONG_A_SHORT_B"
            )
            self.trade_count += 1
            self.logger.info(f"Signal: {signal.side.value} {instrument} — {signal.reason}")

        return signal

    def should_close(self, instrument, timeframe, candles, entry_price, side):
        """Close based on z-score reversion or emergency divergence."""
        if timeframe != self.entry_timeframe:
            return None

        if instrument not in (self.pair_a, self.pair_b):
            return None

        # Update closes for this instrument
        if len(candles) < self.lookback_period + 5:
            return None

        df = candles_to_df(candles)
        if df.empty:
            return None

        closes = df["close"].tolist()
        self._pair_closes[instrument] = closes

        # Recompute z-score
        closes_a = self._pair_closes.get(self.pair_a, [])
        closes_b = self._pair_closes.get(self.pair_b, [])

        if not closes_a or not closes_b:
            return None

        min_len = min(len(closes_a), len(closes_b))
        if min_len < self.lookback_period + 5:
            return None

        arr_a = np.array(closes_a[-min_len:])
        arr_b = np.array(closes_b[-min_len:])

        with np.errstate(divide="ignore", invalid="ignore"):
            spread = arr_a / arr_b

        spread_series = pd.Series(spread)
        rolling_mean = spread_series.rolling(self.lookback_period).mean()
        rolling_std = spread_series.rolling(self.lookback_period).std()

        if rolling_std.iloc[-1] is None or rolling_std.iloc[-1] == 0:
            return None

        z_score = (spread_series.iloc[-1] - rolling_mean.iloc[-1]) / rolling_std.iloc[-1]

        if np.isnan(z_score) or np.isinf(z_score):
            return None

        self._last_z_score = z_score

        # Emergency exit: spread diverging further
        if abs(z_score) > self.emergency_z:
            self._in_position = False
            self._position_side = ""
            self.logger.warning(
                f"StatArb EMERGENCY EXIT {instrument}: z={z_score:.2f} > {self.emergency_z}"
            )
            return "stat_arb_emergency_exit"

        # Normal exit: spread has reverted
        if abs(z_score) < self.exit_z:
            self._in_position = False
            self._position_side = ""
            self.logger.info(
                f"StatArb PROFIT EXIT {instrument}: z={z_score:.2f} < {self.exit_z}"
            )
            return "stat_arb_reversion_exit"

        return None

    def get_required_history(self) -> int:
        """Need enough H1 candles for rolling window + buffer."""
        return self.lookback_period + 50
