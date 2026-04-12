"""London Breakout Strategy — Session open range breakout.

Logic:
- Define range: first 30 minutes of London session (07:00-07:30 UTC)
- Entry BUY: price breaks above range high
- Entry SELL: price breaks below range low
- Stop loss: opposite side of range
- Take profit: 1.5x range size from entry
- Filter: range must be > 15 pips and < 50 pips (avoid too tight or too wide)
- Max 1 trade per session

Best pairs: GBP_USD, EUR_USD, EUR_GBP (London-active pairs)
Time: only active 07:00-12:00 UTC
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from models import Candle, Signal, Side, SignalStrength, FOREX_PAIRS
from strategies.base import Strategy, candles_to_df


class LondonBreakoutStrategy(Strategy):
    name = "london_breakout"
    timeframes = ["M5"]
    instruments = ["EUR_USD", "GBP_USD", "EUR_GBP"]

    # Parameters
    range_start_hour: int = 7   # UTC
    range_start_minute: int = 0
    range_end_hour: int = 7
    range_end_minute: int = 30
    trade_end_hour: int = 12    # Stop looking for breakouts after noon UTC
    min_range_pips: float = 15.0
    max_range_pips: float = 50.0
    tp_range_multiplier: float = 1.5
    primary_timeframe: str = "M5"

    # State: track range and whether we've traded this session
    _session_ranges: dict[str, dict] = {}  # instrument -> {high, low, traded}
    _last_range_date: str = ""

    def on_candle(
        self,
        instrument: str,
        timeframe: str,
        candles: list[Candle],
    ) -> Optional[Signal]:
        if timeframe != self.primary_timeframe:
            return None

        if not candles:
            return None

        now = datetime.now(timezone.utc)

        # Reset ranges at start of each day
        today = now.strftime("%Y-%m-%d")
        if today != self._last_range_date:
            self._session_ranges.clear()
            self._last_range_date = today

        pip_size = FOREX_PAIRS.get(instrument, {}).get("pip_size", 0.0001)

        # --- Phase 1: Build the range (07:00-07:30 UTC) ---
        if (now.hour == self.range_start_hour and
                now.minute < self.range_end_minute):
            self._build_range(instrument, candles, now)
            return None

        # --- Phase 2: Look for breakout (07:30-12:00 UTC) ---
        if now.hour < self.range_end_hour or now.hour >= self.trade_end_hour:
            return None

        rng = self._session_ranges.get(instrument)
        if not rng or rng.get("traded"):
            return None

        range_high = rng["high"]
        range_low = rng["low"]
        range_size = range_high - range_low
        range_pips = range_size / pip_size

        # Filter: range must be reasonable
        if range_pips < self.min_range_pips or range_pips > self.max_range_pips:
            return None

        close = candles[-1].close
        signal = None

        # Breakout above range
        if close > range_high:
            stop_loss = range_low  # opposite side of range
            take_profit = close + (range_size * self.tp_range_multiplier)

            signal = Signal(
                strategy=self.name,
                instrument=instrument,
                side=Side.BUY,
                strength=SignalStrength.MODERATE,
                entry_price=close,
                stop_loss=round(stop_loss, 5),
                take_profit=round(take_profit, 5),
                reason=f"London breakout BUY: range={range_pips:.0f}p, break above {range_high:.5f}",
                metadata={
                    "range_high": range_high,
                    "range_low": range_low,
                    "range_pips": range_pips,
                    "timeframe": timeframe,
                },
            )

        # Breakout below range
        elif close < range_low:
            stop_loss = range_high
            take_profit = close - (range_size * self.tp_range_multiplier)

            signal = Signal(
                strategy=self.name,
                instrument=instrument,
                side=Side.SELL,
                strength=SignalStrength.MODERATE,
                entry_price=close,
                stop_loss=round(stop_loss, 5),
                take_profit=round(take_profit, 5),
                reason=f"London breakout SELL: range={range_pips:.0f}p, break below {range_low:.5f}",
                metadata={
                    "range_high": range_high,
                    "range_low": range_low,
                    "range_pips": range_pips,
                    "timeframe": timeframe,
                },
            )

        if signal:
            rng["traded"] = True  # max 1 trade per session
            self.trade_count += 1
            self.logger.info(f"Signal: {signal.side.value} {instrument} — {signal.reason}")

        return signal

    def _build_range(self, instrument: str, candles: list[Candle], now: datetime):
        """Build the session opening range from M5 candles."""
        # Get candles within the range window
        range_candles = []
        for c in candles:
            if (c.timestamp.hour == self.range_start_hour and
                    c.timestamp.minute >= self.range_start_minute and
                    c.timestamp.minute < self.range_end_minute and
                    c.timestamp.strftime("%Y-%m-%d") == now.strftime("%Y-%m-%d")):
                range_candles.append(c)

        if not range_candles:
            return

        high = max(c.high for c in range_candles)
        low = min(c.low for c in range_candles)

        self._session_ranges[instrument] = {
            "high": high,
            "low": low,
            "traded": False,
            "candle_count": len(range_candles),
        }
        self.logger.debug(
            f"{instrument} range: {low:.5f} - {high:.5f} "
            f"({len(range_candles)} candles)"
        )
