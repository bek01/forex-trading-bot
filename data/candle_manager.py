"""Candle manager — maintains rolling buffers of candle data per instrument/timeframe.

Responsibilities:
1. Fetch initial historical candles on startup
2. Poll for new candles periodically
3. Emit CANDLE_CLOSE events when new complete candles arrive
4. Provide candle access to strategies
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from execution.broker import OandaBroker
from event_bus import Event, bus
from models import Candle

logger = logging.getLogger(__name__)

# Circuit breaker: after this many consecutive poll failures on one
# instrument×timeframe, back off polling that endpoint for the cooldown
# window. Prevents log spam and pointless API traffic during broker outages.
_CB_TRIP_THRESHOLD = 3
_CB_COOLDOWN_SEC = 60.0


class CandleManager:
    """Manages candle buffers for all instruments and timeframes."""

    def __init__(
        self,
        broker: OandaBroker,
        instruments: list[str],
        timeframes: list[str],
        buffer_size: int = 500,
    ):
        self.broker = broker
        self.instruments = instruments
        self.timeframes = timeframes
        self.buffer_size = buffer_size

        # Candle storage: {instrument: {timeframe: [candles]}}
        self._candles: dict[str, dict[str, list[Candle]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # Track last seen candle timestamp to detect new candles
        self._last_timestamps: dict[str, dict[str, Optional[datetime]]] = defaultdict(
            lambda: defaultdict(lambda: None)
        )
        # Per-endpoint circuit breaker: consecutive failure count and
        # cooldown timestamp. Keyed by (instrument, timeframe).
        self._fail_count: dict[tuple, int] = defaultdict(int)
        self._cooldown_until: dict[tuple, float] = {}
        self._initialized = False

    def initialize(self):
        """Fetch initial historical candles for all instrument/timeframe combos."""
        logger.info(
            f"Loading candle history: {len(self.instruments)} instruments × "
            f"{len(self.timeframes)} timeframes"
        )

        for instrument in self.instruments:
            for timeframe in self.timeframes:
                try:
                    candles = self.broker.get_candles(
                        instrument, timeframe, count=self.buffer_size
                    )
                    # Only keep complete candles for historical data
                    complete = [c for c in candles if c.complete]
                    self._candles[instrument][timeframe] = complete[-self.buffer_size:]

                    if complete:
                        self._last_timestamps[instrument][timeframe] = complete[-1].timestamp

                    logger.info(
                        f"  {instrument}/{timeframe}: {len(complete)} candles loaded"
                    )
                except Exception as e:
                    logger.error(f"Failed to load {instrument}/{timeframe}: {e}")

        self._initialized = True
        logger.info("Candle history loaded")

    def poll(self):
        """
        Poll for new candles. Call this periodically (every 5-10 seconds).
        Emits CANDLE_CLOSE for each new complete candle.
        """
        if not self._initialized:
            return

        now = time.monotonic()
        for instrument in self.instruments:
            for timeframe in self.timeframes:
                key = (instrument, timeframe)
                # Skip if endpoint is in cooldown from prior failures
                cd_until = self._cooldown_until.get(key)
                if cd_until and now < cd_until:
                    continue
                try:
                    self._poll_one(instrument, timeframe)
                    # Success — reset failure tracking
                    if self._fail_count.get(key):
                        logger.info(
                            f"Candle poll recovered {instrument}/{timeframe} "
                            f"after {self._fail_count[key]} failure(s)"
                        )
                    self._fail_count[key] = 0
                    self._cooldown_until.pop(key, None)
                except Exception as e:
                    self._fail_count[key] += 1
                    count = self._fail_count[key]
                    if count >= _CB_TRIP_THRESHOLD:
                        self._cooldown_until[key] = now + _CB_COOLDOWN_SEC
                        logger.warning(
                            f"Candle poll circuit-breaker TRIPPED "
                            f"{instrument}/{timeframe}: {count} consecutive "
                            f"failures, cooldown {int(_CB_COOLDOWN_SEC)}s. "
                            f"Last error: {e}"
                        )
                    else:
                        logger.error(
                            f"Candle poll error {instrument}/{timeframe} "
                            f"(fail #{count}): {e}"
                        )

    def _poll_one(self, instrument: str, timeframe: str):
        """Poll one instrument/timeframe for new candles."""
        # Fetch last few candles (2 is enough — current incomplete + last complete)
        candles = self.broker.get_candles(instrument, timeframe, count=5)
        if not candles:
            return

        last_ts = self._last_timestamps[instrument][timeframe]

        for candle in candles:
            if not candle.complete:
                continue
            if last_ts and candle.timestamp <= last_ts:
                continue

            # New complete candle!
            buffer = self._candles[instrument][timeframe]
            buffer.append(candle)

            # Trim buffer
            if len(buffer) > self.buffer_size:
                self._candles[instrument][timeframe] = buffer[-self.buffer_size:]

            self._last_timestamps[instrument][timeframe] = candle.timestamp

            # Emit event
            bus.emit(Event.CANDLE_CLOSE, {
                "instrument": instrument,
                "timeframe": timeframe,
                "candle": candle,
            })

    def get_candles(self, instrument: str, timeframe: str) -> list[Candle]:
        """Get candle buffer for an instrument/timeframe."""
        return self._candles.get(instrument, {}).get(timeframe, [])

    def get_latest(self, instrument: str, timeframe: str) -> Optional[Candle]:
        """Get the most recent complete candle."""
        candles = self.get_candles(instrument, timeframe)
        return candles[-1] if candles else None

    def get_stats(self) -> dict:
        """Return buffer sizes for monitoring."""
        stats = {}
        for inst in self.instruments:
            for tf in self.timeframes:
                count = len(self._candles.get(inst, {}).get(tf, []))
                if count > 0:
                    stats[f"{inst}/{tf}"] = count
        return stats
