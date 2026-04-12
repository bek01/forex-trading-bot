"""Base strategy class — all strategies must implement this interface.

Adding a new strategy takes <20 lines:

    class MyStrategy(Strategy):
        name = "my_strategy"
        timeframes = ["M15", "H1"]

        def on_candle(self, instrument, timeframe, candles):
            if timeframe != "M15":
                return None
            df = candles_to_df(candles)
            # Your logic here...
            if buy_condition:
                return Signal(
                    strategy=self.name,
                    instrument=instrument,
                    side=Side.BUY,
                    entry_price=df.close.iloc[-1],
                    stop_loss=df.close.iloc[-1] - atr * 1.5,
                    take_profit=df.close.iloc[-1] + atr * 2.5,
                )
            return None
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from models import Candle, Signal, Tick

logger = logging.getLogger(__name__)


def candles_to_df(candles: list[Candle]) -> pd.DataFrame:
    """Convert candle list to pandas DataFrame for indicator calculations."""
    if not candles:
        return pd.DataFrame()

    data = {
        "timestamp": [c.timestamp for c in candles],
        "open": [c.open for c in candles],
        "high": [c.high for c in candles],
        "low": [c.low for c in candles],
        "close": [c.close for c in candles],
        "volume": [c.volume for c in candles],
    }
    df = pd.DataFrame(data)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    return df


class Strategy(ABC):
    """Abstract base class for all trading strategies."""

    # Override these in subclass
    name: str = "base"
    timeframes: list[str] = ["H1"]  # Which timeframes this strategy needs
    instruments: list[str] = []  # Empty = all configured instruments

    def __init__(self):
        self.enabled = True
        self.trade_count = 0
        self.logger = logging.getLogger(f"strategy.{self.name}")

    @abstractmethod
    def on_candle(
        self,
        instrument: str,
        timeframe: str,
        candles: list[Candle],
    ) -> Optional[Signal]:
        """
        Called when a new candle closes.

        Args:
            instrument: The forex pair (e.g., "EUR_USD")
            timeframe: Candle timeframe (e.g., "M15", "H1")
            candles: List of recent candles (oldest first), up to buffer size

        Returns:
            Signal if the strategy wants to trade, None otherwise
        """
        pass

    def on_tick(self, tick: Tick) -> Optional[Signal]:
        """
        Called on every price tick (optional override).
        Most strategies should use on_candle instead — tick-based is for scalping.
        Default: do nothing.
        """
        return None

    def should_close(
        self,
        instrument: str,
        timeframe: str,
        candles: list[Candle],
        entry_price: float,
        side: str,
    ) -> Optional[str]:
        """
        Check if an existing position should be closed early (optional).
        Returns close reason string if yes, None if position should stay open.
        Stop loss and take profit are handled by the broker — this is for
        strategy-specific exits (e.g., indicator reversal).
        """
        return None

    def get_required_history(self) -> int:
        """How many candles of history this strategy needs to compute indicators."""
        return 200  # safe default for most indicators

    def __repr__(self):
        return f"{self.name}(enabled={self.enabled})"
