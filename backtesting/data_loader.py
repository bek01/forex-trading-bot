"""Historical data loader for the new-architecture backtest.

Pulls daily OHLC from OANDA and caches to parquet/CSV so repeated backtest
runs don't hammer the API. One file per instrument.

Usage:
    loader = HistoricalDataLoader()
    df = loader.get_daily("EUR_USD", years=6)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from config import get_config
from execution.broker import OandaBroker
from models import Candle

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("/home/ubuntu/trading_bot/db/hist_cache")


class HistoricalDataLoader:
    """Pull + cache historical daily candles for backtesting."""

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._broker: Optional[OandaBroker] = None

    @property
    def broker(self) -> OandaBroker:
        if self._broker is None:
            cfg = get_config()
            self._broker = OandaBroker(cfg.broker)
        return self._broker

    def _cache_path(self, instrument: str, timeframe: str) -> Path:
        # CSV — no parquet/pyarrow dependency
        return self.cache_dir / f"{instrument}_{timeframe}.csv"

    def _candles_to_df(self, candles: list[Candle]) -> pd.DataFrame:
        rows = []
        for c in candles:
            if not c.complete:
                continue
            rows.append({
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        return df

    def fetch(
        self,
        instrument: str,
        timeframe: str = "D",
        count: int = 2000,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch historical candles. Uses parquet cache if fresh.

        OANDA caps `count` at 5000 per request. For >5000 bars, call multiple
        times paging by `to` parameter — not implemented here, 2000 daily bars
        = ~8 years which is enough for walk-forward on FX.
        """
        path = self._cache_path(instrument, timeframe)
        if use_cache and path.exists():
            try:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                df.index = pd.to_datetime(df.index, utc=True)
                age_hours = (datetime.now(timezone.utc) - df.index.max()).total_seconds() / 3600
                if age_hours < 24:
                    logger.info(f"Cache hit: {instrument}/{timeframe} ({len(df)} rows, age {age_hours:.1f}h)")
                    return df
                logger.info(f"Cache stale: {instrument}/{timeframe} ({age_hours:.1f}h)")
            except Exception as e:
                logger.warning(f"Cache read failed {path}: {e}")

        logger.info(f"Fetching {instrument}/{timeframe} count={count} from OANDA")
        candles = self.broker.get_candles(instrument, timeframe, count=count)
        df = self._candles_to_df(candles)
        if df.empty:
            logger.warning(f"No candles returned for {instrument}/{timeframe}")
            return df

        try:
            df.to_csv(path)
            logger.info(f"Cached {instrument}/{timeframe}: {len(df)} rows → {path}")
        except Exception as e:
            logger.warning(f"Cache write failed {path}: {e}")
        return df

    def fetch_universe(
        self,
        instruments: list[str],
        timeframe: str = "D",
        count: int = 2000,
    ) -> dict[str, pd.DataFrame]:
        """Bulk fetch for a list of instruments."""
        out = {}
        for inst in instruments:
            try:
                df = self.fetch(inst, timeframe, count)
                if not df.empty:
                    out[inst] = df
            except Exception as e:
                logger.error(f"Failed to fetch {inst}: {e}")
        return out
