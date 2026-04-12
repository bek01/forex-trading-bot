"""Session Momentum Strategy — Exploits London/NY session volatility patterns.

Logic:
- Track Asian session range (21:00-07:00 UTC): high and low
- At London open (07:00 UTC), wait for breakout + momentum confirmation:
  - Price breaks Asian high/low by > 0.3x ATR
  - RSI(14) confirms direction (>55 for long, <45 for short)
  - ADX(14) > 20 (momentum building)
- Second entry window at NY open (13:00 UTC):
  - If London established a trend, look for continuation pullbacks
  - Price retraces to EMA(20), then resumes in trend direction
- Stop loss: opposite side of Asian range OR 2x ATR, whichever is tighter
- Take profit: 2x risk (2:1 R:R minimum)
- Max 2 trades per day (1 London, 1 NY)
- Pairs: EUR_USD, GBP_USD only

Best timeframe: M15 for entries
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


class SessionMomentumStrategy(Strategy):
    name = "session_momentum"
    timeframes = ["M15"]
    instruments = ["EUR_USD", "GBP_USD"]

    # --- Tunable parameters ---

    # Session times (UTC hours)
    asian_start_hour: int = 21  # previous day 21:00
    asian_end_hour: int = 7    # 07:00 (London open)
    london_open_hour: int = 7
    london_window_end_hour: int = 10  # entry window: 07:00-10:00
    ny_open_hour: int = 13
    ny_window_end_hour: int = 16  # entry window: 13:00-16:00

    # Breakout parameters
    atr_period: int = 14
    breakout_atr_multiplier: float = 0.3  # break by > 0.3x ATR

    # RSI confirmation
    rsi_period: int = 14
    rsi_long_min: float = 55.0
    rsi_short_max: float = 45.0

    # ADX filter
    adx_period: int = 14
    adx_min: float = 20.0

    # EMA for NY pullback
    ema_period: int = 20
    pullback_tolerance_atr: float = 0.5  # within 0.5x ATR of EMA

    # Risk management
    sl_atr_multiplier: float = 2.0  # ATR-based SL fallback
    min_rr_ratio: float = 2.0      # minimum reward:risk

    # Max trades per day
    max_london_trades: int = 1
    max_ny_trades: int = 1

    # State tracking
    _asian_ranges: dict[str, dict] = {}  # instrument -> {high, low, date}
    _london_trend: dict[str, str] = {}   # instrument -> "UP" / "DOWN" / None
    _daily_trades: dict[str, dict] = {}  # instrument -> {london: 0, ny: 0, date: ""}

    def __init__(self):
        super().__init__()
        self._asian_ranges = {}
        self._london_trend = {}
        self._daily_trades = {}

    def on_candle(
        self,
        instrument: str,
        timeframe: str,
        candles: list[Candle],
    ) -> Optional[Signal]:
        if timeframe != "M15":
            return None

        if ta is None:
            return None

        min_periods = max(self.atr_period, self.rsi_period, self.adx_period, self.ema_period) + 10
        if len(candles) < min_periods:
            return None

        df = candles_to_df(candles)
        if df.empty:
            return None

        # Current candle timestamp
        last_candle = candles[-1]
        now = last_candle.timestamp
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        current_hour = now.hour
        today_str = now.strftime("%Y-%m-%d")

        # Reset daily trade counts
        trade_state = self._daily_trades.get(instrument, {"london": 0, "ny": 0, "date": ""})
        if trade_state["date"] != today_str:
            trade_state = {"london": 0, "ny": 0, "date": today_str}
            self._daily_trades[instrument] = trade_state
            self._london_trend[instrument] = ""

        # --- Update Asian session range ---
        self._update_asian_range(instrument, candles, now)

        # --- Calculate indicators ---
        atr = ta.atr(df["high"], df["low"], df["close"], length=self.atr_period)
        rsi = ta.rsi(df["close"], length=self.rsi_period)
        adx_df = ta.adx(df["high"], df["low"], df["close"], length=self.adx_period)
        ema = ta.ema(df["close"], length=self.ema_period)

        if any(x is None for x in [atr, rsi, adx_df, ema]):
            return None

        close = df["close"].iloc[-1]
        current_atr = atr.iloc[-1]
        current_rsi = rsi.iloc[-1]
        current_adx = adx_df.iloc[-1, 0] if not adx_df.empty else 0.0
        current_ema = ema.iloc[-1]

        if current_atr <= 0:
            return None

        asian = self._asian_ranges.get(instrument)
        if not asian or asian.get("date") != today_str:
            return None

        asian_high = asian["high"]
        asian_low = asian["low"]
        asian_range = asian_high - asian_low

        if asian_range <= 0:
            return None

        signal = None

        # --- London open window (07:00-10:00 UTC) ---
        if self.london_open_hour <= current_hour < self.london_window_end_hour:
            if trade_state["london"] >= self.max_london_trades:
                return None

            signal = self._check_breakout(
                instrument, close, asian_high, asian_low,
                current_atr, current_rsi, current_adx, "london",
            )
            if signal:
                trade_state["london"] += 1
                # Record London trend direction for NY continuation
                self._london_trend[instrument] = signal.side.value

        # --- NY open window (13:00-16:00 UTC) ---
        elif self.ny_open_hour <= current_hour < self.ny_window_end_hour:
            if trade_state["ny"] >= self.max_ny_trades:
                return None

            london_dir = self._london_trend.get(instrument, "")
            if london_dir:
                signal = self._check_ny_continuation(
                    instrument, close, london_dir,
                    current_atr, current_rsi, current_adx, current_ema,
                    asian_high, asian_low,
                )
                if signal:
                    trade_state["ny"] += 1

        if signal:
            self.trade_count += 1
            self.logger.info(f"Signal: {signal.side.value} {instrument} — {signal.reason}")

        return signal

    def _update_asian_range(
        self, instrument: str, candles: list[Candle], now: datetime
    ):
        """Track the Asian session (21:00-07:00 UTC) high/low."""
        today_str = now.strftime("%Y-%m-%d")

        # Only rebuild if we don't have today's range yet, or during Asian session
        existing = self._asian_ranges.get(instrument)
        if existing and existing.get("date") == today_str and now.hour >= self.asian_end_hour:
            return  # Already have today's finalized range

        high = float("-inf")
        low = float("inf")
        found = False

        for c in candles:
            ts = c.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            # Asian session: previous day 21:00 to today 07:00
            is_prev_evening = (
                ts.strftime("%Y-%m-%d") == (now - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                and ts.hour >= self.asian_start_hour
            )
            is_early_morning = (
                ts.strftime("%Y-%m-%d") == today_str
                and ts.hour < self.asian_end_hour
            )

            if is_prev_evening or is_early_morning:
                high = max(high, c.high)
                low = min(low, c.low)
                found = True

        if found and high > low:
            self._asian_ranges[instrument] = {
                "high": high,
                "low": low,
                "date": today_str,
            }

    def _check_breakout(
        self,
        instrument: str,
        close: float,
        asian_high: float,
        asian_low: float,
        atr: float,
        rsi: float,
        adx: float,
        session: str,
    ) -> Optional[Signal]:
        """Check for Asian range breakout with momentum confirmation."""
        breakout_distance = atr * self.breakout_atr_multiplier

        signal = None

        # Bullish breakout above Asian high
        if close > asian_high + breakout_distance:
            if rsi >= self.rsi_long_min and adx >= self.adx_min:
                # SL: Asian low or 2x ATR, whichever is tighter (closer to entry)
                sl_asian = asian_low
                sl_atr = close - (atr * self.sl_atr_multiplier)
                stop_loss = max(sl_asian, sl_atr)  # tighter = higher value for long

                risk = close - stop_loss
                if risk <= 0:
                    return None
                take_profit = close + (risk * self.min_rr_ratio)

                strength = SignalStrength.STRONG if adx > 30 else SignalStrength.MODERATE
                signal = Signal(
                    strategy=self.name,
                    instrument=instrument,
                    side=Side.BUY,
                    strength=strength,
                    entry_price=close,
                    stop_loss=round(stop_loss, 5),
                    take_profit=round(take_profit, 5),
                    reason=(
                        f"{session.title()} breakout BUY: "
                        f"RSI={rsi:.0f}, ADX={adx:.0f}, "
                        f"Asian H/L={asian_high:.5f}/{asian_low:.5f}"
                    ),
                    metadata={
                        "session": session,
                        "rsi": rsi,
                        "adx": adx,
                        "atr": atr,
                        "asian_high": asian_high,
                        "asian_low": asian_low,
                        "breakout_distance": close - asian_high,
                    },
                )

        # Bearish breakout below Asian low
        elif close < asian_low - breakout_distance:
            if rsi <= self.rsi_short_max and adx >= self.adx_min:
                sl_asian = asian_high
                sl_atr = close + (atr * self.sl_atr_multiplier)
                stop_loss = min(sl_asian, sl_atr)  # tighter = lower value for short

                risk = stop_loss - close
                if risk <= 0:
                    return None
                take_profit = close - (risk * self.min_rr_ratio)

                strength = SignalStrength.STRONG if adx > 30 else SignalStrength.MODERATE
                signal = Signal(
                    strategy=self.name,
                    instrument=instrument,
                    side=Side.SELL,
                    strength=strength,
                    entry_price=close,
                    stop_loss=round(stop_loss, 5),
                    take_profit=round(take_profit, 5),
                    reason=(
                        f"{session.title()} breakout SELL: "
                        f"RSI={rsi:.0f}, ADX={adx:.0f}, "
                        f"Asian H/L={asian_high:.5f}/{asian_low:.5f}"
                    ),
                    metadata={
                        "session": session,
                        "rsi": rsi,
                        "adx": adx,
                        "atr": atr,
                        "asian_high": asian_high,
                        "asian_low": asian_low,
                        "breakout_distance": asian_low - close,
                    },
                )

        return signal

    def _check_ny_continuation(
        self,
        instrument: str,
        close: float,
        london_dir: str,
        atr: float,
        rsi: float,
        adx: float,
        ema: float,
        asian_high: float,
        asian_low: float,
    ) -> Optional[Signal]:
        """Check for NY continuation of London trend via EMA pullback."""
        # Price must be near EMA (pulled back)
        distance_to_ema = abs(close - ema)
        if distance_to_ema > atr * self.pullback_tolerance_atr:
            return None  # not a pullback — too far from EMA

        # ADX must still show trend
        if adx < self.adx_min:
            return None

        signal = None

        if london_dir == "BUY" and close > ema and rsi >= self.rsi_long_min:
            # Pullback to EMA in an uptrend, resuming
            sl_atr = close - (atr * self.sl_atr_multiplier)
            stop_loss = max(asian_low, sl_atr)
            risk = close - stop_loss
            if risk <= 0:
                return None
            take_profit = close + (risk * self.min_rr_ratio)

            signal = Signal(
                strategy=self.name,
                instrument=instrument,
                side=Side.BUY,
                strength=SignalStrength.MODERATE,
                entry_price=close,
                stop_loss=round(stop_loss, 5),
                take_profit=round(take_profit, 5),
                reason=f"NY continuation BUY: pullback to EMA, RSI={rsi:.0f}, ADX={adx:.0f}",
                metadata={
                    "session": "ny",
                    "rsi": rsi,
                    "adx": adx,
                    "atr": atr,
                    "ema": ema,
                    "london_trend": london_dir,
                },
            )

        elif london_dir == "SELL" and close < ema and rsi <= self.rsi_short_max:
            sl_atr = close + (atr * self.sl_atr_multiplier)
            stop_loss = min(asian_high, sl_atr)
            risk = stop_loss - close
            if risk <= 0:
                return None
            take_profit = close - (risk * self.min_rr_ratio)

            signal = Signal(
                strategy=self.name,
                instrument=instrument,
                side=Side.SELL,
                strength=SignalStrength.MODERATE,
                entry_price=close,
                stop_loss=round(stop_loss, 5),
                take_profit=round(take_profit, 5),
                reason=f"NY continuation SELL: pullback to EMA, RSI={rsi:.0f}, ADX={adx:.0f}",
                metadata={
                    "session": "ny",
                    "rsi": rsi,
                    "adx": adx,
                    "atr": atr,
                    "ema": ema,
                    "london_trend": london_dir,
                },
            )

        return signal

    def should_close(self, instrument, timeframe, candles, entry_price, side):
        """Close if price reverses back inside Asian range (failed breakout)."""
        if timeframe != "M15" or ta is None:
            return None

        if len(candles) < self.rsi_period + 5:
            return None

        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        asian = self._asian_ranges.get(instrument)
        if not asian or asian.get("date") != today_str:
            return None

        df = candles_to_df(candles)
        close = df["close"].iloc[-1]

        # If long and price falls back below Asian high — failed breakout
        if side == "BUY" and close < asian["low"]:
            return "failed_breakout_reversal"

        # If short and price rises back above Asian low — failed breakout
        if side == "SELL" and close > asian["high"]:
            return "failed_breakout_reversal"

        # End of day exit: close before Asian session starts (21:00 UTC)
        now = datetime.now(timezone.utc)
        if now.hour >= 20:
            return "end_of_day_exit"

        return None

    def get_required_history(self) -> int:
        """Need enough M15 candles to cover Asian session + indicators."""
        # Asian session is ~10 hours = 40 M15 candles, plus indicator warmup
        return 200
