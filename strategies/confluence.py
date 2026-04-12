"""Multi-Indicator Confluence Strategy — Score-based entry using 7 confirmations.

Logic:
- Uses a SCORING SYSTEM: each confirming indicator adds +1 to a score
- Only trade when score >= threshold (default 4 out of 7 confirmations)
- Confirmations for BUY:
  1. Price above EMA(200) — daily trend filter
  2. EMA(20) > EMA(50) — medium-term trend
  3. RSI(14) between 30-60 — not overbought, room to run
  4. MACD histogram positive and increasing
  5. ADX > 20 — trending market (not choppy)
  6. Price near lower Bollinger Band or recently bounced
  7. Volume above 20-period average (tick volume confirmation)
- Mirror for SELL signals
- Stop loss: 1.5x ATR(14)
- Take profit: 2.5x ATR(14)
- Timeframes: H1 for entry, D for trend confirmation

Best pairs: EUR_USD, GBP_USD (strong trending + reverting behavior)
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


class ConfluenceStrategy(Strategy):
    name = "confluence"
    timeframes = ["H1", "D"]

    # --- Tunable parameters (optimizer-friendly class attributes) ---

    # EMA periods
    ema_fast: int = 20
    ema_slow: int = 50
    ema_trend: int = 200  # daily trend filter

    # RSI
    rsi_period: int = 14
    rsi_buy_low: float = 30.0
    rsi_buy_high: float = 60.0
    rsi_sell_low: float = 40.0
    rsi_sell_high: float = 70.0

    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # ADX
    adx_period: int = 14
    adx_min: float = 20.0

    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0
    bb_proximity_pct: float = 0.02  # within 2% of band counts as "near"

    # Volume
    vol_avg_period: int = 20

    # ATR for SL/TP
    atr_period: int = 14
    sl_atr_multiplier: float = 1.5
    tp_atr_multiplier: float = 2.5

    # Score threshold — minimum confirmations to trade
    score_threshold: int = 4  # out of 7

    # Timeframes
    entry_timeframe: str = "H1"
    trend_timeframe: str = "D"

    # State: daily trend direction
    _daily_trend: dict[str, str] = {}

    def on_candle(
        self,
        instrument: str,
        timeframe: str,
        candles: list[Candle],
    ) -> Optional[Signal]:
        if ta is None:
            return None

        # Update daily trend when D candle arrives
        if timeframe == self.trend_timeframe:
            self._update_daily_trend(instrument, candles)
            return None

        # Only generate signals on entry timeframe
        if timeframe != self.entry_timeframe:
            return None

        min_periods = max(
            self.ema_trend, self.ema_slow, self.bb_period,
            self.macd_slow + self.macd_signal, self.adx_period,
            self.vol_avg_period, self.atr_period,
        ) + 10
        if len(candles) < min_periods:
            return None

        df = candles_to_df(candles)
        if df.empty:
            return None

        # --- Compute all indicators ---
        ema_fast = ta.ema(df["close"], length=self.ema_fast)
        ema_slow = ta.ema(df["close"], length=self.ema_slow)
        ema_trend = ta.ema(df["close"], length=self.ema_trend)
        rsi = ta.rsi(df["close"], length=self.rsi_period)
        macd_df = ta.macd(
            df["close"],
            fast=self.macd_fast,
            slow=self.macd_slow,
            signal=self.macd_signal,
        )
        adx_df = ta.adx(df["high"], df["low"], df["close"], length=self.adx_period)
        bb = ta.bbands(df["close"], length=self.bb_period, std=self.bb_std)
        atr = ta.atr(df["high"], df["low"], df["close"], length=self.atr_period)

        # Validate all indicators computed
        if any(x is None for x in [ema_fast, ema_slow, ema_trend, rsi, macd_df, adx_df, bb, atr]):
            return None

        # --- Latest values ---
        close = df["close"].iloc[-1]
        ema_f = ema_fast.iloc[-1]
        ema_s = ema_slow.iloc[-1]
        ema_t = ema_trend.iloc[-1]
        current_rsi = rsi.iloc[-1]
        current_atr = atr.iloc[-1]

        # ADX is first column of adx_df
        current_adx = adx_df.iloc[-1, 0] if not adx_df.empty else 0.0

        # MACD histogram is third column (MACDh_*)
        macd_hist = macd_df.iloc[-1, 2] if macd_df.shape[1] >= 3 else 0.0
        macd_hist_prev = macd_df.iloc[-2, 2] if macd_df.shape[1] >= 3 and len(macd_df) >= 2 else 0.0

        # Bollinger Band values: BBL, BBM, BBU
        lower_bb = bb.iloc[-1, 0]
        mid_bb = bb.iloc[-1, 1]
        upper_bb = bb.iloc[-1, 2]

        # Volume
        vol_avg = df["volume"].rolling(self.vol_avg_period).mean().iloc[-1]
        current_vol = df["volume"].iloc[-1]

        if current_atr <= 0:
            return None

        # --- Score BUY confirmations ---
        buy_score = 0
        buy_reasons = []

        # 1. Price above EMA(200) — daily trend
        if close > ema_t:
            buy_score += 1
            buy_reasons.append("above_EMA200")

        # 2. EMA(20) > EMA(50) — medium-term trend
        if ema_f > ema_s:
            buy_score += 1
            buy_reasons.append("EMA20>EMA50")

        # 3. RSI in buy zone (30-60)
        if self.rsi_buy_low <= current_rsi <= self.rsi_buy_high:
            buy_score += 1
            buy_reasons.append(f"RSI={current_rsi:.0f}")

        # 4. MACD histogram positive and increasing
        if macd_hist > 0 and macd_hist > macd_hist_prev:
            buy_score += 1
            buy_reasons.append("MACD+")

        # 5. ADX > 20 — trending market
        if current_adx > self.adx_min:
            buy_score += 1
            buy_reasons.append(f"ADX={current_adx:.0f}")

        # 6. Price near lower BB or recently bounced
        bb_range = upper_bb - lower_bb
        if bb_range > 0 and (close - lower_bb) / bb_range < self.bb_proximity_pct * 10:
            # In lower 20% of BB range
            buy_score += 1
            buy_reasons.append("near_BB_low")

        # 7. Volume above average
        if vol_avg > 0 and current_vol > vol_avg:
            buy_score += 1
            buy_reasons.append("vol_confirm")

        # --- Score SELL confirmations ---
        sell_score = 0
        sell_reasons = []

        # 1. Price below EMA(200)
        if close < ema_t:
            sell_score += 1
            sell_reasons.append("below_EMA200")

        # 2. EMA(20) < EMA(50)
        if ema_f < ema_s:
            sell_score += 1
            sell_reasons.append("EMA20<EMA50")

        # 3. RSI in sell zone (40-70)
        if self.rsi_sell_low <= current_rsi <= self.rsi_sell_high:
            sell_score += 1
            sell_reasons.append(f"RSI={current_rsi:.0f}")

        # 4. MACD histogram negative and decreasing
        if macd_hist < 0 and macd_hist < macd_hist_prev:
            sell_score += 1
            sell_reasons.append("MACD-")

        # 5. ADX > 20
        if current_adx > self.adx_min:
            sell_score += 1
            sell_reasons.append(f"ADX={current_adx:.0f}")

        # 6. Price near upper BB
        if bb_range > 0 and (upper_bb - close) / bb_range < self.bb_proximity_pct * 10:
            sell_score += 1
            sell_reasons.append("near_BB_high")

        # 7. Volume above average
        if vol_avg > 0 and current_vol > vol_avg:
            sell_score += 1
            sell_reasons.append("vol_confirm")

        # --- Daily trend filter ---
        daily_trend = self._daily_trend.get(instrument, "FLAT")

        signal = None

        # --- BUY signal ---
        if buy_score >= self.score_threshold and daily_trend in ("UP", "FLAT"):
            stop_loss = close - (current_atr * self.sl_atr_multiplier)
            take_profit = close + (current_atr * self.tp_atr_multiplier)

            risk = close - stop_loss
            reward = take_profit - close
            if risk > 0 and reward / risk >= 1.5:
                strength = SignalStrength.STRONG if buy_score >= 6 else SignalStrength.MODERATE
                signal = Signal(
                    strategy=self.name,
                    instrument=instrument,
                    side=Side.BUY,
                    strength=strength,
                    entry_price=close,
                    stop_loss=round(stop_loss, 5),
                    take_profit=round(take_profit, 5),
                    reason=f"Confluence BUY {buy_score}/7: {', '.join(buy_reasons)}",
                    metadata={
                        "score": buy_score,
                        "confirmations": buy_reasons,
                        "rsi": current_rsi,
                        "adx": current_adx,
                        "macd_hist": macd_hist,
                        "atr": current_atr,
                        "daily_trend": daily_trend,
                        "timeframe": timeframe,
                    },
                )

        # --- SELL signal ---
        elif sell_score >= self.score_threshold and daily_trend in ("DOWN", "FLAT"):
            stop_loss = close + (current_atr * self.sl_atr_multiplier)
            take_profit = close - (current_atr * self.tp_atr_multiplier)

            risk = stop_loss - close
            reward = close - take_profit
            if risk > 0 and reward / risk >= 1.5:
                strength = SignalStrength.STRONG if sell_score >= 6 else SignalStrength.MODERATE
                signal = Signal(
                    strategy=self.name,
                    instrument=instrument,
                    side=Side.SELL,
                    strength=strength,
                    entry_price=close,
                    stop_loss=round(stop_loss, 5),
                    take_profit=round(take_profit, 5),
                    reason=f"Confluence SELL {sell_score}/7: {', '.join(sell_reasons)}",
                    metadata={
                        "score": sell_score,
                        "confirmations": sell_reasons,
                        "rsi": current_rsi,
                        "adx": current_adx,
                        "macd_hist": macd_hist,
                        "atr": current_atr,
                        "daily_trend": daily_trend,
                        "timeframe": timeframe,
                    },
                )

        if signal:
            self.trade_count += 1
            self.logger.info(f"Signal: {signal.side.value} {instrument} — {signal.reason}")

        return signal

    def _update_daily_trend(self, instrument: str, candles: list[Candle]):
        """Determine daily trend using EMA(200)."""
        if len(candles) < self.ema_trend + 5:
            self._daily_trend[instrument] = "FLAT"
            return

        df = candles_to_df(candles)
        ema_200 = ta.ema(df["close"], length=self.ema_trend)
        if ema_200 is None:
            self._daily_trend[instrument] = "FLAT"
            return

        close = df["close"].iloc[-1]
        ema_val = ema_200.iloc[-1]
        threshold = ema_val * 0.005

        if close > ema_val + threshold:
            self._daily_trend[instrument] = "UP"
        elif close < ema_val - threshold:
            self._daily_trend[instrument] = "DOWN"
        else:
            self._daily_trend[instrument] = "FLAT"

        self.logger.debug(f"{instrument} daily trend: {self._daily_trend[instrument]}")

    def should_close(self, instrument, timeframe, candles, entry_price, side):
        """Close if score reverses to opposing threshold or MACD flips."""
        if timeframe != self.entry_timeframe or ta is None:
            return None

        min_periods = max(self.macd_slow + self.macd_signal, self.adx_period) + 10
        if len(candles) < min_periods:
            return None

        df = candles_to_df(candles)
        if df.empty:
            return None

        rsi = ta.rsi(df["close"], length=self.rsi_period)
        macd_df = ta.macd(
            df["close"],
            fast=self.macd_fast,
            slow=self.macd_slow,
            signal=self.macd_signal,
        )

        if rsi is None or macd_df is None:
            return None

        current_rsi = rsi.iloc[-1]
        macd_hist = macd_df.iloc[-1, 2] if macd_df.shape[1] >= 3 else 0.0

        # Exit long if RSI overbought and MACD histogram turns negative
        if side == "BUY" and current_rsi > 70 and macd_hist < 0:
            return "confluence_reversal_exit"

        # Exit short if RSI oversold and MACD histogram turns positive
        if side == "SELL" and current_rsi < 30 and macd_hist > 0:
            return "confluence_reversal_exit"

        return None

    def get_required_history(self) -> int:
        """Need enough candles for EMA(200) + buffer."""
        return self.ema_trend + 50
