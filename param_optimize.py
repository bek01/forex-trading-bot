"""Fast parameter optimization — pre-computes indicators once per param combo."""
import sys, random, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import pandas_ta as pta

from execution.broker import OandaBroker
from config import get_config
from strategies.base import candles_to_df
from models import FOREX_PAIRS

cfg = get_config()
broker = OandaBroker(cfg.broker)

candles_m15 = [c for c in broker.get_candles('EUR_USD', 'M15', 5000) if c.complete]
candles_h1 = [c for c in broker.get_candles('EUR_USD', 'H1', 5000) if c.complete]
print(f'EUR/USD M15: {len(candles_m15)} candles, H1: {len(candles_h1)} candles')
broker.close()

pip_size = FOREX_PAIRS.get('EUR_USD', {}).get('pip_size', 0.0001)
SPREAD_PIPS = 1.5
SLIPPAGE_PIPS = 0.5
INITIAL_CAPITAL = 10000.0
RISK_PCT = 1.0
spread = SPREAD_PIPS * pip_size
slippage = SLIPPAGE_PIPS * pip_size


def compute_metrics(trades_pnl, equity_curve):
    """Compute Sharpe, PF, WR, etc. from trade PnL list."""
    if not trades_pnl:
        return 0.0, 0.0, 0.0, 0, 0.0, 0.0

    wins = [p for p in trades_pnl if p > 0]
    losses = [p for p in trades_pnl if p < 0]
    total = len(trades_pnl)
    wr = len(wins) / total * 100
    gross_win = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    pf = gross_win / gross_loss if gross_loss > 0 else 99.0
    ret = (equity_curve[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak * 100
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0

    # Sharpe from equity curve returns
    if len(equity_curve) > 2:
        returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
        std = np.std(returns)
        sharpe = float(np.mean(returns) / std * np.sqrt(252)) if std > 0 else 0.0
    else:
        sharpe = 0.0

    return sharpe, pf, wr, total, ret, max_dd


def simulate_trades(signals, df, initial_capital=INITIAL_CAPITAL):
    """
    Given a list of (index, side, sl, tp) signals and a DataFrame,
    simulate SL/TP execution and return metrics.
    side: 'BUY' or 'SELL'
    """
    equity = initial_capital
    equity_curve = [equity]
    trades_pnl = []
    in_trade = False
    entry_price = sl = tp = 0.0
    trade_side = ''
    units = 0.0

    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    n = len(closes)

    sig_idx = 0
    for i in range(n):
        if in_trade:
            # Check SL/TP
            hit = False
            if trade_side == 'BUY':
                if lows[i] <= sl:
                    pnl = (sl - entry_price) * units
                    hit = True
                elif tp and highs[i] >= tp:
                    pnl = (tp - entry_price) * units
                    hit = True
            else:  # SELL
                if highs[i] >= sl:
                    pnl = (entry_price - sl) * units
                    hit = True
                elif tp and lows[i] <= tp:
                    pnl = (entry_price - tp) * units
                    hit = True

            if hit:
                equity += pnl
                trades_pnl.append(pnl)
                equity_curve.append(equity)
                in_trade = False
                continue

        if not in_trade and sig_idx < len(signals):
            si, side, s_sl, s_tp = signals[sig_idx]
            if i >= si:
                sig_idx += 1
                # Position sizing
                risk_amount = equity * (RISK_PCT / 100.0)
                sl_dist = abs(closes[i] - s_sl)
                if sl_dist <= 0:
                    continue
                units = risk_amount / sl_dist
                entry_price = closes[i] + (spread / 2 + slippage if side == 'BUY' else -(spread / 2 + slippage))
                sl = s_sl
                tp = s_tp
                trade_side = side
                in_trade = True

        equity_curve.append(equity)

    # Close remaining position
    if in_trade:
        if trade_side == 'BUY':
            pnl = (closes[-1] - entry_price) * units
        else:
            pnl = (entry_price - closes[-1]) * units
        equity += pnl
        trades_pnl.append(pnl)
        equity_curve.append(equity)

    return compute_metrics(trades_pnl, equity_curve)


# ========== Pre-compute DataFrames ==========
df_m15 = candles_to_df(candles_m15)
df_h1 = candles_to_df(candles_h1)


# ========== MEAN REVERSION ==========
print('\n=== MEAN REVERSION ===')
best_mr = []

for i in range(20):
    bb_period = random.choice([15, 20, 25, 30])
    bb_std = random.choice([1.5, 2.0, 2.5])
    rsi_period = random.choice([10, 14, 21])
    rsi_oversold = random.choice([25, 30, 35])
    rsi_overbought = random.choice([65, 70, 75])
    adx_max = random.choice([20, 25, 30, 35])
    sl_atr_mult = random.choice([1.0, 1.5, 2.0, 2.5])
    atr_period = 14

    # Pre-compute indicators ONCE
    bb = pta.bbands(df_m15['close'], length=bb_period, std=bb_std)
    rsi = pta.rsi(df_m15['close'], length=rsi_period)
    atr = pta.atr(df_m15['high'], df_m15['low'], df_m15['close'], length=atr_period)
    adx_df = pta.adx(df_m15['high'], df_m15['low'], df_m15['close'], length=14)

    if bb is None or rsi is None or atr is None or adx_df is None:
        continue

    closes = df_m15['close'].values
    bbl = bb.iloc[:, 0].values  # lower
    bbm = bb.iloc[:, 1].values  # mid
    bbu = bb.iloc[:, 2].values  # upper
    rsi_v = rsi.values
    atr_v = atr.values
    adx_v = adx_df.iloc[:, 0].values

    signals = []
    start = max(bb_period, rsi_period, atr_period, 14) + 10

    for j in range(start, len(closes)):
        if np.isnan(adx_v[j]) or np.isnan(rsi_v[j]) or np.isnan(atr_v[j]) or np.isnan(bbl[j]):
            continue
        if adx_v[j] > adx_max or atr_v[j] <= 0:
            continue

        c = closes[j]
        # BUY: close <= lower BB and RSI < oversold
        if c <= bbl[j] and rsi_v[j] < rsi_oversold:
            sl = c - atr_v[j] * sl_atr_mult
            tp = bbm[j]
            risk = c - sl
            reward = tp - c
            if risk > 0 and reward / risk >= 1.5:
                signals.append((j, 'BUY', sl, tp))
        # SELL: close >= upper BB and RSI > overbought
        elif c >= bbu[j] and rsi_v[j] > rsi_overbought:
            sl = c + atr_v[j] * sl_atr_mult
            tp = bbm[j]
            risk = sl - c
            reward = c - tp
            if risk > 0 and reward / risk >= 1.5:
                signals.append((j, 'SELL', sl, tp))

    sharpe, pf, wr, trades, ret, dd = simulate_trades(signals, df_m15)
    params = f'BB({bb_period},{bb_std}) RSI({rsi_period}) OS={rsi_oversold} OB={rsi_overbought} ADX<{adx_max} SL={sl_atr_mult}xATR'
    best_mr.append((sharpe, pf, wr, trades, ret, dd, params))

best_mr.sort(reverse=True)
for sharpe, pf, wr, trades, ret, dd, params in best_mr[:5]:
    print(f'  Sharpe={sharpe:+.2f} PF={pf:.2f} WR={wr:.0f}% Trades={trades} Ret={ret:+.1f}% DD={dd:.1f}% | {params}')


# ========== TREND FOLLOWING ==========
print('\n=== TREND FOLLOWING ===')
best_tf = []

for i in range(20):
    fast_ema = random.choice([8, 10, 15, 20, 25])
    slow_ema = random.choice([30, 40, 50, 60, 80])
    adx_min = random.choice([15, 20, 25, 30])
    sl_atr_mult = random.choice([1.0, 1.5, 2.0, 2.5, 3.0])
    tp_atr_mult = random.choice([2.0, 2.5, 3.0, 4.0, 5.0])

    ema_f = pta.ema(df_h1['close'], length=fast_ema)
    ema_s = pta.ema(df_h1['close'], length=slow_ema)
    atr = pta.atr(df_h1['high'], df_h1['low'], df_h1['close'], length=14)
    adx_df = pta.adx(df_h1['high'], df_h1['low'], df_h1['close'], length=14)

    if ema_f is None or ema_s is None or atr is None or adx_df is None:
        continue

    closes = df_h1['close'].values
    ef = ema_f.values
    es = ema_s.values
    atr_v = atr.values
    adx_v = adx_df.iloc[:, 0].values

    signals = []
    start = slow_ema + 10

    for j in range(start, len(closes)):
        if np.isnan(ef[j]) or np.isnan(es[j]) or np.isnan(adx_v[j]) or np.isnan(atr_v[j]):
            continue
        if adx_v[j] < adx_min or atr_v[j] <= 0:
            continue

        # Crossover check
        bullish = ef[j - 1] <= es[j - 1] and ef[j] > es[j]
        bearish = ef[j - 1] >= es[j - 1] and ef[j] < es[j]

        if bullish:
            sl = closes[j] - atr_v[j] * sl_atr_mult
            tp = closes[j] + atr_v[j] * tp_atr_mult
            signals.append((j, 'BUY', sl, tp))
        elif bearish:
            sl = closes[j] + atr_v[j] * sl_atr_mult
            tp = closes[j] - atr_v[j] * tp_atr_mult
            signals.append((j, 'SELL', sl, tp))

    sharpe, pf, wr, trades, ret, dd = simulate_trades(signals, df_h1)
    params = f'EMA({fast_ema}/{slow_ema}) ADX>{adx_min} SL={sl_atr_mult}xATR TP={tp_atr_mult}xATR'
    best_tf.append((sharpe, pf, wr, trades, ret, dd, params))

best_tf.sort(reverse=True)
for sharpe, pf, wr, trades, ret, dd, params in best_tf[:5]:
    print(f'  Sharpe={sharpe:+.2f} PF={pf:.2f} WR={wr:.0f}% Trades={trades} Ret={ret:+.1f}% DD={dd:.1f}% | {params}')


# ========== CONFLUENCE ==========
print('\n=== CONFLUENCE ===')
best_conf = []

for i in range(20):
    score_threshold = random.choice([3, 4, 5])
    trend_ema_period = random.choice([100, 150, 200])
    fast_ema_period = random.choice([10, 15, 20])
    slow_ema_period = random.choice([40, 50, 60])
    rsi_period = random.choice([10, 14, 21])
    adx_min = random.choice([15, 20, 25])
    sl_atr_mult = random.choice([1.0, 1.5, 2.0])
    tp_atr_mult = random.choice([2.0, 2.5, 3.0, 4.0])

    ema_f = pta.ema(df_h1['close'], length=fast_ema_period)
    ema_s = pta.ema(df_h1['close'], length=slow_ema_period)
    ema_t = pta.ema(df_h1['close'], length=trend_ema_period)
    rsi = pta.rsi(df_h1['close'], length=rsi_period)
    macd_df = pta.macd(df_h1['close'], fast=12, slow=26, signal=9)
    adx_df = pta.adx(df_h1['high'], df_h1['low'], df_h1['close'], length=14)
    bb = pta.bbands(df_h1['close'], length=20, std=2.0)
    atr = pta.atr(df_h1['high'], df_h1['low'], df_h1['close'], length=14)
    vol_avg = df_h1['volume'].rolling(20).mean()

    if any(x is None for x in [ema_f, ema_s, ema_t, rsi, macd_df, adx_df, bb, atr]):
        continue

    closes = df_h1['close'].values
    volumes = df_h1['volume'].values
    ef = ema_f.values
    es = ema_s.values
    et = ema_t.values
    rsi_v = rsi.values
    macd_hist = macd_df.iloc[:, 2].values  # histogram
    adx_v = adx_df.iloc[:, 0].values
    bbl = bb.iloc[:, 0].values
    bbu = bb.iloc[:, 2].values
    atr_v = atr.values
    vol_avg_v = vol_avg.values

    signals = []
    start = trend_ema_period + 10

    for j in range(start, len(closes)):
        if np.isnan(et[j]) or np.isnan(atr_v[j]) or atr_v[j] <= 0:
            continue

        c = closes[j]
        bb_range = bbu[j] - bbl[j] if not np.isnan(bbu[j]) and not np.isnan(bbl[j]) else 0

        # BUY score
        buy_score = 0
        if c > et[j]: buy_score += 1
        if not np.isnan(ef[j]) and not np.isnan(es[j]) and ef[j] > es[j]: buy_score += 1
        if not np.isnan(rsi_v[j]) and 30 <= rsi_v[j] <= 60: buy_score += 1
        if not np.isnan(macd_hist[j]) and j > 0 and not np.isnan(macd_hist[j-1]) and macd_hist[j] > 0 and macd_hist[j] > macd_hist[j-1]: buy_score += 1
        if not np.isnan(adx_v[j]) and adx_v[j] > adx_min: buy_score += 1
        if bb_range > 0 and (c - bbl[j]) / bb_range < 0.2: buy_score += 1
        if not np.isnan(vol_avg_v[j]) and vol_avg_v[j] > 0 and volumes[j] > vol_avg_v[j]: buy_score += 1

        # SELL score
        sell_score = 0
        if c < et[j]: sell_score += 1
        if not np.isnan(ef[j]) and not np.isnan(es[j]) and ef[j] < es[j]: sell_score += 1
        if not np.isnan(rsi_v[j]) and 40 <= rsi_v[j] <= 70: sell_score += 1
        if not np.isnan(macd_hist[j]) and j > 0 and not np.isnan(macd_hist[j-1]) and macd_hist[j] < 0 and macd_hist[j] < macd_hist[j-1]: sell_score += 1
        if not np.isnan(adx_v[j]) and adx_v[j] > adx_min: sell_score += 1
        if bb_range > 0 and (bbu[j] - c) / bb_range < 0.2: sell_score += 1
        if not np.isnan(vol_avg_v[j]) and vol_avg_v[j] > 0 and volumes[j] > vol_avg_v[j]: sell_score += 1

        if buy_score >= score_threshold:
            sl = c - atr_v[j] * sl_atr_mult
            tp = c + atr_v[j] * tp_atr_mult
            risk = c - sl
            reward = tp - c
            if risk > 0 and reward / risk >= 1.5:
                signals.append((j, 'BUY', sl, tp))
        elif sell_score >= score_threshold:
            sl = c + atr_v[j] * sl_atr_mult
            tp = c - atr_v[j] * tp_atr_mult
            risk = sl - c
            reward = c - tp
            if risk > 0 and reward / risk >= 1.5:
                signals.append((j, 'SELL', sl, tp))

    sharpe, pf, wr, trades, ret, dd = simulate_trades(signals, df_h1)
    params = f'Score>={score_threshold} EMA({fast_ema_period}/{slow_ema_period}/{trend_ema_period}) RSI({rsi_period}) ADX>{adx_min} SL={sl_atr_mult} TP={tp_atr_mult}'
    best_conf.append((sharpe, pf, wr, trades, ret, dd, params))

best_conf.sort(reverse=True)
for sharpe, pf, wr, trades, ret, dd, params in best_conf[:5]:
    print(f'  Sharpe={sharpe:+.2f} PF={pf:.2f} WR={wr:.0f}% Trades={trades} Ret={ret:+.1f}% DD={dd:.1f}% | {params}')

print('\nDone!')
