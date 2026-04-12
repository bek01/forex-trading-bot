#!/usr/bin/env python3
"""
Fast vectorized parameter optimizer.

The event-driven backtest engine is too slow (~2.5min/combo on 5000 candles).
This optimizer pre-computes ALL indicators once as numpy arrays, then tests
signal logic vectorized — ~100x faster.

Usage:
    python -m backtesting.fast_optimize
"""

import sys
import random
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pandas_ta as ta
except ImportError:
    ta = None

from execution.broker import OandaBroker
from config import get_config


def download_data(broker, pair, timeframe, count=5000):
    """Download candle data and return as DataFrame."""
    candles = broker.get_candles(pair, timeframe, count)
    complete = [c for c in candles if c.complete]
    df = pd.DataFrame({
        "open": [c.open for c in complete],
        "high": [c.high for c in complete],
        "low": [c.low for c in complete],
        "close": [c.close for c in complete],
        "volume": [c.volume for c in complete],
    })
    return df


def precompute_indicators(df):
    """Compute ALL possible indicators once. Returns dict of arrays."""
    ind = {}
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]

    # EMAs
    for p in [8, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200, 250]:
        ind[f"ema_{p}"] = ta.ema(close, length=p)

    # RSI
    for p in [7, 10, 14, 21]:
        ind[f"rsi_{p}"] = ta.rsi(close, length=p)

    # Bollinger Bands
    for period in [15, 20, 25, 30]:
        for std in [1.5, 2.0, 2.5, 3.0]:
            bb = ta.bbands(close, length=period, std=std)
            if bb is not None and not bb.empty:
                ind[f"bbl_{period}_{std}"] = bb.iloc[:, 0]  # lower
                ind[f"bbm_{period}_{std}"] = bb.iloc[:, 1]  # mid
                ind[f"bbu_{period}_{std}"] = bb.iloc[:, 2]  # upper

    # ATR
    for p in [10, 14, 20]:
        ind[f"atr_{p}"] = ta.atr(high, low, close, length=p)

    # ADX
    for p in [10, 14, 20]:
        adx_df = ta.adx(high, low, close, length=p)
        if adx_df is not None and not adx_df.empty:
            ind[f"adx_{p}"] = adx_df.iloc[:, 0]

    # MACD
    macd = ta.macd(close)
    if macd is not None:
        ind["macd_hist"] = macd.iloc[:, 2]  # histogram

    # Volume MA
    for p in [20]:
        ind[f"vol_ma_{p}"] = ta.sma(vol.astype(float), length=p)

    return ind


def simulate_trades(signals, df, sl_prices, tp_prices, pip_size=0.0001, spread_pips=1.5, initial_capital=100000):
    """
    Given signal arrays, simulate trades and return metrics.

    signals: array of 0 (no trade), 1 (buy), -1 (sell)
    sl_prices: array of stop loss prices per bar
    tp_prices: array of take profit prices per bar
    """
    spread = spread_pips * pip_size
    equity = initial_capital
    peak = equity
    trades = []
    in_trade = False
    entry_price = 0
    entry_side = 0
    entry_sl = 0
    entry_tp = 0

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    for i in range(len(close)):
        # Check exit for active trade
        if in_trade:
            if entry_side == 1:  # long
                if low[i] <= entry_sl:
                    pnl = (entry_sl - entry_price) * units
                    trades.append(pnl)
                    equity += pnl
                    in_trade = False
                elif high[i] >= entry_tp:
                    pnl = (entry_tp - entry_price) * units
                    trades.append(pnl)
                    equity += pnl
                    in_trade = False
            else:  # short
                if high[i] >= entry_sl:
                    pnl = (entry_price - entry_sl) * units
                    trades.append(pnl)
                    equity += pnl
                    in_trade = False
                elif low[i] <= entry_tp:
                    pnl = (entry_price - entry_tp) * units
                    trades.append(pnl)
                    equity += pnl
                    in_trade = False

            peak = max(peak, equity)

        # Enter new trade
        if not in_trade and signals[i] != 0:
            entry_side = signals[i]
            entry_sl = sl_prices[i]
            entry_tp = tp_prices[i]

            if entry_side == 1:
                entry_price = close[i] + spread / 2
            else:
                entry_price = close[i] - spread / 2

            sl_dist = abs(entry_price - entry_sl)
            if sl_dist > 0:
                risk = equity * 0.01  # 1% risk
                units = risk / sl_dist
                in_trade = True

    # Compute metrics
    if not trades:
        return {"trades": 0, "wr": 0, "pf": 0, "sharpe": 0, "dd": 0, "ret": 0}

    trades_arr = np.array(trades)
    wins = trades_arr[trades_arr > 0]
    losses = trades_arr[trades_arr < 0]

    total = len(trades_arr)
    wr = len(wins) / total * 100 if total > 0 else 0
    pf = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else (99 if len(wins) > 0 else 0)

    # Equity curve for Sharpe
    eq_curve = np.cumsum(trades_arr) + initial_capital
    eq_all = np.concatenate([[initial_capital], eq_curve])
    returns = np.diff(eq_all) / eq_all[:-1]
    sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0

    # Max drawdown
    peak_arr = np.maximum.accumulate(eq_all)
    dd = ((peak_arr - eq_all) / peak_arr * 100)
    max_dd = float(np.max(dd))

    ret = (eq_all[-1] - initial_capital) / initial_capital * 100

    return {
        "trades": total,
        "wr": wr,
        "pf": min(pf, 99),
        "sharpe": sharpe,
        "dd": max_dd,
        "ret": ret,
    }


def optimize_mean_reversion(df, ind, n_samples=200):
    """Optimize mean reversion parameters."""
    close = df["close"].values
    pip_size = 0.0001

    param_space = {
        "bb_period": [15, 20, 25, 30],
        "bb_std": [1.5, 2.0, 2.5, 3.0],
        "rsi_period": [7, 10, 14, 21],
        "rsi_oversold": [20, 25, 30, 35],
        "rsi_overbought": [65, 70, 75, 80],
        "adx_max": [20, 25, 30, 35, 40],
        "sl_atr_mult": [1.0, 1.5, 2.0, 2.5],
    }

    keys = list(param_space.keys())
    all_combos = list(product(*[param_space[k] for k in keys]))
    random.shuffle(all_combos)
    combos = all_combos[:n_samples]

    results = []
    for i, values in enumerate(combos):
        p = dict(zip(keys, values))

        bbl_key = f"bbl_{p['bb_period']}_{p['bb_std']}"
        bbu_key = f"bbu_{p['bb_period']}_{p['bb_std']}"
        bbm_key = f"bbm_{p['bb_period']}_{p['bb_std']}"
        rsi_key = f"rsi_{p['rsi_period']}"
        atr_key = "atr_14"
        adx_key = "adx_14"

        if bbl_key not in ind or rsi_key not in ind:
            continue

        bbl = ind[bbl_key].values
        bbu = ind[bbu_key].values
        bbm = ind[bbm_key].values
        rsi = ind[rsi_key].values
        atr = ind[atr_key].values
        adx = ind[adx_key].values

        n = len(close)
        signals = np.zeros(n)
        sl_prices = np.zeros(n)
        tp_prices = np.zeros(n)

        for j in range(max(p["bb_period"], p["rsi_period"], 14) + 5, n):
            if np.isnan(bbl[j]) or np.isnan(rsi[j]) or np.isnan(atr[j]) or np.isnan(adx[j]):
                continue
            if adx[j] > p["adx_max"]:
                continue

            # BUY: price at lower BB + RSI oversold
            if close[j] <= bbl[j] and rsi[j] < p["rsi_oversold"]:
                sl = close[j] - atr[j] * p["sl_atr_mult"]
                tp = bbm[j]
                risk = close[j] - sl
                reward = tp - close[j]
                if risk > 0 and reward / risk >= 1.5:
                    signals[j] = 1
                    sl_prices[j] = sl
                    tp_prices[j] = tp

            # SELL: price at upper BB + RSI overbought
            elif close[j] >= bbu[j] and rsi[j] > p["rsi_overbought"]:
                sl = close[j] + atr[j] * p["sl_atr_mult"]
                tp = bbm[j]
                risk = sl - close[j]
                reward = close[j] - tp
                if risk > 0 and reward / risk >= 1.5:
                    signals[j] = -1
                    sl_prices[j] = sl
                    tp_prices[j] = tp

        metrics = simulate_trades(signals, df, sl_prices, tp_prices, pip_size)
        if metrics["trades"] >= 5:
            metrics["params"] = p
            results.append(metrics)

        if (i + 1) % 50 == 0:
            best_sh = max((r["sharpe"] for r in results), default=0)
            print(f"    [{i+1}/{len(combos)}] best sharpe={best_sh:.3f}")

    results.sort(key=lambda x: x["sharpe"], reverse=True)
    return results


def optimize_trend_following(df, ind, n_samples=200):
    """Optimize trend following parameters."""
    close = df["close"].values
    pip_size = 0.0001

    param_space = {
        "fast_ema": [8, 10, 15, 20, 25],
        "slow_ema": [30, 40, 50, 60, 80],
        "adx_min": [15, 20, 25, 30],
        "sl_atr_mult": [1.0, 1.5, 2.0, 2.5, 3.0],
        "tp_atr_mult": [2.0, 2.5, 3.0, 4.0, 5.0],
    }

    keys = list(param_space.keys())
    all_combos = list(product(*[param_space[k] for k in keys]))
    random.shuffle(all_combos)
    combos = all_combos[:n_samples]

    results = []
    for i, values in enumerate(combos):
        p = dict(zip(keys, values))

        if p["fast_ema"] >= p["slow_ema"]:
            continue

        fast_key = f"ema_{p['fast_ema']}"
        slow_key = f"ema_{p['slow_ema']}"
        atr_key = "atr_14"
        adx_key = "adx_14"

        if fast_key not in ind or slow_key not in ind:
            continue

        fast = ind[fast_key].values
        slow = ind[slow_key].values
        atr = ind[atr_key].values
        adx = ind[adx_key].values

        n = len(close)
        signals = np.zeros(n)
        sl_prices = np.zeros(n)
        tp_prices = np.zeros(n)

        for j in range(p["slow_ema"] + 5, n):
            if np.isnan(fast[j]) or np.isnan(slow[j]) or np.isnan(atr[j]) or np.isnan(adx[j]):
                continue
            if adx[j] < p["adx_min"]:
                continue

            # Bullish crossover
            if fast[j-1] <= slow[j-1] and fast[j] > slow[j]:
                sl = close[j] - atr[j] * p["sl_atr_mult"]
                tp = close[j] + atr[j] * p["tp_atr_mult"]
                signals[j] = 1
                sl_prices[j] = sl
                tp_prices[j] = tp

            # Bearish crossover
            elif fast[j-1] >= slow[j-1] and fast[j] < slow[j]:
                sl = close[j] + atr[j] * p["sl_atr_mult"]
                tp = close[j] - atr[j] * p["tp_atr_mult"]
                signals[j] = -1
                sl_prices[j] = sl
                tp_prices[j] = tp

        metrics = simulate_trades(signals, df, sl_prices, tp_prices, pip_size)
        if metrics["trades"] >= 5:
            metrics["params"] = p
            results.append(metrics)

        if (i + 1) % 50 == 0:
            best_sh = max((r["sharpe"] for r in results), default=0)
            print(f"    [{i+1}/{len(combos)}] best sharpe={best_sh:.3f}")

    results.sort(key=lambda x: x["sharpe"], reverse=True)
    return results


def optimize_confluence(df, ind, n_samples=300):
    """Optimize confluence scoring strategy."""
    close = df["close"].values
    vol = df["volume"].values
    pip_size = 0.0001

    param_space = {
        "score_min": [3, 4, 5, 6],
        "ema_trend": [100, 150, 200, 250],
        "ema_fast": [15, 20, 25],
        "ema_slow": [40, 50, 60],
        "rsi_period": [10, 14, 21],
        "adx_min": [15, 20, 25],
        "sl_atr_mult": [1.0, 1.5, 2.0, 2.5],
        "tp_atr_mult": [2.0, 2.5, 3.0, 4.0],
    }

    keys = list(param_space.keys())
    all_combos = list(product(*[param_space[k] for k in keys]))
    random.shuffle(all_combos)
    combos = all_combos[:n_samples]

    results = []
    for i, values in enumerate(combos):
        p = dict(zip(keys, values))

        if p["ema_fast"] >= p["ema_slow"]:
            continue

        ema_t = ind.get(f"ema_{p['ema_trend']}")
        ema_f = ind.get(f"ema_{p['ema_fast']}")
        ema_s = ind.get(f"ema_{p['ema_slow']}")
        rsi = ind.get(f"rsi_{p['rsi_period']}")
        adx = ind.get(f"adx_14")
        atr = ind.get(f"atr_14")
        macd_hist = ind.get("macd_hist")
        bbl = ind.get("bbl_20_2.0")
        vol_ma = ind.get("vol_ma_20")

        if any(x is None for x in [ema_t, ema_f, ema_s, rsi, adx, atr, macd_hist, bbl, vol_ma]):
            continue

        ema_tv = ema_t.values
        ema_fv = ema_f.values
        ema_sv = ema_s.values
        rsiv = rsi.values
        adxv = adx.values
        atrv = atr.values
        macdv = macd_hist.values
        bblv = bbl.values
        vol_mav = vol_ma.values

        n = len(close)
        signals = np.zeros(n)
        sl_prices = np.zeros(n)
        tp_prices = np.zeros(n)
        start = max(p["ema_trend"], 50) + 5

        for j in range(start, n):
            if any(np.isnan(x) for x in [ema_tv[j], ema_fv[j], ema_sv[j], rsiv[j], adxv[j], atrv[j]]):
                continue

            # Score for BUY
            buy_score = 0
            if close[j] > ema_tv[j]: buy_score += 1  # trend
            if ema_fv[j] > ema_sv[j]: buy_score += 1  # medium trend
            if 30 <= rsiv[j] <= 60: buy_score += 1     # RSI room to run
            if not np.isnan(macdv[j]) and macdv[j] > 0: buy_score += 1  # MACD
            if adxv[j] > p["adx_min"]: buy_score += 1  # trending
            if not np.isnan(bblv[j]) and close[j] < bblv[j] * 1.01: buy_score += 1  # near BB
            if not np.isnan(vol_mav[j]) and vol[j] > vol_mav[j]: buy_score += 1  # volume

            if buy_score >= p["score_min"]:
                sl = close[j] - atrv[j] * p["sl_atr_mult"]
                tp = close[j] + atrv[j] * p["tp_atr_mult"]
                risk = close[j] - sl
                reward = tp - close[j]
                if risk > 0 and reward / risk >= 1.5:
                    signals[j] = 1
                    sl_prices[j] = sl
                    tp_prices[j] = tp
                    continue

            # Score for SELL
            sell_score = 0
            if close[j] < ema_tv[j]: sell_score += 1
            if ema_fv[j] < ema_sv[j]: sell_score += 1
            if 40 <= rsiv[j] <= 70: sell_score += 1
            if not np.isnan(macdv[j]) and macdv[j] < 0: sell_score += 1
            if adxv[j] > p["adx_min"]: sell_score += 1
            bbu = ind.get("bbu_20_2.0")
            if bbu is not None and not np.isnan(bbu.values[j]) and close[j] > bbu.values[j] * 0.99: sell_score += 1
            if not np.isnan(vol_mav[j]) and vol[j] > vol_mav[j]: sell_score += 1

            if sell_score >= p["score_min"]:
                sl = close[j] + atrv[j] * p["sl_atr_mult"]
                tp = close[j] - atrv[j] * p["tp_atr_mult"]
                risk = sl - close[j]
                reward = close[j] - tp
                if risk > 0 and reward / risk >= 1.5:
                    signals[j] = -1
                    sl_prices[j] = sl
                    tp_prices[j] = tp

        metrics = simulate_trades(signals, df, sl_prices, tp_prices, pip_size)
        if metrics["trades"] >= 5:
            metrics["params"] = p
            results.append(metrics)

        if (i + 1) % 50 == 0:
            best_sh = max((r["sharpe"] for r in results), default=0)
            print(f"    [{i+1}/{len(combos)}] best sharpe={best_sh:.3f}")

    results.sort(key=lambda x: x["sharpe"], reverse=True)
    return results


def print_top(results, label, top_n=10):
    """Print top results with parameters."""
    print(f"\n  {'='*65}")
    print(f"  TOP {top_n} — {label}")
    print(f"  {'='*65}")
    if not results:
        print("  No valid results found")
        return

    print(f"  {'#':<4} {'Trades':<7} {'WR%':<7} {'PF':<7} {'Sharpe':<8} {'DD%':<7} {'Ret%':<9}")
    print(f"  {'-'*55}")
    for i, r in enumerate(results[:top_n]):
        print(f"  {i+1:<4} {r['trades']:<7} {r['wr']:<7.1f} {r['pf']:<7.2f} {r['sharpe']:<8.3f} {r['dd']:<7.1f} {r['ret']:<+9.2f}")

    if results:
        best = results[0]
        print(f"\n  BEST PARAMS (Sharpe={best['sharpe']:.3f}, WR={best['wr']:.1f}%, PF={best['pf']:.2f}):")
        for k, v in best["params"].items():
            print(f"    {k} = {v}")

    # Show top 3 for diversity
    print(f"\n  TOP 3 PARAM SETS:")
    for rank, r in enumerate(results[:3]):
        print(f"    #{rank+1}: {r['params']}")


def main():
    cfg = get_config()
    broker = OandaBroker(cfg.broker)

    print("=" * 70)
    print("  FAST VECTORIZED PARAMETER OPTIMIZER")
    print("=" * 70)

    pairs_tf = [
        ("EUR_USD", "M15"),
        ("GBP_USD", "M15"),
        ("EUR_USD", "H1"),
        ("GBP_USD", "H1"),
        ("USD_JPY", "H1"),
    ]

    data = {}
    indicators = {}

    print("\nDownloading data & pre-computing indicators...")
    for pair, tf in pairs_tf:
        key = f"{pair}/{tf}"
        print(f"  {key}...", end=" ", flush=True)
        df = download_data(broker, pair, tf, 5000)
        data[key] = df
        indicators[key] = precompute_indicators(df)
        print(f"{len(df)} candles, {len(indicators[key])} indicators")

    broker.close()

    # ===== MEAN REVERSION (M15) =====
    print(f"\n{'#'*70}")
    print(f"# MEAN REVERSION OPTIMIZATION")
    print(f"{'#'*70}")
    for pair in ["GBP_USD", "EUR_USD"]:
        key = f"{pair}/M15"
        results = optimize_mean_reversion(data[key], indicators[key], n_samples=300)
        print_top(results, f"Mean Reversion — {pair}/M15")

    # ===== TREND FOLLOWING (H1) =====
    print(f"\n{'#'*70}")
    print(f"# TREND FOLLOWING OPTIMIZATION")
    print(f"{'#'*70}")
    for pair in ["EUR_USD", "GBP_USD", "USD_JPY"]:
        key = f"{pair}/H1"
        results = optimize_trend_following(data[key], indicators[key], n_samples=300)
        print_top(results, f"Trend Following — {pair}/H1")

    # ===== CONFLUENCE (H1) =====
    print(f"\n{'#'*70}")
    print(f"# CONFLUENCE OPTIMIZATION")
    print(f"{'#'*70}")
    for pair in ["EUR_USD", "GBP_USD"]:
        key = f"{pair}/H1"
        results = optimize_confluence(data[key], indicators[key], n_samples=400)
        print_top(results, f"Confluence — {pair}/H1")

    print(f"\n{'='*70}")
    print(f"  OPTIMIZATION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
