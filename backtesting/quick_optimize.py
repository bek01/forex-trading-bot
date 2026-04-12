#!/usr/bin/env python3
"""
Lean parameter optimizer — single-process, memory-efficient.
Designed for 2-CPU / 2GB RAM VPS.

Usage:
    python -m backtesting.quick_optimize
"""

import sys
import random
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtesting.engine import BacktestEngine
from execution.broker import OandaBroker
from config import get_config
from strategies.mean_reversion import MeanReversionStrategy
from strategies.trend_following import TrendFollowingStrategy
from strategies.confluence import ConfluenceStrategy
from strategies.session_momentum import SessionMomentumStrategy


def optimize_strategy(name, strategy_class, param_grid, candles, pair, engine, n_samples=50):
    """Test n_samples random parameter combinations."""
    # Generate all combos, sample N
    keys = list(param_grid.keys())
    all_values = [param_grid[k] for k in keys]

    # Total combos
    total = 1
    for v in all_values:
        total *= len(v)

    combos = []
    for _ in range(min(n_samples, total)):
        combo = {k: random.choice(v) for k, v in zip(keys, all_values)}
        combos.append(combo)

    print(f"\n{'='*70}")
    print(f"  {name.upper()} on {pair} — testing {len(combos)} combos (of {total} total)")
    print(f"{'='*70}")

    results = []
    for i, params in enumerate(combos):
        strat = strategy_class()
        for k, v in params.items():
            setattr(strat, k, v)

        try:
            result = engine.run(strat, candles, pair)
        except Exception:
            continue

        if result.total_trades < 5:
            continue

        results.append({
            "params": params,
            "trades": result.total_trades,
            "wr": result.win_rate,
            "pf": result.profit_factor,
            "sharpe": result.sharpe_ratio,
            "dd": result.max_drawdown_pct,
            "ret": result.total_return_pct,
            "expectancy": result.expectancy,
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(combos)}] best so far: sharpe={max((r['sharpe'] for r in results), default=0):.2f}")

    # Sort by sharpe
    results.sort(key=lambda x: x["sharpe"], reverse=True)
    return results


def print_results(results, top_n=10):
    """Print top results."""
    if not results:
        print("  No valid results")
        return

    print(f"\n  {'Rank':<5} {'Trades':<7} {'WR%':<7} {'PF':<6} {'Sharpe':<8} {'DD%':<7} {'Return%':<10}")
    print(f"  {'-'*55}")
    for i, r in enumerate(results[:top_n]):
        print(f"  {i+1:<5} {r['trades']:<7} {r['wr']:<7.1f} {r['pf']:<6.2f} {r['sharpe']:<8.2f} {r['dd']:<7.1f} {r['ret']:<+10.2f}")

    # Print best params
    best = results[0]
    print(f"\n  BEST PARAMETERS (Sharpe={best['sharpe']:.2f}):")
    for k, v in best["params"].items():
        print(f"    {k} = {v}")


def main():
    cfg = get_config()
    broker = OandaBroker(cfg.broker)
    engine = BacktestEngine(initial_capital=100000, risk_per_trade_pct=1.0, spread_pips=1.5)

    # Download data
    print("Downloading data...")
    data = {}
    for pair in ["EUR_USD", "GBP_USD", "USD_JPY"]:
        for tf in ["M15", "H1"]:
            key = f"{pair}/{tf}"
            raw = broker.get_candles(pair, tf, 5000)
            data[key] = [c for c in raw if c.complete]
            print(f"  {key}: {len(data[key])} candles")
    broker.close()

    # ===== MEAN REVERSION =====
    mr_grid = {
        "bb_period": [15, 20, 25, 30],
        "bb_std": [1.5, 2.0, 2.5, 3.0],
        "rsi_period": [10, 14, 21],
        "rsi_oversold": [20, 25, 30, 35],
        "rsi_overbought": [65, 70, 75, 80],
        "adx_max": [20, 25, 30, 35, 40],
        "sl_atr_multiplier": [1.0, 1.5, 2.0, 2.5],
    }

    # Focus on GBP_USD (best default result) + EUR_USD
    for pair in ["GBP_USD", "EUR_USD"]:
        results = optimize_strategy(
            "mean_reversion", MeanReversionStrategy, mr_grid,
            data[f"{pair}/M15"], pair, engine, n_samples=80
        )
        print_results(results)

    # ===== TREND FOLLOWING =====
    tf_grid = {
        "fast_ema": [8, 10, 15, 20, 25],
        "slow_ema": [30, 40, 50, 60, 80],
        "adx_min": [15, 20, 25, 30],
        "sl_atr_multiplier": [1.0, 1.5, 2.0, 2.5, 3.0],
        "tp_atr_multiplier": [2.0, 2.5, 3.0, 4.0, 5.0],
    }

    for pair in ["EUR_USD", "GBP_USD"]:
        results = optimize_strategy(
            "trend_following", TrendFollowingStrategy, tf_grid,
            data[f"{pair}/H1"], pair, engine, n_samples=80
        )
        print_results(results)

    # ===== CONFLUENCE =====
    conf_grid = {
        "score_threshold": [4, 5, 6],
        "ema_trend_period": [100, 150, 200, 250],
        "ema_fast_period": [15, 20, 25, 30],
        "ema_slow_period": [40, 50, 60, 80],
        "rsi_period": [10, 14, 21],
        "adx_period": [10, 14, 20],
        "adx_min": [15, 20, 25, 30],
        "sl_atr_multiplier": [1.0, 1.5, 2.0, 2.5],
        "tp_atr_multiplier": [2.0, 2.5, 3.0, 4.0],
    }

    for pair in ["EUR_USD", "GBP_USD"]:
        results = optimize_strategy(
            "confluence", ConfluenceStrategy, conf_grid,
            data[f"{pair}/H1"], pair, engine, n_samples=100
        )
        print_results(results)

    # ===== SESSION MOMENTUM =====
    sm_grid = {
        "breakout_atr_multiplier": [0.2, 0.3, 0.5, 0.7],
        "rsi_long_threshold": [50, 55, 60],
        "rsi_short_threshold": [40, 45, 50],
        "adx_min": [15, 20, 25],
        "sl_atr_multiplier": [1.5, 2.0, 2.5, 3.0],
        "tp_rr_ratio": [1.5, 2.0, 2.5, 3.0],
    }

    for pair in ["EUR_USD", "GBP_USD"]:
        results = optimize_strategy(
            "session_momentum", SessionMomentumStrategy, sm_grid,
            data[f"{pair}/M15"], pair, engine, n_samples=60
        )
        print_results(results)

    print("\n" + "=" * 70)
    print("  OPTIMIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
