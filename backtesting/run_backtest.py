#!/usr/bin/env python3
"""
Run backtests on strategies with historical data.

Usage:
    python -m backtesting.run_backtest                         # All strategies, all pairs
    python -m backtesting.run_backtest --strategy mean_reversion --pair EUR_USD
    python -m backtesting.run_backtest --monte-carlo           # Include Monte Carlo
    python -m backtesting.run_backtest --download              # Download fresh data first
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtesting.engine import BacktestEngine
from config import get_config
from execution.broker import OandaBroker
from strategies.mean_reversion import MeanReversionStrategy
from strategies.trend_following import TrendFollowingStrategy
from strategies.london_breakout import LondonBreakoutStrategy

logger = logging.getLogger(__name__)


STRATEGY_MAP = {
    "mean_reversion": MeanReversionStrategy,
    "trend_following": TrendFollowingStrategy,
    "london_breakout": LondonBreakoutStrategy,
}


def download_data(broker: OandaBroker, instrument: str, timeframe: str, count: int = 5000):
    """Download candle data from OANDA."""
    print(f"Downloading {count} {timeframe} candles for {instrument}...")
    candles = broker.get_candles(instrument, timeframe, count=count)
    complete = [c for c in candles if c.complete]
    print(f"  Got {len(complete)} complete candles")
    if complete:
        print(f"  Range: {complete[0].timestamp} → {complete[-1].timestamp}")
    return complete


def run_backtest(
    strategy_name: str,
    instrument: str,
    timeframe: str,
    candles,
    initial_capital: float = 10000.0,
    monte_carlo: bool = False,
):
    """Run a single backtest."""
    strategy_class = STRATEGY_MAP.get(strategy_name)
    if not strategy_class:
        print(f"Unknown strategy: {strategy_name}")
        return None

    strategy = strategy_class()
    engine = BacktestEngine(
        initial_capital=initial_capital,
        risk_per_trade_pct=1.0,
        spread_pips=1.5,
        slippage_pips=0.5,
    )

    print(f"\nRunning backtest: {strategy_name} on {instrument}/{timeframe}...")
    result = engine.run(strategy, candles, instrument)

    print("\n" + result.summary())

    if monte_carlo and result.trades:
        print("\n--- Monte Carlo Simulation ---")
        mc = engine.monte_carlo(result.trades, simulations=1000)
        print(f"Profitable: {mc['profitable_pct']:.1f}% of simulations")
        print(f"Equity (5th/50th/95th): "
              f"${mc['equity_p5']:,.0f} / ${mc['equity_median']:,.0f} / ${mc['equity_p95']:,.0f}")
        print(f"Max DD (median/95th): {mc['max_dd_median']:.1f}% / {mc['max_dd_p95']:.1f}%")
        print(f"Ruin risk (<50% capital): {mc['ruin_pct']:.1f}%")

    return result


def main():
    parser = argparse.ArgumentParser(description="Run strategy backtests")
    parser.add_argument("--strategy", type=str, default=None,
                        help=f"Strategy to test: {list(STRATEGY_MAP.keys())}")
    parser.add_argument("--pair", type=str, default=None,
                        help="Instrument (e.g., EUR_USD)")
    parser.add_argument("--timeframe", type=str, default="H1",
                        help="Candle timeframe (default: H1)")
    parser.add_argument("--candles", type=int, default=5000,
                        help="Number of candles to fetch (default: 5000)")
    parser.add_argument("--capital", type=float, default=10000.0,
                        help="Starting capital (default: 10000)")
    parser.add_argument("--monte-carlo", action="store_true",
                        help="Run Monte Carlo simulation")
    parser.add_argument("--download", action="store_true",
                        help="Download fresh data from OANDA")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    config = get_config()

    strategies = [args.strategy] if args.strategy else list(STRATEGY_MAP.keys())
    pairs = [args.pair] if args.pair else config.strategy.instruments

    # Download data
    broker = OandaBroker(config.broker)
    data_cache = {}

    print("=" * 60)
    print("BACKTEST SUITE")
    print("=" * 60)

    for pair in pairs:
        key = f"{pair}/{args.timeframe}"
        if key not in data_cache:
            try:
                candles = download_data(broker, pair, args.timeframe, args.candles)
                data_cache[key] = candles
            except Exception as e:
                print(f"Failed to download data for {pair}: {e}")
                continue

        candles = data_cache[key]
        if not candles:
            print(f"No data for {pair}, skipping")
            continue

        for strat_name in strategies:
            run_backtest(
                strat_name, pair, args.timeframe, candles,
                args.capital, args.monte_carlo,
            )
            print("\n" + "-" * 60)

    broker.close()
    print("\n✅ Backtest suite complete")


if __name__ == "__main__":
    main()
