"""Run the EWMAC + vol-targeted portfolio backtest and print a report.

Usage:
    python -m backtesting.run_ewmac_bt
    python -m backtesting.run_ewmac_bt --years 6 --capital 100000
    python -m backtesting.run_ewmac_bt --walk-forward
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

from backtesting.data_loader import HistoricalDataLoader
from backtesting.forecast_engine import ForecastPortfolioBacktester

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_INSTRUMENTS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD",
    "USD_CAD", "NZD_USD", "USD_CHF", "EUR_JPY",
]

# Pass criteria per the research memo
PASS_SHARPE = 0.6
PASS_MIN_TRADES = 100
PASS_MAX_DD_PCT = -25.0


def _split_dates(df_index: pd.DatetimeIndex, sub_periods: int = 3) -> list[tuple]:
    """Split the date range into N equal sub-periods for stability testing."""
    total = len(df_index)
    chunk = total // sub_periods
    return [
        (df_index[i * chunk], df_index[(i + 1) * chunk - 1])
        for i in range(sub_periods)
    ]


def run_full_period_backtest(args: argparse.Namespace) -> int:
    loader = HistoricalDataLoader()
    logger.info(f"Loading history for {len(DEFAULT_INSTRUMENTS)} instruments, {args.years}y daily")
    count = min(args.years * 260 + 50, 5000)
    price_data = loader.fetch_universe(DEFAULT_INSTRUMENTS, timeframe="D", count=count)
    if not price_data:
        logger.error("No data loaded — check OANDA access")
        return 1
    for inst, df in price_data.items():
        logger.info(f"  {inst}: {len(df)} bars, {df.index.min().date()} → {df.index.max().date()}")

    bt = ForecastPortfolioBacktester()
    result = bt.run(price_data, initial_capital=args.capital)

    print()
    print("=" * 70)
    print("EWMAC + vol-targeted portfolio backtest — FULL PERIOD")
    print("=" * 70)
    print(result.summary())

    # Per-sub-period stability check
    print("Sub-period stability:")
    segments = _split_dates(result.equity_curve.index, sub_periods=3)
    for i, (start, end) in enumerate(segments):
        seg = result.equity_curve.loc[start:end]
        ret = (seg.iloc[-1] / seg.iloc[0] - 1) * 100
        daily_ret = seg.pct_change().dropna()
        sharpe = (daily_ret.mean() / daily_ret.std() * (252 ** 0.5)) if daily_ret.std() > 0 else 0
        print(f"  [{i+1}] {start.date()} → {end.date()}: return {ret:+.1f}%, Sharpe {sharpe:.2f}")

    # Pass / fail verdict vs research criteria
    print()
    print("Pass criteria (from reference_fx_architecture.md):")
    pass_flags = {
        f"Sharpe > {PASS_SHARPE}": result.sharpe_ratio > PASS_SHARPE,
        f"Trades >= {PASS_MIN_TRADES}": result.n_trades >= PASS_MIN_TRADES,
        f"Max DD > {PASS_MAX_DD_PCT}%": result.max_drawdown_pct > PASS_MAX_DD_PCT,
    }
    for crit, ok in pass_flags.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {crit}")
    all_pass = all(pass_flags.values())
    print()
    print(f"OVERALL: {'PASS — safe to deploy to demo' if all_pass else 'FAIL — do NOT deploy'}")
    print("=" * 70)

    return 0 if all_pass else 2


def run_walk_forward(args: argparse.Namespace) -> int:
    """Rolling walk-forward: train/test windows, report OOS aggregate."""
    loader = HistoricalDataLoader()
    count = min(args.years * 260 + 50, 5000)
    price_data = loader.fetch_universe(DEFAULT_INSTRUMENTS, timeframe="D", count=count)
    if not price_data:
        logger.error("No data loaded")
        return 1

    # Build common index
    index = None
    for df in price_data.values():
        index = df.index if index is None else index.intersection(df.index)
    index = index.sort_values()

    # Walk-forward parameters: IS=train, OOS=test, roll by OOS_DAYS
    IS_YEARS = 3
    OOS_DAYS = 252  # 1 year
    is_bars = IS_YEARS * 252
    step = OOS_DAYS

    if len(index) < is_bars + OOS_DAYS * 2:
        logger.error(f"Need at least {is_bars + OOS_DAYS*2} bars, got {len(index)}")
        return 1

    bt = ForecastPortfolioBacktester()
    oos_results = []

    start_idx = is_bars
    while start_idx + OOS_DAYS <= len(index):
        oos_start = index[start_idx]
        oos_end = index[min(start_idx + OOS_DAYS - 1, len(index) - 1)]
        # For EWMAC we have no fit params, so "train" is not needed; we just
        # test OOS with the standard Carver parameters. This is walk-forward
        # OOS testing of a fixed-parameter strategy — the point is to confirm
        # performance doesn't depend on a specific period.
        oos_price_data = {
            inst: df.loc[:oos_end].tail(is_bars + OOS_DAYS + 50)
            for inst, df in price_data.items()
        }
        result = bt.run(oos_price_data, initial_capital=args.capital)
        # Restrict equity_curve to OOS window only
        oos_curve = result.equity_curve.loc[oos_start:oos_end]
        oos_trades = [t for t in result.trades if oos_start <= t.timestamp <= oos_end]
        oos_results.append({
            "start": oos_start,
            "end": oos_end,
            "curve": oos_curve,
            "trades": oos_trades,
            "return_pct": (oos_curve.iloc[-1] / oos_curve.iloc[0] - 1) * 100,
        })
        start_idx += step

    print()
    print("=" * 70)
    print(f"Walk-forward OOS analysis — {len(oos_results)} windows of {OOS_DAYS/252:.0f}y each")
    print("=" * 70)
    for r in oos_results:
        daily_ret = r["curve"].pct_change().dropna()
        sharpe = (daily_ret.mean() / daily_ret.std() * (252 ** 0.5)) if daily_ret.std() > 0 else 0
        print(f"  {r['start'].date()} → {r['end'].date()}: "
              f"return {r['return_pct']:+6.1f}%, Sharpe {sharpe:+.2f}, trades {len(r['trades'])}")

    # Aggregate OOS stats by concatenating returns
    all_rets = pd.concat([r["curve"].pct_change().dropna() for r in oos_results])
    agg_sharpe = (all_rets.mean() / all_rets.std() * (252 ** 0.5)) if all_rets.std() > 0 else 0
    total_trades = sum(len(r["trades"]) for r in oos_results)
    print()
    print(f"AGGREGATE OOS:   Sharpe {agg_sharpe:.2f}   trades {total_trades}")
    print("=" * 70)

    return 0 if agg_sharpe >= PASS_SHARPE else 2


def main():
    parser = argparse.ArgumentParser(description="EWMAC portfolio backtest")
    parser.add_argument("--years", type=int, default=6, help="Years of history")
    parser.add_argument("--capital", type=float, default=100_000, help="Starting capital")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward analysis")
    args = parser.parse_args()

    if args.walk_forward:
        sys.exit(run_walk_forward(args))
    else:
        sys.exit(run_full_period_backtest(args))


if __name__ == "__main__":
    main()
