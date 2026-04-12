#!/usr/bin/env python3
"""
Parameter Optimization Engine — Grid search + walk-forward analysis.

Supports three search modes:
1. Random search: sample N random parameter combinations from the full grid
2. Fine-tuning: narrow grid around best results from random search
3. Walk-forward: train/test split validation to detect overfitting

Usage:
    python -m backtesting.optimizer --strategy mean_reversion --iterations 500
    python -m backtesting.optimizer --strategy trend_following --pair EUR_USD --iterations 300
    python -m backtesting.optimizer --strategy london_breakout --timeframe M5 --iterations 200
    python -m backtesting.optimizer --all --iterations 500
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtesting.engine import BacktestEngine, BacktestResult
from config import get_config
from execution.broker import OandaBroker
from models import Candle
from strategies.base import Strategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.trend_following import TrendFollowingStrategy
from strategies.london_breakout import LondonBreakoutStrategy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strategy registry & parameter grids
# ---------------------------------------------------------------------------

STRATEGY_MAP = {
    "mean_reversion": MeanReversionStrategy,
    "trend_following": TrendFollowingStrategy,
    "london_breakout": LondonBreakoutStrategy,
}

PARAM_GRIDS: dict[str, dict[str, list]] = {
    "mean_reversion": {
        "bb_period": [15, 20, 25, 30],
        "bb_std": [1.5, 2.0, 2.5],
        "rsi_period": [10, 14, 21],
        "rsi_oversold": [25, 30, 35],
        "rsi_overbought": [65, 70, 75],
        "adx_max": [20, 25, 30],
        "sl_atr_multiplier": [1.0, 1.5, 2.0],
    },
    "trend_following": {
        "fast_ema": [10, 15, 20, 25],
        "slow_ema": [40, 50, 60],
        "adx_min": [20, 25, 30],
        "sl_atr_multiplier": [1.5, 2.0, 2.5],
        "tp_atr_multiplier": [2.0, 3.0, 4.0],
    },
    "london_breakout": {
        "min_range_pips": [10, 15, 20],
        "max_range_pips": [40, 50, 60],
        "tp_range_multiplier": [1.0, 1.5, 2.0, 2.5],
    },
}

DEFAULT_PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """Holds results for one parameter combination."""
    params: dict[str, Any] = field(default_factory=dict)
    strategy_name: str = ""
    # Aggregated across all pairs
    sharpe_ratio: float = 0.0
    total_return_pct: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    max_drawdown_pct: float = 0.0
    expectancy: float = 0.0
    sortino_ratio: float = 0.0
    # Per-pair breakdown
    pair_results: dict[str, dict] = field(default_factory=dict)
    # Walk-forward
    is_oos_profitable: float = 0.0  # fraction of OOS windows profitable
    oos_sharpe: float = 0.0
    oos_return_pct: float = 0.0


# ---------------------------------------------------------------------------
# Helper: build strategy with custom params
# ---------------------------------------------------------------------------

def _make_strategy(strategy_name: str, params: dict) -> Strategy:
    """Instantiate a strategy with overridden parameters."""
    cls = STRATEGY_MAP[strategy_name]
    strat = cls()
    for k, v in params.items():
        setattr(strat, k, v)
    return strat


# ---------------------------------------------------------------------------
# Worker function for multiprocessing
# ---------------------------------------------------------------------------

def _run_single_combo(args: tuple) -> Optional[dict]:
    """
    Run one parameter combination across all pairs.
    Returns a dict with aggregated metrics or None on error.

    args: (strategy_name, params_dict, data_by_pair, engine_kwargs)
    """
    strategy_name, params, data_by_pair, engine_kwargs = args

    try:
        engine = BacktestEngine(**engine_kwargs)
        all_trades = []
        pair_results = {}

        for pair, candles in data_by_pair.items():
            if not candles:
                continue
            strat = _make_strategy(strategy_name, params)
            result = engine.run(strat, candles, pair)
            pair_results[pair] = {
                "sharpe": result.sharpe_ratio,
                "return_pct": result.total_return_pct,
                "trades": result.total_trades,
                "win_rate": result.win_rate,
                "profit_factor": result.profit_factor,
                "max_dd": result.max_drawdown_pct,
                "expectancy": result.expectancy,
            }
            all_trades.extend(result.trades)

        if not all_trades:
            return None

        # Aggregate metrics across pairs
        pnls = [t.pnl for t in all_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        total_trades = len(pnls)
        win_rate = len(wins) / total_trades * 100 if total_trades else 0
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        expectancy = float(np.mean(pnls)) if pnls else 0

        # Aggregate sharpe: average across pairs (weighted by trades)
        sharpes = []
        sortinos = []
        returns = []
        max_dds = []
        for pr in pair_results.values():
            if pr["trades"] > 0:
                sharpes.append(pr["sharpe"])
                returns.append(pr["return_pct"])
                max_dds.append(pr["max_dd"])

        avg_sharpe = float(np.mean(sharpes)) if sharpes else 0
        avg_return = float(np.mean(returns)) if returns else 0
        max_dd = float(np.max(max_dds)) if max_dds else 0

        return {
            "params": params,
            "strategy_name": strategy_name,
            "sharpe_ratio": avg_sharpe,
            "total_return_pct": avg_return,
            "profit_factor": profit_factor,
            "win_rate": win_rate,
            "total_trades": total_trades,
            "max_drawdown_pct": max_dd,
            "expectancy": expectancy,
            "pair_results": pair_results,
        }

    except Exception as e:
        logger.debug(f"Error in combo {params}: {e}")
        return None


# ---------------------------------------------------------------------------
# Walk-forward worker
# ---------------------------------------------------------------------------

def _run_walk_forward_window(args: tuple) -> Optional[dict]:
    """
    Run one walk-forward window: optimize on train, validate on test.

    args: (strategy_name, best_params_list, train_data, test_data, engine_kwargs)
    Returns dict with in-sample and out-of-sample metrics for the best params.
    """
    strategy_name, best_params_list, train_data, test_data, engine_kwargs = args

    try:
        engine = BacktestEngine(**engine_kwargs)

        # Find best params on training data
        best_sharpe = -999
        best_params = best_params_list[0]

        for params in best_params_list:
            sharpes = []
            for pair, candles in train_data.items():
                if not candles:
                    continue
                strat = _make_strategy(strategy_name, params)
                result = engine.run(strat, candles, pair)
                if result.total_trades > 0:
                    sharpes.append(result.sharpe_ratio)

            avg_sharpe = float(np.mean(sharpes)) if sharpes else -999
            if avg_sharpe > best_sharpe:
                best_sharpe = avg_sharpe
                best_params = params

        # Validate on test data
        oos_sharpes = []
        oos_returns = []
        oos_trades = 0
        for pair, candles in test_data.items():
            if not candles:
                continue
            strat = _make_strategy(strategy_name, best_params)
            result = engine.run(strat, candles, pair)
            if result.total_trades > 0:
                oos_sharpes.append(result.sharpe_ratio)
                oos_returns.append(result.total_return_pct)
                oos_trades += result.total_trades

        oos_sharpe = float(np.mean(oos_sharpes)) if oos_sharpes else 0
        oos_return = float(np.mean(oos_returns)) if oos_returns else 0

        return {
            "best_params": best_params,
            "is_sharpe": best_sharpe,
            "oos_sharpe": oos_sharpe,
            "oos_return_pct": oos_return,
            "oos_trades": oos_trades,
            "oos_profitable": oos_return > 0,
        }

    except Exception as e:
        logger.debug(f"Walk-forward window error: {e}")
        return None


# ---------------------------------------------------------------------------
# Main Optimizer class
# ---------------------------------------------------------------------------

class StrategyOptimizer:
    """Grid search + walk-forward parameter optimizer."""

    def __init__(
        self,
        strategy_name: str,
        pairs: list[str] | None = None,
        timeframe: str = "H1",
        candle_count: int = 5000,
        initial_capital: float = 10000.0,
        n_workers: int | None = None,
    ):
        self.strategy_name = strategy_name
        self.pairs = pairs or DEFAULT_PAIRS
        self.timeframe = timeframe
        self.candle_count = candle_count
        self.initial_capital = initial_capital
        self.n_workers = n_workers or max(1, cpu_count() - 1)

        if strategy_name not in STRATEGY_MAP:
            raise ValueError(
                f"Unknown strategy '{strategy_name}'. "
                f"Available: {list(STRATEGY_MAP.keys())}"
            )

        self.param_grid = PARAM_GRIDS[strategy_name]
        self.engine_kwargs = {
            "initial_capital": initial_capital,
            "risk_per_trade_pct": 1.0,
            "spread_pips": 1.5,
            "slippage_pips": 0.5,
        }

        # Will be populated by download_data()
        self.data_by_pair: dict[str, list[Candle]] = {}

    # --- Data download ---

    def download_data(self) -> None:
        """Download historical data from OANDA for all pairs."""
        config = get_config()
        broker = OandaBroker(config.broker)

        print(f"\n{'='*70}")
        print(f"DOWNLOADING DATA")
        print(f"{'='*70}")

        for pair in self.pairs:
            print(f"  Fetching {self.candle_count} {self.timeframe} candles for {pair}...", end=" ")
            try:
                candles = broker.get_candles(pair, self.timeframe, count=self.candle_count)
                complete = [c for c in candles if c.complete]
                self.data_by_pair[pair] = complete
                if complete:
                    print(f"{len(complete)} candles ({complete[0].timestamp.date()} to {complete[-1].timestamp.date()})")
                else:
                    print("0 candles")
            except Exception as e:
                print(f"FAILED: {e}")
                self.data_by_pair[pair] = []

        broker.close()

        total = sum(len(v) for v in self.data_by_pair.values())
        print(f"\n  Total: {total} candles across {len([p for p in self.data_by_pair if self.data_by_pair[p]])} pairs")

    # --- Grid enumeration ---

    def _full_grid_size(self) -> int:
        """Total number of combinations in the full grid."""
        sizes = [len(v) for v in self.param_grid.values()]
        result = 1
        for s in sizes:
            result *= s
        return result

    def _sample_random_combos(self, n: int) -> list[dict]:
        """Sample N random parameter combinations from the grid."""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        full_size = self._full_grid_size()

        if n >= full_size:
            # Just do exhaustive if N >= total combos
            return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

        seen = set()
        combos = []
        max_attempts = n * 10  # prevent infinite loop
        attempts = 0

        while len(combos) < n and attempts < max_attempts:
            attempts += 1
            chosen = tuple(random.choice(v) for v in values)
            if chosen not in seen:
                seen.add(chosen)
                combos.append(dict(zip(keys, chosen)))

        return combos

    def _generate_fine_grid(self, best_params: dict, radius: int = 1) -> list[dict]:
        """
        Generate a fine-tuning grid around the best parameters.
        For each parameter, sample values near the best value within the original grid.
        """
        keys = list(self.param_grid.keys())
        fine_values = {}

        for key in keys:
            grid_vals = sorted(self.param_grid[key])
            best_val = best_params[key]

            # Find index of best value (or nearest)
            try:
                idx = grid_vals.index(best_val)
            except ValueError:
                idx = min(range(len(grid_vals)), key=lambda i: abs(grid_vals[i] - best_val))

            # Take a window of values around the best
            lo = max(0, idx - radius)
            hi = min(len(grid_vals), idx + radius + 1)
            neighbors = grid_vals[lo:hi]

            # Also interpolate between neighbors for finer resolution
            interpolated = set(neighbors)
            for i in range(len(neighbors) - 1):
                mid = (neighbors[i] + neighbors[i + 1]) / 2
                # Round appropriately
                if isinstance(grid_vals[0], int):
                    mid = int(round(mid))
                else:
                    mid = round(mid, 2)
                interpolated.add(mid)

            fine_values[key] = sorted(interpolated)

        # Enumerate all combos in the fine grid
        fine_combos = [
            dict(zip(keys, combo))
            for combo in itertools.product(*[fine_values[k] for k in keys])
        ]

        return fine_combos

    # --- Random search phase ---

    def run_random_search(self, iterations: int = 500) -> list[OptimizationResult]:
        """
        Phase 1: Random search over the parameter space.
        Returns results sorted by Sharpe ratio (descending).
        """
        full_size = self._full_grid_size()
        actual_iters = min(iterations, full_size)

        print(f"\n{'='*70}")
        print(f"PHASE 1: RANDOM SEARCH")
        print(f"{'='*70}")
        print(f"  Strategy:     {self.strategy_name}")
        print(f"  Pairs:        {', '.join(self.pairs)}")
        print(f"  Timeframe:    {self.timeframe}")
        print(f"  Full grid:    {full_size:,} combinations")
        print(f"  Sampling:     {actual_iters:,} combinations")
        print(f"  Workers:      {self.n_workers}")
        print()

        combos = self._sample_random_combos(actual_iters)

        # Build work items for multiprocessing
        # Serialize candle data as lists since Candle objects may not pickle cleanly
        work_items = [
            (self.strategy_name, params, self.data_by_pair, self.engine_kwargs)
            for params in combos
        ]

        results: list[OptimizationResult] = []
        start_time = time.time()

        # Run in parallel
        completed = 0
        with Pool(processes=self.n_workers) as pool:
            for raw in pool.imap_unordered(_run_single_combo, work_items, chunksize=4):
                completed += 1
                if completed % 50 == 0 or completed == actual_iters:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (actual_iters - completed) / rate if rate > 0 else 0
                    print(
                        f"  Progress: {completed}/{actual_iters} "
                        f"({completed/actual_iters*100:.0f}%) "
                        f"| {rate:.1f} combos/sec "
                        f"| ETA {eta:.0f}s"
                    )

                if raw is None:
                    continue

                opt = OptimizationResult(
                    params=raw["params"],
                    strategy_name=raw["strategy_name"],
                    sharpe_ratio=raw["sharpe_ratio"],
                    total_return_pct=raw["total_return_pct"],
                    profit_factor=raw["profit_factor"],
                    win_rate=raw["win_rate"],
                    total_trades=raw["total_trades"],
                    max_drawdown_pct=raw["max_drawdown_pct"],
                    expectancy=raw["expectancy"],
                    pair_results=raw["pair_results"],
                )
                results.append(opt)

        elapsed = time.time() - start_time
        print(f"\n  Completed {completed} combos in {elapsed:.1f}s")
        print(f"  Valid results: {len(results)}")

        # Sort by Sharpe
        results.sort(key=lambda r: r.sharpe_ratio, reverse=True)
        return results

    # --- Fine-tuning phase ---

    def run_fine_tuning(self, top_results: list[OptimizationResult], top_n: int = 5) -> list[OptimizationResult]:
        """
        Phase 2: Fine-tune around the top N parameter sets from random search.
        """
        if not top_results:
            print("\n  No results to fine-tune.")
            return []

        candidates = top_results[:top_n]

        print(f"\n{'='*70}")
        print(f"PHASE 2: FINE-TUNING (top {top_n} from random search)")
        print(f"{'='*70}")

        all_fine_combos = []
        seen = set()
        for res in candidates:
            fine = self._generate_fine_grid(res.params)
            for combo in fine:
                key = tuple(sorted(combo.items()))
                if key not in seen:
                    seen.add(key)
                    all_fine_combos.append(combo)

        print(f"  Fine-tuning combos: {len(all_fine_combos)}")

        work_items = [
            (self.strategy_name, params, self.data_by_pair, self.engine_kwargs)
            for params in all_fine_combos
        ]

        results: list[OptimizationResult] = []
        start_time = time.time()
        total = len(all_fine_combos)
        completed = 0

        with Pool(processes=self.n_workers) as pool:
            for raw in pool.imap_unordered(_run_single_combo, work_items, chunksize=4):
                completed += 1
                if completed % 25 == 0 or completed == total:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    print(f"  Fine-tuning: {completed}/{total} ({completed/total*100:.0f}%)")

                if raw is None:
                    continue

                opt = OptimizationResult(
                    params=raw["params"],
                    strategy_name=raw["strategy_name"],
                    sharpe_ratio=raw["sharpe_ratio"],
                    total_return_pct=raw["total_return_pct"],
                    profit_factor=raw["profit_factor"],
                    win_rate=raw["win_rate"],
                    total_trades=raw["total_trades"],
                    max_drawdown_pct=raw["max_drawdown_pct"],
                    expectancy=raw["expectancy"],
                    pair_results=raw["pair_results"],
                )
                results.append(opt)

        results.sort(key=lambda r: r.sharpe_ratio, reverse=True)
        elapsed = time.time() - start_time
        print(f"  Fine-tuning done in {elapsed:.1f}s, {len(results)} valid results")
        return results

    # --- Walk-forward analysis ---

    def run_walk_forward(
        self,
        top_results: list[OptimizationResult],
        top_n: int = 10,
        n_windows: int = 5,
        train_pct: float = 0.70,
    ) -> list[dict]:
        """
        Phase 3: Walk-forward validation.
        Splits data into rolling train/test windows.
        For each window, re-optimizes on train and validates on test.

        A parameter set passes if profitable in >70% of OOS windows.
        """
        if not top_results:
            print("\n  No results for walk-forward.")
            return []

        print(f"\n{'='*70}")
        print(f"PHASE 3: WALK-FORWARD ANALYSIS")
        print(f"{'='*70}")
        print(f"  Windows:      {n_windows}")
        print(f"  Train/Test:   {train_pct*100:.0f}% / {(1-train_pct)*100:.0f}%")
        print(f"  Top params:   {top_n}")

        best_params_list = [r.params for r in top_results[:top_n]]

        # Create rolling windows for each pair
        windows = []
        for w in range(n_windows):
            train_data = {}
            test_data = {}

            for pair, candles in self.data_by_pair.items():
                if not candles:
                    continue
                n = len(candles)
                # Sliding window: each window uses a different portion
                window_size = int(n * 0.8)  # each window uses 80% of total data
                start = int(w * (n - window_size) / max(n_windows - 1, 1))
                end = start + window_size

                window_candles = candles[start:end]
                split = int(len(window_candles) * train_pct)

                train_data[pair] = window_candles[:split]
                test_data[pair] = window_candles[split:]

            windows.append((train_data, test_data))

        # Run each window
        work_items = [
            (self.strategy_name, best_params_list, train, test, self.engine_kwargs)
            for train, test in windows
        ]

        wf_results = []
        with Pool(processes=min(self.n_workers, n_windows)) as pool:
            for i, raw in enumerate(pool.imap(_run_walk_forward_window, work_items)):
                if raw:
                    wf_results.append(raw)
                    status = "PROFIT" if raw["oos_profitable"] else "LOSS"
                    print(
                        f"  Window {i+1}/{n_windows}: "
                        f"IS Sharpe={raw['is_sharpe']:.2f} | "
                        f"OOS Sharpe={raw['oos_sharpe']:.2f} | "
                        f"OOS Return={raw['oos_return_pct']:+.1f}% | "
                        f"{status}"
                    )
                else:
                    print(f"  Window {i+1}/{n_windows}: SKIPPED (no data/trades)")

        # Summary
        if wf_results:
            profitable_windows = sum(1 for r in wf_results if r["oos_profitable"])
            total_windows = len(wf_results)
            pass_rate = profitable_windows / total_windows * 100

            avg_oos_sharpe = float(np.mean([r["oos_sharpe"] for r in wf_results]))
            avg_oos_return = float(np.mean([r["oos_return_pct"] for r in wf_results]))

            print(f"\n  Walk-Forward Summary:")
            print(f"    OOS Profitable: {profitable_windows}/{total_windows} ({pass_rate:.0f}%)")
            print(f"    Avg OOS Sharpe: {avg_oos_sharpe:.2f}")
            print(f"    Avg OOS Return: {avg_oos_return:+.1f}%")

            if pass_rate >= 70:
                print(f"    PASSED (>= 70% OOS profitable)")
            else:
                print(f"    FAILED (< 70% OOS profitable — likely overfit)")

        return wf_results

    # --- Output ---

    def print_top_results(self, results: list[OptimizationResult], n: int = 10, label: str = ""):
        """Print the top N results with parameters."""
        if not results:
            print("\n  No results to display.")
            return

        header = f"TOP {min(n, len(results))} RESULTS"
        if label:
            header += f" ({label})"

        print(f"\n{'='*70}")
        print(header)
        print(f"{'='*70}")

        for i, res in enumerate(results[:n]):
            print(f"\n  --- Rank #{i+1} ---")
            print(f"  Sharpe:   {res.sharpe_ratio:.3f}")
            print(f"  Return:   {res.total_return_pct:+.2f}%")
            print(f"  PF:       {res.profit_factor:.2f}")
            print(f"  WR:       {res.win_rate:.1f}%")
            print(f"  Trades:   {res.total_trades}")
            print(f"  Max DD:   {res.max_drawdown_pct:.1f}%")
            print(f"  E[trade]: ${res.expectancy:.2f}")

            # Per-pair breakdown
            for pair, pr in res.pair_results.items():
                print(
                    f"    {pair}: Sharpe={pr['sharpe']:.2f} "
                    f"Return={pr['return_pct']:+.1f}% "
                    f"WR={pr['win_rate']:.0f}% "
                    f"Trades={pr['trades']}"
                )

            # Parameters
            print(f"  Parameters:")
            for k, v in sorted(res.params.items()):
                print(f"    {k}: {v}")

    def print_config_snippet(self, best: OptimizationResult):
        """Print the exact parameter values to paste into config/strategy."""
        print(f"\n{'='*70}")
        print(f"RECOMMENDED CONFIG for {self.strategy_name}")
        print(f"{'='*70}")
        print(f"# Paste into your strategy class or config override:")
        print(f"# Sharpe={best.sharpe_ratio:.3f} Return={best.total_return_pct:+.2f}% "
              f"WR={best.win_rate:.1f}% PF={best.profit_factor:.2f}")
        print()

        cls_name = STRATEGY_MAP[self.strategy_name].__name__
        print(f"# In {self.strategy_name}.py ({cls_name}):")
        for k, v in sorted(best.params.items()):
            if isinstance(v, float):
                print(f"    {k}: float = {v}")
            elif isinstance(v, int):
                print(f"    {k}: int = {v}")
            else:
                print(f"    {k} = {v!r}")

        print()
        print("# Or as a dict for programmatic override:")
        print(f"OPTIMIZED_{self.strategy_name.upper()} = {{")
        for k, v in sorted(best.params.items()):
            print(f'    "{k}": {v!r},')
        print("}")

    def save_results_csv(self, results: list[OptimizationResult], filename: str | None = None):
        """Save all results to a CSV file."""
        if not results:
            return

        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_{self.strategy_name}_{ts}.csv"

        output_dir = Path(__file__).parent.parent / "results"
        output_dir.mkdir(exist_ok=True)
        filepath = output_dir / filename

        # Build rows
        param_keys = sorted(self.param_grid.keys())
        fieldnames = (
            ["rank", "sharpe_ratio", "total_return_pct", "profit_factor",
             "win_rate", "total_trades", "max_drawdown_pct", "expectancy"]
            + param_keys
            + [f"{p}_sharpe" for p in self.pairs]
            + [f"{p}_return" for p in self.pairs]
        )

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for i, res in enumerate(results):
                row = {
                    "rank": i + 1,
                    "sharpe_ratio": round(res.sharpe_ratio, 4),
                    "total_return_pct": round(res.total_return_pct, 2),
                    "profit_factor": round(res.profit_factor, 2),
                    "win_rate": round(res.win_rate, 1),
                    "total_trades": res.total_trades,
                    "max_drawdown_pct": round(res.max_drawdown_pct, 1),
                    "expectancy": round(res.expectancy, 2),
                }
                for k in param_keys:
                    row[k] = res.params.get(k, "")
                for p in self.pairs:
                    pr = res.pair_results.get(p, {})
                    row[f"{p}_sharpe"] = round(pr.get("sharpe", 0), 3)
                    row[f"{p}_return"] = round(pr.get("return_pct", 0), 2)

                writer.writerow(row)

        print(f"\n  Results saved to: {filepath}")
        return filepath

    # --- Full pipeline ---

    def optimize(self, iterations: int = 500, fine_tune_top: int = 5, wf_windows: int = 5):
        """
        Run the full optimization pipeline:
        1. Download data
        2. Random search
        3. Fine-tuning around top results
        4. Walk-forward validation
        5. Print results and save CSV
        """
        total_start = time.time()

        # Step 1: Download data
        self.download_data()
        if not any(self.data_by_pair.values()):
            print("\nERROR: No data downloaded. Check OANDA credentials and connection.")
            return

        # Step 2: Random search
        random_results = self.run_random_search(iterations)
        if not random_results:
            print("\nERROR: No valid results from random search.")
            return

        self.print_top_results(random_results, n=10, label="Random Search")

        # Step 3: Fine-tuning
        fine_results = self.run_fine_tuning(random_results, top_n=fine_tune_top)

        # Merge and re-sort all results
        all_results = random_results + fine_results
        # Deduplicate by params
        seen_params = set()
        unique_results = []
        for r in all_results:
            key = tuple(sorted(r.params.items()))
            if key not in seen_params:
                seen_params.add(key)
                unique_results.append(r)
        unique_results.sort(key=lambda r: r.sharpe_ratio, reverse=True)

        if fine_results:
            self.print_top_results(unique_results, n=10, label="After Fine-Tuning")

        # Step 4: Walk-forward
        wf_results = self.run_walk_forward(unique_results, top_n=10, n_windows=wf_windows)

        # Step 5: Output
        if unique_results:
            self.print_config_snippet(unique_results[0])

        # In-sample vs out-of-sample comparison
        if wf_results and unique_results:
            print(f"\n{'='*70}")
            print(f"IN-SAMPLE vs OUT-OF-SAMPLE")
            print(f"{'='*70}")
            best = unique_results[0]
            avg_oos_sharpe = float(np.mean([r["oos_sharpe"] for r in wf_results]))
            avg_oos_return = float(np.mean([r["oos_return_pct"] for r in wf_results]))
            print(f"  In-Sample  Sharpe: {best.sharpe_ratio:.3f}  Return: {best.total_return_pct:+.2f}%")
            print(f"  Out-of-Sample Sharpe: {avg_oos_sharpe:.3f}  Return: {avg_oos_return:+.2f}%")

            sharpe_decay = (1 - avg_oos_sharpe / best.sharpe_ratio) * 100 if best.sharpe_ratio != 0 else 0
            print(f"  Sharpe Decay: {sharpe_decay:.0f}%")
            if sharpe_decay > 50:
                print(f"  WARNING: >50% Sharpe decay suggests overfitting")
            elif sharpe_decay > 30:
                print(f"  CAUTION: 30-50% Sharpe decay — moderate overfitting risk")
            else:
                print(f"  OK: <30% Sharpe decay — parameters appear robust")

        # Save CSV
        csv_path = self.save_results_csv(unique_results)

        total_elapsed = time.time() - total_start
        print(f"\n{'='*70}")
        print(f"OPTIMIZATION COMPLETE in {total_elapsed:.0f}s")
        print(f"{'='*70}")

        return unique_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Strategy parameter optimizer (grid search + walk-forward)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m backtesting.optimizer --strategy mean_reversion --iterations 500
  python -m backtesting.optimizer --strategy trend_following --pair EUR_USD --iterations 300
  python -m backtesting.optimizer --strategy london_breakout --timeframe M5 --iterations 200
  python -m backtesting.optimizer --all --iterations 500
        """,
    )
    parser.add_argument(
        "--strategy", type=str, default=None,
        help=f"Strategy to optimize: {list(STRATEGY_MAP.keys())}",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Optimize all strategies sequentially",
    )
    parser.add_argument(
        "--pair", type=str, default=None,
        help="Single pair to optimize on (default: EUR_USD, GBP_USD, USD_JPY)",
    )
    parser.add_argument(
        "--timeframe", type=str, default="H1",
        help="Candle timeframe (default: H1)",
    )
    parser.add_argument(
        "--iterations", type=int, default=500,
        help="Number of random search iterations (default: 500)",
    )
    parser.add_argument(
        "--candles", type=int, default=5000,
        help="Number of historical candles to fetch (default: 5000)",
    )
    parser.add_argument(
        "--capital", type=float, default=10000.0,
        help="Starting capital for backtest (default: 10000)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help=f"Number of parallel workers (default: {max(1, cpu_count()-1)})",
    )
    parser.add_argument(
        "--wf-windows", type=int, default=5,
        help="Walk-forward window count (default: 5)",
    )
    parser.add_argument(
        "--fine-tune-top", type=int, default=5,
        help="Fine-tune around top N results (default: 5)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.strategy and not args.all:
        parser.error("Specify --strategy NAME or --all")

    pairs = [args.pair] if args.pair else DEFAULT_PAIRS

    strategies = list(STRATEGY_MAP.keys()) if args.all else [args.strategy]

    for strat_name in strategies:
        if strat_name not in STRATEGY_MAP:
            print(f"ERROR: Unknown strategy '{strat_name}'. Available: {list(STRATEGY_MAP.keys())}")
            continue

        # Use M5 by default for london_breakout since it runs on M5 candles
        timeframe = args.timeframe
        if strat_name == "london_breakout" and timeframe == "H1":
            timeframe = "M5"
            print(f"NOTE: London breakout uses M5 candles, switching timeframe to M5")

        optimizer = StrategyOptimizer(
            strategy_name=strat_name,
            pairs=pairs,
            timeframe=timeframe,
            candle_count=args.candles,
            initial_capital=args.capital,
            n_workers=args.workers,
        )

        optimizer.optimize(
            iterations=args.iterations,
            fine_tune_top=args.fine_tune_top,
            wf_windows=args.wf_windows,
        )

        print("\n")


if __name__ == "__main__":
    main()
