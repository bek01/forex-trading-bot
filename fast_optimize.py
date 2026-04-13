#!/usr/bin/env python3
"""Fast parameter optimizer — runs 10 combos per strategy on 2000 candles."""
import warnings, random, gc, sys
warnings.filterwarnings('ignore')

from backtesting.engine import BacktestEngine
from execution.broker import OandaBroker
from config import get_config
from strategies.mean_reversion import MeanReversionStrategy
from strategies.trend_following import TrendFollowingStrategy
from strategies.confluence import ConfluenceStrategy

cfg = get_config()
broker = OandaBroker(cfg.broker)

print('Downloading...', flush=True)
m15 = [c for c in broker.get_candles('EUR_USD', 'M15', 2000) if c.complete]
h1  = [c for c in broker.get_candles('EUR_USD', 'H1', 2000) if c.complete]
broker.close()
print(f'M15:{len(m15)} H1:{len(h1)}', flush=True)

engine = BacktestEngine(initial_capital=10000, risk_per_trade_pct=1.0, spread_pips=1.5)

# MEAN REVERSION
print('\n=== MEAN REVERSION (10 combos) ===', flush=True)
results = []
for i in range(10):
    s = MeanReversionStrategy()
    s.bb_period = random.choice([15,20,25,30])
    s.bb_std = random.choice([1.5,2.0,2.5,3.0])
    s.rsi_period = random.choice([7,10,14,21])
    s.rsi_oversold = random.choice([20,25,30,35])
    s.rsi_overbought = random.choice([65,70,75,80])
    s.adx_max = random.choice([20,25,30,35,40])
    s.sl_atr_multiplier = random.choice([1.0,1.5,2.0,2.5])
    r = engine.run(s, m15, 'EUR_USD')
    p = f'bb={s.bb_period}/{s.bb_std} rsi={s.rsi_period}/{s.rsi_oversold}/{s.rsi_overbought} adx<{s.adx_max} sl={s.sl_atr_multiplier}'
    results.append((r.sharpe_ratio, r.win_rate, r.profit_factor, r.total_trades, r.total_return_pct, p))
    print(f'  {i+1}/10: trades={r.total_trades} wr={r.win_rate:.0f}% sharpe={r.sharpe_ratio:.2f} ret={r.total_return_pct:+.1f}%', flush=True)
    gc.collect()

results.sort(reverse=True)
print('\nTop 5:', flush=True)
for i,(sh,wr,pf,tr,ret,p) in enumerate(results[:5]):
    print(f'  #{i+1} Sharpe={sh:.2f} WR={wr:.0f}% PF={pf:.2f} Trades={tr} Ret={ret:+.1f}% | {p}', flush=True)

# TREND FOLLOWING
print('\n=== TREND FOLLOWING (10 combos) ===', flush=True)
results2 = []
for i in range(10):
    s = TrendFollowingStrategy()
    s.fast_ema = random.choice([8,10,15,20,25])
    s.slow_ema = random.choice([30,40,50,60,80])
    s.adx_min = random.choice([15,20,25,30])
    s.sl_atr_multiplier = random.choice([1.0,1.5,2.0,2.5,3.0])
    s.tp_atr_multiplier = random.choice([2.0,2.5,3.0,4.0,5.0])
    r = engine.run(s, h1, 'EUR_USD')
    p = f'ema={s.fast_ema}/{s.slow_ema} adx>{s.adx_min} sl={s.sl_atr_multiplier} tp={s.tp_atr_multiplier}'
    results2.append((r.sharpe_ratio, r.win_rate, r.profit_factor, r.total_trades, r.total_return_pct, p))
    print(f'  {i+1}/10: trades={r.total_trades} wr={r.win_rate:.0f}% sharpe={r.sharpe_ratio:.2f} ret={r.total_return_pct:+.1f}%', flush=True)
    gc.collect()

results2.sort(reverse=True)
print('\nTop 5:', flush=True)
for i,(sh,wr,pf,tr,ret,p) in enumerate(results2[:5]):
    print(f'  #{i+1} Sharpe={sh:.2f} WR={wr:.0f}% PF={pf:.2f} Trades={tr} Ret={ret:+.1f}% | {p}', flush=True)

# CONFLUENCE
print('\n=== CONFLUENCE (10 combos) ===', flush=True)
results3 = []
for i in range(10):
    s = ConfluenceStrategy()
    s.score_threshold = random.choice([3,4,5,6])
    s.ema_fast = random.choice([10,15,20,25])
    s.ema_slow = random.choice([40,50,60])
    s.rsi_period = random.choice([10,14,21])
    s.adx_min = random.choice([15,20,25])
    s.sl_atr_multiplier = random.choice([1.0,1.5,2.0,2.5])
    s.tp_atr_multiplier = random.choice([2.0,2.5,3.0,4.0])
    r = engine.run(s, h1, 'EUR_USD')
    p = f'score>={s.score_threshold} ema={s.ema_fast}/{s.ema_slow} rsi={s.rsi_period} adx>{s.adx_min} sl={s.sl_atr_multiplier} tp={s.tp_atr_multiplier}'
    results3.append((r.sharpe_ratio, r.win_rate, r.profit_factor, r.total_trades, r.total_return_pct, p))
    print(f'  {i+1}/10: trades={r.total_trades} wr={r.win_rate:.0f}% sharpe={r.sharpe_ratio:.2f} ret={r.total_return_pct:+.1f}%', flush=True)
    gc.collect()

results3.sort(reverse=True)
print('\nTop 5:', flush=True)
for i,(sh,wr,pf,tr,ret,p) in enumerate(results3[:5]):
    print(f'  #{i+1} Sharpe={sh:.2f} WR={wr:.0f}% PF={pf:.2f} Trades={tr} Ret={ret:+.1f}% | {p}', flush=True)

print('\nDONE', flush=True)
