# Forex Trading Bot

Automated FX trading system on OANDA, Python 3. Multi-strategy signal generation with layered risk management, live + demo-account support, Telegram monitoring, and a vectorised backtest harness for walk-forward validation.

> **Personal-capital tool, not a product. Use a demo account first. Most retail FX bots lose money — this one might too. Evidence-based development only.**

---

## Current State (2026-04-21)

**Observation mode** — bots running on demo + live with TrendFilter and PairGuard disabled. Collecting real-trade data for analysis at ~2026-06-02. No new strategies deploy during this window without passing walk-forward backtest (`backtesting/run_ewmac_bt.py`).

- **Demo** (OANDA practice, $98K): 7 strategies unblocked, risk manager fully on
- **Live** ($94): same config, same strategies, lower capital
- **Evidence on file**: EWMAC trend following on pure FX majors 2020-2026 fails every pass criterion — do not re-attempt without expanding the asset universe

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           MAIN EVENT LOOP                                │
│                  (poll candles every 5s, heartbeat ~60s)                 │
└──────────────────────────────────────────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────────────────┐
│                         DATA LAYER                                       │
│                                                                          │
│   OANDA v20 REST ──► CandleManager ──► rolling buffers per pair × TF     │
│    (retry + CB)         │                (M5/M15/H1/H4/D)                │
│                         ▼                                                │
│                   emits CANDLE_CLOSE events                              │
│                                                                          │
│   Sentiment:  Yahoo DXY, OANDA retail positioning, COT                   │
│   EconCal:    ForexFactory high-impact events (± 30min no-trade window)  │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────────────────┐
│                       STRATEGY LAYER                                     │
│                                                                          │
│   Current architecture (emit Signal objects):                            │
│     mean_reversion   trend_following   london_breakout   confluence      │
│     session_momentum   stat_arb   range_scalp                            │
│                                                                          │
│   New architecture — STAGED, NOT WIRED (emit Forecast objects):          │
│     ewmac (D1 EWMAC crossovers)   carry (OANDA swap rates)               │
│     → portfolio/vol_target.py (forecast combination + 20%/yr vol target) │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
                             Signal events
                                   │
┌──────────────────────────────────▼───────────────────────────────────────┐
│                          GATE LAYER                                      │
│                                                                          │
│   1. PairGuard          ──  block pairs with 3 consecutive losses        │
│                             (DISABLED via DISABLE_PAIR_GUARD=true)       │
│                                                                          │
│   2. GlobalTrendFilter  ──  block counter-trend / FLAT-consensus         │
│                             (DISABLED via DISABLE_TREND_FILTER=true)     │
│                                                                          │
│   3. RiskManager        ──  ALWAYS ON — 10 rules:                        │
│                             • SL required, min R:R 1.5                   │
│                             • max 6 open positions, correlation cap      │
│                             • same-currency bucket cap                   │
│                             • daily profit lock + equity trail           │
│                             • spread filter                              │
│                             • drawdown halts (warn 5%, crit 8%, halt 10%)│
│                             • daily loss limit (3%)                      │
│                             • Friday size reduction                      │
│                             • news event blackout                        │
│                             • stop-loss distance cap                     │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
                          Approved orders + units
                                   │
┌──────────────────────────────────▼───────────────────────────────────────┐
│                       EXECUTION LAYER                                    │
│                                                                          │
│   OandaBroker.place_order  ──►  OANDA v20 REST  ──►  FillEvent           │
│         │                                                                │
│         └──► ProfitManager (trailing stops, partial closes at stages)    │
│                  • stage 1: +10 pips → close 25%, lock +3 pips           │
│                  • stage 2: +20 pips → close 25%, trail 10 pips          │
│                  • stage 3: +30 pips → close 25%, trail 7 pips           │
│                                                                          │
│   BookItRule: close profitable trades at London close                    │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────────────────┐
│                    PERSISTENCE + MONITORING                              │
│                                                                          │
│   SQLite DB:  orders, positions, equity snapshots                        │
│   Telegram:   trade open/close, daily digest, /status /kill commands     │
│   Logs:       rotated, 4 files × 2MB                                     │
└──────────────────────────────────────────────────────────────────────────┘
```

### Signal lifecycle

```
  CANDLE_CLOSE
       │
       ▼
  Strategy.on_candle(instrument, timeframe, candles)
       │
   emits Signal(side, entry, stop_loss, take_profit, metadata)
       │
       ▼
  bus.emit(SIGNAL) ──► _on_signal handler
       │
       ├─► PairGuard.is_blocked?          ─ yes ─► DROP
       ├─► TrendFilter.filter_signal?     ─ no  ─► DROP
       ├─► RiskManager.evaluate_signal?   ─ no  ─► DROP
       │                                  ─ yes ─► (units)
       ▼
  Executor.execute_signal ──► Broker.place_order ──► OANDA
       │                                                │
       ▼                                                ▼
  DB.save_order                              Order FILLED event
                                                        │
                                                        ▼
                                              Position OPENED
                                                        │
                                             Monitored by ProfitManager
```

---

## Strategies

### Active (emit `Signal` objects)
| Name | TF | Logic | Notes |
|---|---|---|---|
| `mean_reversion` | M15 | BB touch + RSI extreme | Tuned for ranging markets, ADX < 35 |
| `trend_following` | H1 | EMA15/50 cross + ADX | ADX ≥ 25, daily EMA200 trend filter |
| `london_breakout` | M5 | Break of 07:00-07:30 UTC range | Range 15-50 pips, max 1/pair/session |
| `confluence` | H1 | 7-indicator score | Score ≥ 6/7 |
| `session_momentum` | M15 | Asian range breakout + RSI/ADX | Max 1 London + 1 NY /day |
| `stat_arb` | H1 | EUR/GBP z-score spread | z > 2 entry, z < 0.5 exit |
| `range_scalp` | M15 | BB touch + RSI extreme in ADX 15-25 | 25% risk, quarter-Kelly sizing |

### Scaffolded (emit `Forecast` objects, NOT YET WIRED)
| Name | TF | Evidence | Status |
|---|---|---|---|
| `ewmac` | D1 | Peer-reviewed TSMOM (MOP 2012, AQR) | **Backtest FAILED** on FX 2020-2026 — do not deploy |
| `carry` | D1 | Peer-reviewed (Burnside, Daniel/Hodrick/Lu) | Cannot backtest — no historical rate data |

---

## Risk Management (always on)

| Rule | Threshold |
|---|---|
| Per-trade risk | 0.5% of equity |
| Min R:R | 1.5 |
| Max open positions | 6 |
| Max same-currency exposure | 2 positions |
| Daily loss limit | 3% → halt |
| Drawdown warn / crit / halt | 5% / 8% / 10% |
| Daily profit lock | +2% → halt new entries |
| Intraday trail | -1.5% from peak → halt new entries |
| Max stop distance | 50 pips |
| News blackout | ±30 min high-impact |
| Friday size reduction | 50% after 18:00 UTC |

---

## Data Sources

- **OANDA v20 REST** — candles, pricing, account, instruments, financing rates
- **OANDA streaming** — price ticks (optional, not wired in main loop today)
- **Yahoo Finance** — DXY proxy for USD strength (FLAT fallback to basket)
- **OANDA orderBook endpoint** — retail positioning (401 on live, gracefully demoted)
- **ForexFactory** — high-impact news calendar

---

## Project Structure

```
forex-trading-bot/
├── main.py                       # Orchestrator + event loop + gate wiring
├── models.py                     # Tick, Candle, Signal, Order, Position
├── event_bus.py                  # pubsub events
├── monitor.py                    # CLI dashboard + health checks
├── CLAUDE.md                     # Project guide for Claude Code sessions
│
├── config/
│   └── __init__.py               # Pydantic settings (broker, risk, toggles)
│
├── strategies/                   # Trading rules
│   ├── base.py                   # Strategy ABC + candles_to_df
│   ├── mean_reversion.py
│   ├── trend_following.py
│   ├── london_breakout.py
│   ├── confluence.py
│   ├── session_momentum.py
│   ├── stat_arb.py
│   ├── range_scalp.py            # NEW 2026-04-20
│   ├── ewmac.py                  # STAGED — forecast-based
│   └── carry.py                  # STAGED — forecast-based
│
├── portfolio/                    # Portfolio-level sizing (new architecture)
│   └── vol_target.py             # Forecast combination + vol targeting
│
├── execution/
│   ├── broker.py                 # OANDA v20 adapter (with retry + CB)
│   ├── order_executor.py         # Signal → Order
│   └── trailing_stop.py          # ProfitManager staged trail + partial close
│
├── risk/
│   ├── risk_manager.py           # 10 rules, position sizing, kill switch
│   └── pair_guard.py             # Per-pair auto-block (currently disabled)
│
├── data/
│   ├── candle_manager.py         # Rolling buffers + circuit breaker
│   ├── trend_filter.py           # Global trend/regime filter (disabled)
│   ├── economic_calendar.py      # ForexFactory news
│   └── sentiment.py              # DXY, retail positioning, COT
│
├── monitoring/
│   └── telegram_bot.py           # Alerts, daily reports, commands
│
├── backtesting/
│   ├── engine.py                 # Signal-based backtester (legacy)
│   ├── data_loader.py            # OANDA → CSV cache
│   ├── forecast_engine.py        # NEW: vectorised forecast-based sim
│   ├── run_ewmac_bt.py           # NEW: CLI walk-forward runner
│   ├── optimizer.py              # Parameter grid search
│   └── run_backtest.py           # Legacy CLI
│
├── db/
│   ├── database.py               # SQLite: orders, positions, equity, candles
│   ├── trades.db                 # demo trades (gitignored)
│   ├── trades_live.db            # live trades (gitignored)
│   └── hist_cache/               # Backtest data CSV (gitignored)
│
├── tests/                        # pytest tests
├── .env.example                  # Config template
├── .env / .env.live              # Real configs (gitignored)
├── trading-bot.service           # Systemd unit (demo)
├── trading-bot-live.service      # Systemd unit (live)
└── README.md
```

---

## Quick Start

```bash
# 1. Clone + install
git clone https://github.com/bek01/forex-trading-bot.git
cd forex-trading-bot
python3 -m venv ~/venv
~/venv/bin/pip install -e .

# 2. Configure
cp .env.example .env
# Edit .env with OANDA_ACCOUNT_ID, OANDA_API_TOKEN, OANDA_ENVIRONMENT=practice

# 3. Run backtest FIRST
~/venv/bin/python3 -m backtesting.run_ewmac_bt --years 6

# 4. Paper trade
~/venv/bin/python3 main.py

# 5. Deploy as service
sudo cp trading-bot.service /etc/systemd/system/
sudo systemctl enable --now trading-bot
journalctl -u trading-bot -f
```

---

## Backtest Harness

Two backtest paths:

### Signal-based (legacy, for current architecture)
```bash
python3 -m backtesting.run_backtest --instrument EUR_USD --years 3
```

### Forecast-based (new, walk-forward-capable)
```bash
python3 -m backtesting.run_ewmac_bt --years 6               # full-period
python3 -m backtesting.run_ewmac_bt --walk-forward          # OOS rolling windows
```

**Pass criteria** (from `memory/reference_fx_architecture.md`):
- OOS Sharpe > 0.6
- ≥ 100 OOS trades
- Max DD > −25%
- Stable across 3 sub-periods

EWMAC on FX-only failed on all three. See `memory/ewmac_backtest_failed_2026_04_21.md`.

---

## Config toggles

Environment variables in `.env` / `.env.live`:

```
TRADING_MODE=paper|live
OANDA_ENVIRONMENT=practice|live
DISABLE_TREND_FILTER=true|false      # skip global trend/regime filter
DISABLE_PAIR_GUARD=true|false        # skip per-pair auto-block
RISK_MAX_RISK_PER_TRADE_PCT=0.5
RISK_MAX_OPEN_POSITIONS=6
RISK_DAILY_LOSS_LIMIT_PCT=3.0
```

---

## Why OANDA

OANDA's v20 API is the most developer-friendly retail FX broker for algo trading:
- REST + streaming both documented
- Practice (demo) and live on same API shape, just different URLs
- No commission (spreads only)
- Reasonable fill quality for daily-timeframe strategies
- Known weakness: short-term scalping is cost-prohibitive vs prime brokers

---

## What actually works in retail FX (evidence base)

Only two strategy families have peer-reviewed multi-decade evidence:
1. **Trend-following on daily+ bars** across wide asset universes (Moskowitz/Ooi/Pedersen 2012, AQR Century of Evidence)
2. **Currency carry** (Daniel/Hodrick/Lu, Burnside/Rebelo)

Both have Sharpe 0.4-1.0 with severe tail risk. **FX-only subsets are weaker than multi-asset** — our EWMAC backtest on 8 FX majors 2020-2026 confirmed no edge. Reference: `memory/reference_fx_architecture.md`.

Anti-patterns that don't work (all present in this codebase historically):
- Indicator stacking (RSI + MACD + BB + ADX scoring)
- Global regime filters bolted on to rescue broken strategies
- Reactive parameter tuning after each bad day
- Over-diversification across 20+ correlated pairs
- "Optimized" results on 30-100 trades (curve-fit)

---

## Telegram Commands

```
/status      Current equity, DD, open positions
/trades      Recent trades
/kill        Immediate halt
/resume      Resume after halt
/help
```

---

## Disclaimer

This is a research and personal-capital tool. Past backtest performance does not guarantee future results. Most retail FX bots lose money. Do not deploy live without a demonstrated walk-forward edge. Losses are yours to bear.
