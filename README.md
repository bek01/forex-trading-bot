# 🤖 Forex Trading Bot

A personal automated trading system for Forex markets. Connects to OANDA's API, executes trades autonomously based on 6 configurable strategies, with comprehensive risk management, backtesting, parameter optimization, and real-time monitoring.

> **⚠️ This is a personal capital tool, not a consumer product. Use at your own risk. Start with a demo account.**

---

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Main Event Loop                                │
│                       (asyncio, 24/5 runtime)                           │
├──────────┬───────────┬──────────┬───────────┬──────────┬────────────────┤
│  Market  │ Strategy  │   Risk   │  Order    │  Data    │  Monitoring    │
│  Data    │ Engine    │  Manager │  Executor │  Sources │  & Alerts      │
│  Feed    │           │          │           │          │                │
├──────────┼───────────┼──────────┼───────────┼──────────┼────────────────┤
│ OANDA    │ 6 Active  │ 10 Risk  │ Paper/    │ Econ     │ Telegram Bot   │
│ Streaming│ Strategies│ Rules    │ Live Mode │ Calendar │ /kill /status  │
│ Candle   │ Scoring   │ Position │ Retry     │ CFTC COT │ Equity Curve  │
│ Builder  │ Multi-TF  │ Sizing   │ State     │ DXY      │ Daily Reports │
│ Tick     │ Session   │ Drawdown │ Recovery  │ Sentiment│ Heartbeat     │
│ Buffer   │ Stat Arb  │ Circuit  │           │          │               │
│          │           │ Breaker  │           │          │               │
└──────────┴───────────┴──────────┴───────────┴──────────┴────────────────┘
                                    ↓
                           ┌────────────────┐
                           │   SQLite DB    │
                           │ trades, candles│
                           │ equity, config │
                           └────────────────┘
```

### Event Flow

```
Price Tick/Candle Close
       ↓
Strategy.on_candle() → Signal (BUY/SELL + SL + TP)
       ↓
RiskManager.evaluate_signal()
  ├── Stop loss required? ✓
  ├── R:R ratio >= 1.5?  ✓
  ├── Max positions OK?  ✓
  ├── Correlation check? ✓
  ├── Spread < 2x avg?   ✓
  ├── Drawdown < 10%?    ✓
  ├── Daily loss < 2%?   ✓
  ├── News filter clear?  ✓
  └── Position size calc  → units
       ↓
OrderExecutor → Broker API → FILL
       ↓
EventBus: POSITION_OPENED → DB + Telegram
```

---

## 🏦 Why OANDA? — Broker Comparison

| Feature | **OANDA** ⭐ | Interactive Brokers | Alpaca | MetaTrader 5 | IG Markets |
|---------|-------------|--------------------|---------|--------------|-----------| 
| **API Quality** | ★★★★★ REST + Stream, excellent docs | ★★★☆☆ TWS API, Java-dependent, complex | ★★★★☆ Clean REST, stocks only | ★★★☆☆ Python bridge, Windows-only | ★★★★☆ REST, decent docs |
| **Python SDK** | `oandapyV20` — mature, well-maintained | `ib_insync` — works but fragile | `alpaca-trade-api` — excellent | `MetaTrader5` — Windows DLL, no Linux | `ig-markets-api` — community |
| **Paper Trading** | ✅ Free, identical API, unlimited | ✅ Free but complex setup | ✅ Free, same API | ✅ Free demo account | ✅ Free demo |
| **Forex Pairs** | ✅ 70+ pairs | ✅ 100+ pairs | ❌ No forex | ✅ Broker-dependent | ✅ 80+ pairs |
| **US Stocks** | ❌ No stocks | ✅ All US + global | ✅ All US, commission-free | ✅ Via broker CFDs | ✅ CFDs only |
| **Spreads (EUR/USD)** | 1.0-1.4 pips | 0.1-0.5 pips (best) | N/A | Broker-dependent | 0.6-1.0 pips |
| **Commission** | $0 (spread-only) | $2/trade + spread | $0 stocks | Broker-dependent | $0 (spread-only) |
| **Min Deposit** | $0 demo, $0 live | $0 but $10K+ practical | $0 | Broker-dependent | $250 |
| **Streaming API** | ✅ HTTP streaming | ✅ Socket-based | ✅ WebSocket | ❌ Must poll | ✅ Lightstreamer |
| **Execution Speed** | ~50ms | ~10ms (fastest) | ~100ms | Broker-dependent | ~50ms |
| **Linux/VPS** | ✅ Full support | ✅ Needs IB Gateway | ✅ Full support | ❌ Windows only | ✅ Full support |
| **Reliability** | ★★★★★ | ★★★★☆ TWS disconnects | ★★★★☆ | ★★★☆☆ | ★★★★☆ |
| **Rate Limits** | 120 req/sec | 50 msg/sec | 200 req/min | N/A | 60 req/min |
| **Best For** | **Solo algo dev** | Institutional/pro | US stock algos | Manual + EA | Spread betting UK |

### Why OANDA Won

1. **Best developer experience** — clean REST API, excellent docs, no Java dependency
2. **Identical paper/live API** — switch with one config change, no code changes
3. **Linux/VPS native** — runs perfectly on headless servers (MT5 requires Windows)
4. **Zero minimum deposit** — start testing with any amount
5. **Streaming + REST** — real-time prices via HTTP streaming, orders via REST
6. **Reliable** — 15+ years of API stability, used by thousands of algo traders

### When to Choose Others

- **Interactive Brokers**: If you need the absolute tightest spreads and trade >$50K
- **Alpaca**: If you want US stocks, not forex
- **MT5**: If you want the MQL5 ecosystem and manual trading alongside algos
- **IG Markets**: If you're UK-based and want spread betting tax advantages

---

## 📈 Strategies (6 Active)

### 1. Mean Reversion (`mean_reversion`)
- **Logic**: Bollinger Band(20,2) touch + RSI(14) oversold/overbought + ADX < 25 (ranging market)
- **Timeframe**: M15
- **Target**: Mid BB (mean reversion to center)
- **Best session**: Asian (low volatility, range-bound)

### 2. Trend Following (`trend_following`)
- **Logic**: EMA(20) crosses EMA(50) + ADX > 25 + Daily EMA(200) trend filter
- **Timeframe**: H1 entry, D for trend
- **Target**: 3x ATR from entry
- **Best session**: London/NY overlap

### 3. London Breakout (`london_breakout`)
- **Logic**: First 30min of London session (07:00-07:30 UTC) establishes range, trade breakout
- **Timeframe**: M5
- **Target**: 1.5x range size
- **Pairs**: EUR_USD, GBP_USD, EUR_GBP only

### 4. Multi-Indicator Confluence (`confluence`) ⭐ Highest Expected WR
- **Logic**: 7-point scoring system — each confirming indicator adds +1
  - Price > EMA(200), EMA(20) > EMA(50), RSI 30-60, MACD histogram positive
  - ADX > 20, near Bollinger Band, volume above average
- **Entry**: Score >= 4 out of 7
- **Timeframe**: H1 entry, D for trend
- **Why**: Multi-indicator confluence is the most validated approach in open-source backtests

### 5. Session Momentum (`session_momentum`)
- **Logic**: Track Asian session range → London breakout + momentum confirmation → NY continuation
- **Timeframe**: M15
- **Max 2 trades/day**: 1 London window, 1 NY window
- **Pairs**: EUR_USD, GBP_USD

### 6. Statistical Arbitrage (`stat_arb`)
- **Logic**: Trade mean reversion of EUR_USD / GBP_USD price ratio spread
- **Entry**: Z-score > 2.0 (sell overpriced, buy underpriced)
- **Exit**: Z-score < 0.5 (spread reverted)
- **Market-neutral**: Always 1 long + 1 short position

---

## 🛡️ Risk Management (10 Enforced Rules)

| # | Rule | Default | Notes |
|---|------|---------|-------|
| 1 | **Stop loss required** | Mandatory | Code rejects orders without SL. Non-negotiable. |
| 2 | **Max risk per trade** | 1% of equity | ATR-based position sizing with hard USD cap |
| 3 | **Min R:R ratio** | 1.5:1 | No low-quality trades |
| 4 | **Max open positions** | 3 | Prevents overexposure |
| 5 | **Correlation check** | 1.5x weighting | EUR/USD + GBP/USD = 1.5 effective positions |
| 6 | **Daily loss limit** | -2% | Auto-halt trading for the day |
| 7 | **Max drawdown** | -10% from peak | Kill switch: close everything, alert |
| 8 | **Spread filter** | 2x average | Skip when spreads are elevated |
| 9 | **News filter** | 30min buffer | No trades near high-impact events (via ForexFactory) |
| 10 | **Friday reduction** | 50% size | Reduce exposure before weekend gap risk |

### Drawdown Circuit Breaker

```
Equity vs Peak
  ├── -5%  → WARNING: reduce size 25%, alert via Telegram
  ├── -8%  → CRITICAL: reduce size 50%, alert
  └── -10% → HALT: close ALL positions, kill switch, immediate alert
```

---

## 📡 Data Sources

| Source | Data | Update Freq | Auth |
|--------|------|------------|------|
| **OANDA API** | Price ticks, candles, order book | Real-time streaming | API key (configured) |
| **ForexFactory** | Economic calendar, high-impact events | Hourly refresh | None (free) |
| **CFTC COT** | Institutional positioning | Weekly (Fri) | None (free) |
| **Yahoo Finance** | DXY, Gold, Oil, VIX | 5-min intervals | None (free) |
| **OANDA Order Book** | Retail positioning, pending orders | Every 20 min | Same API key |

---

## 🚀 Quick Start

### 1. Prerequisites
```bash
# Python 3.11+, pip, git
python3 --version  # must be 3.11+
```

### 2. Setup
```bash
git clone https://github.com/bek01/forex-trading-bot.git
cd forex-trading-bot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"
# OR manually:
pip install oandapyV20 pandas numpy pandas-ta httpx aiohttp \
    python-dotenv pydantic pydantic-settings \
    python-telegram-bot apscheduler structlog aiosqlite \
    pytest pytest-asyncio ruff

# Configure
cp .env.example .env
# Edit .env with your OANDA credentials
```

### 3. Get OANDA API Credentials
1. Create a free practice account at [oanda.com/demo-account](https://www.oanda.com/demo-account/)
2. Go to **Manage API Access** → Generate personal access token
3. Note your Account ID (format: `101-001-12345678-001`)
4. Add both to `.env`

### 4. Test Connection
```bash
python main.py --status
# Should show: Balance: $100,000.00
```

### 5. Run Backtests First!
```bash
# Single strategy
python -m backtesting.run_backtest --strategy confluence --pair EUR_USD --timeframe H1 --monte-carlo

# All strategies
python -m backtesting.run_backtest --monte-carlo

# Parameter optimization (finds best settings)
python -m backtesting.optimizer --strategy confluence --iterations 500
python -m backtesting.optimizer --all --iterations 300
```

### 6. Paper Trading
```bash
# Start paper trading (demo account, no real money)
python main.py --paper

# Monitor in another terminal
python monitor.py
python monitor.py trades
python monitor.py strategies
python monitor.py equity
```

### 7. Deploy as Service
```bash
sudo cp trading-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start trading-bot
sudo systemctl enable trading-bot  # start on boot

# Check logs
journalctl -u trading-bot -f
```

---

## 📁 Project Structure

```
forex-trading-bot/
├── main.py                          # Bot orchestrator + event loop
├── models.py                        # Data models: Tick, Candle, Signal, Order, Position
├── event_bus.py                     # Event-driven communication system
├── monitor.py                       # CLI dashboard + health checks
│
├── config/
│   └── __init__.py                  # Pydantic settings: broker, risk, strategy, telegram
│
├── strategies/                      # Trading strategies (add new ones here)
│   ├── base.py                      # Abstract Strategy base class
│   ├── mean_reversion.py            # BB + RSI bounce
│   ├── trend_following.py           # EMA crossover + ADX
│   ├── london_breakout.py           # Session range breakout
│   ├── confluence.py                # Multi-indicator scoring (7 confirmations)
│   ├── session_momentum.py          # Asian range → London/NY breakout
│   └── stat_arb.py                  # Correlated pair spread trading
│
├── execution/
│   ├── broker.py                    # OANDA v20 REST API adapter
│   └── order_executor.py            # Signal → Order conversion + execution
│
├── risk/
│   └── risk_manager.py              # 10 risk rules, position sizing, kill switch
│
├── data/
│   ├── candle_manager.py            # Multi-instrument/timeframe candle buffers
│   ├── economic_calendar.py         # ForexFactory high-impact event filter
│   └── sentiment.py                 # COT + OANDA positioning + DXY correlation
│
├── monitoring/
│   └── telegram_bot.py              # Alerts, daily reports, /kill command
│
├── backtesting/
│   ├── engine.py                    # Backtest engine + Monte Carlo simulation
│   ├── run_backtest.py              # CLI backtest runner
│   └── optimizer.py                 # Parameter grid search + walk-forward analysis
│
├── db/
│   └── database.py                  # SQLite: orders, positions, equity, candles
│
├── tests/
│   ├── test_risk_manager.py         # 12 risk management tests
│   ├── test_strategies.py           # 9 strategy tests
│   └── test_database.py             # 5 database tests
│
├── .env.example                     # Config template
├── pyproject.toml                   # Dependencies + tool config
├── trading-bot.service              # Systemd service file
├── setup.sh                         # One-command setup script
└── README.md                        # This file
```

---

## 🔧 Adding a New Strategy

Create a new file in `strategies/` — it only takes ~20 lines:

```python
from strategies.base import Strategy, candles_to_df
from models import Signal, Side, SignalStrength
import pandas_ta as ta

class MyStrategy(Strategy):
    name = "my_strategy"
    timeframes = ["H1"]

    def on_candle(self, instrument, timeframe, candles):
        if timeframe != "H1" or len(candles) < 50:
            return None

        df = candles_to_df(candles)
        rsi = ta.rsi(df["close"], length=14)
        atr = ta.atr(df["high"], df["low"], df["close"], length=14)

        if rsi.iloc[-1] < 30:  # oversold
            return Signal(
                strategy=self.name,
                instrument=instrument,
                side=Side.BUY,
                entry_price=df["close"].iloc[-1],
                stop_loss=df["close"].iloc[-1] - atr.iloc[-1] * 2,
                take_profit=df["close"].iloc[-1] + atr.iloc[-1] * 3,
            )
        return None
```

Then register it in `main.py` → `_load_strategies()`.

---

## 📊 Backtesting & Optimization

### Minimum Bar for Live Trading

A strategy must pass ALL of these in backtesting AND paper trading:

| Metric | Threshold | Why |
|--------|-----------|-----|
| Sharpe Ratio | > 1.5 | Risk-adjusted return |
| Max Drawdown | < 15% | Capital preservation |
| Profit Factor | > 1.5 | Gross profit / gross loss |
| Win Rate | > 45% | Statistical significance |
| Total Trades | > 200 | Avoid overfitting to small samples |
| Walk-Forward | > 70% OOS profitable | Out-of-sample validation |
| Monte Carlo | > 90% profitable | Robustness across trade orderings |

### Paper Trading Validation

**Minimum 4 weeks** of paper trading before real money. Paper results must match backtest metrics within 1 standard deviation. If paper is >30% worse → strategy is overfit, discard.

---

## 📱 Telegram Commands

| Command | Action |
|---------|--------|
| `/status` | Current equity, positions, drawdown, daily P&L |
| `/trades` | Last 5 closed trades with P&L |
| `/kill` | Emergency: close all positions, halt trading |
| `/resume` | Resume trading after kill switch |
| `/help` | Show available commands |

Auto-alerts: trade opened/closed, daily report, drawdown warnings, errors, kill switch.

---

## ⚠️ Disclaimer

This software is for personal educational use. Trading forex involves significant risk of loss. Past backtesting performance does not guarantee future results. Never trade with money you cannot afford to lose. The author is not a financial advisor.

---

## 📄 License

MIT License — Personal use. See [LICENSE](LICENSE).
