# FX Trading Bot — Project Guide

## Quick Reference
- **Project**: `/home/ubuntu/trading_bot/`
- **Python**: `/home/ubuntu/venv/bin/python3`
- **Service (live)**: `trading-bot-live` (systemd unit in repo: `trading-bot-live.service`)
- **Service (paper)**: `trading-bot` (unit: `trading-bot.service`)
- **Logs**: `logs/trading_bot.log` (+ rotated `.1/.2/.3`)
- **DB (live)**: `db/trades_live.db` + `db/pair_guard_live.json`
- **DB (paper)**: `db/trades.db` + `db/pair_guard.json`
- **Config**: `config/__init__.py` (Pydantic, reads `.env` via env prefixes `OANDA_`, `RISK_`, `STRATEGY_`, `TELEGRAM_`)

## Architecture
- `main.py` — Main loop, event bus wiring, signal pipeline
- `models.py` — Candle, Signal, Side, SignalStrength, FOREX_PAIRS
- `event_bus.py` — pubsub events (CANDLE_CLOSE, SIGNAL, RISK_BLOCKED, …)
- `strategies/` — Six strategies: `mean_reversion`, `trend_following`, `london_breakout`, `confluence`, `session_momentum`, `stat_arb`
- `risk/risk_manager.py` — Position limits, drawdown halt, daily loss limit, correlation cap
- `risk/pair_guard.py` — Auto-block pairs after 3 consecutive losses (exponential backoff: 7d→14d→28d→56d→90d)
- `data/trend_filter.py` — **GLOBAL** counter-trend filter sitting between strategies and risk manager
- `data/candle_manager.py`, `data/sentiment.py`, `data/economic_calendar.py`

## Signal Pipeline (order of gates)
```
CANDLE_CLOSE → strategy.on_candle() → SIGNAL event
    → _on_signal():
        1. PairGuard.is_blocked(pair)         ← blocks for N days after 3 losses
        2. GlobalTrendFilter.filter_signal()  ← blocks counter-trend / FLAT consensus
        3. RiskManager.evaluate_signal()      ← position cap, drawdown, corr, spread
        4. Executor.execute_signal()          ← OANDA order
```

## New Architecture (IN-PROGRESS — not yet live)
Evidence-based rebuild started 2026-04-21 per `memory/reference_fx_architecture.md`.

**Scaffolding built, NOT yet wired into main loop** — needs backtest validation first.

Files:
- `strategies/ewmac.py` — EWMAC(16/64) + EWMAC(32/128) on D1, vol-normalised, Carver-style forecasts ±20
- `strategies/carry.py` — cross-sectional carry using OANDA `get_instrument_financing()` rates; long high-yield quintile, short low-yield
- `portfolio/vol_target.py` — forecast combination + 20%/yr vol-targeted sizing. IDM 1.2, FDM 1.2
- `execution/broker.py::get_instrument_financing()` — fetches OANDA swap rates

Time horizon: position trading. ~20-40 trades/month total across 8 pairs. Avg hold weeks-months.

Instruments (8): EUR_USD, GBP_USD, USD_JPY, AUD_USD, USD_CAD, NZD_USD, USD_CHF, EUR_JPY.

**Next steps before this architecture trades**:
1. Walk-forward backtest (vectorbt) with pass criteria: OOS Sharpe > 0.6, ≥100 trades OOS, stable across 3 sub-periods
2. Monte Carlo resampling of trade sequence
3. ATR-scaled ProfitManager stages for daily-bar hold times (+1/+2/+3 ATR, not fixed pips)
4. Wire into main.py event loop (currently scaffolding only)
5. Enable behind `ENABLE_NEW_ARCH` flag, demo-only for 6 months

## Gate Toggles (2026-04-21 → current)
`DISABLE_TREND_FILTER=true` and `DISABLE_PAIR_GUARD=true` set on BOTH demo and live `.env` files. Risk manager (positions, drawdown, daily loss limit) still fully active. Current-arch strategies now reach executor without counter-trend / per-pair block gates — running unfiltered on demo+live while new arch is built.

**2026-04-22 PM**: `RISK_MAX_SAME_CURRENCY_EXPOSURE=1` (was 2) on both accounts. Triggered by CHF-long concentration (EUR_CHF + GBP_CHF SELLs simultaneously) costing ~$1.3K on demo. Blocks the second correlated exposure to one side of a currency. Env parser note: .env comments MUST be on their own line — pydantic doesn't strip inline comments.

## Active Strategies (all enabled as of 2026-04-20)
| Strategy | TF | Entry condition | Key filter |
|---|---|---|---|
| `mean_reversion` | M15 | BB touch + RSI extreme | `ADX < 35` (ranging) |
| `trend_following` | H1 | EMA15/50 cross | `ADX > 30`, daily EMA200 trend |
| `london_breakout` | M5 | Break of 07:00-07:30 UTC range | Range 15-50 pips, max 1/pair/session |
| `confluence` | H1 | 7-indicator score | **score ≥ 6/7 AND daily trend must be UP/DOWN (not FLAT)** |
| `session_momentum` | M15 | Asian range breakout + RSI/ADX | Max 1 London + 1 NY /day |
| `stat_arb` | H1 | EUR/GBP spread z-score > 2 | EUR_USD + GBP_USD only |

## Global Trend Filter Rules (in `data/trend_filter.py`)
These are a SECOND layer on top of strategy-internal filters:
- **Rule 0**: `consensus == "FLAT"` (H4 or D not aligned) → BLOCK ALL signals
- **Rule 1**: `consensus == "UP"` + `side == SELL` → BLOCK (counter-trend)
- **Rule 1**: `consensus == "DOWN"` + `side == BUY` → BLOCK (counter-trend)
- **Rule 2**: `regime == "TRENDING"` (ADX>30) → block mean_reversion; `regime == "RANGING"` (ADX<20) → block trend_following
- **Rule 3**: DXY direction check for all USD pairs
- **Rule 4**: Retail positioning — HARD block at ≥75% on same side; WARN at 65-75%

**Consequence**: a FLAT-consensus pair (very common on quiet days) is completely untradable by any strategy. This is the #1 cause of low signal counts.

## Risk Limits (config/__init__.py defaults)
- `max_risk_per_trade_pct`: 0.5%
- `max_open_positions`: 6
- `max_same_currency_exposure`: 2
- `daily_loss_limit_pct`: 3.0%
- `daily_profit_lock_pct`: 2.0% (halts NEW entries, runners continue)
- `daily_equity_trail_pct`: 1.5% (halt after -1.5% from today's peak)
- `max_drawdown_halt_pct`: 10.0%

## PairGuard Backoff
- 3 consecutive losses → block 7 days
- Loss again after unblock → ×2 block days (14→28→56→90)
- `is_blocked()` compares `now` to `blocked_until` ISO timestamp
- `clear_block(pair)` resets to unblocked but KEEPS loss counter
- State files: `db/pair_guard.json` (paper) and `db/pair_guard_live.json` (live)

## Hard Rules / Critical Knowledge
- **NEVER set `require_stop_loss = False`** — every trade must have a hard SL
- **NEVER disable the trend filter entirely** — it prevents the Apr 13 counter-trend blowup
- **Live/Paper DB split** — `.env` switches `trading_mode`; paper uses `trades.db`, live uses `trades_live.db`
- **OANDA env split** — `OANDA_environment=practice|live` selects REST/Stream URLs
- **PairGuard WIN fix (2026-04-20)**: a win must clear `blocked_until`. Earlier bug persisted stale blocks for winning pairs.

## Current State (2026-04-20 18:43 UTC)
- Balance (live): $94.62
- Bot has restarted 5+ times today (14:06, 14:57, 15:03, 15:09 UTC)
- 3 unique signals all day, **0 executed** (all blocked by trend filter / pair guard)
- Live PairGuard still blocks: `CAD_CHF` (→4/23), `GBP_CAD` (→4/23), `AUD_CAD` (→5/14)
- Pairs 1 loss away from block: `EUR_CAD` (2), `EUR_CHF` (1), `GBP_USD` (1), `EUR_AUD` (1)

## Broker Resilience (2026-04-21)
- `execution/broker.py::_request()` retries transient errors: HTTP 429/500/502/503/504, ConnectError, timeouts. 3 attempts, 0.5s+1.5s backoff. 4xx fails immediately (no retry).
- `data/candle_manager.py` has a per-endpoint circuit breaker: 3 consecutive poll failures → 60s cooldown. Logs `circuit-breaker TRIPPED` on trip, `recovered` on first success after.
- OANDA demo (`api-fxpractice`) has historical pattern of transient 502s. Live (`api-fxtrade`) is more stable. Neither our problem — client resilience is the answer.

## Known Open Problems
1. **Flat-consensus lockout** — Trend Filter Rule 0 blocks ALL signals when H4 or D is FLAT. On a chop day, no strategy can fire. Needs per-strategy relaxation (mean_reversion should be allowed in FLAT+RANGING).
2. **No ranging-regime specialist strategy** — Gap between "ADX 20-30, FLAT trend" where nothing fires.
3. **Confluence too strict post-2026-04-16** — required daily UP/DOWN (no FLAT), score threshold 6/7. Very rarely fires.
4. **Trend Following ADX=30 threshold** — most majors sat at ADX 15-25 on 2026-04-20.

## User Preferences
- Wants automated monitoring + Telegram notifications
- Goal: consistent daily profit, conservative scaling
- Prefers honest assessment over optimistic projections
- Wants session work recorded to memory for continuity
- Audit reports generated via Telegram by bot or script

## Memory System
- Auto-memory index at `~/.claude/projects/-home-ubuntu/memory/MEMORY.md`
- Session files: `session_YYYY_MM_DD.md`
- Also stores per-project files when they span many sessions
