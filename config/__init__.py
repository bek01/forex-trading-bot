"""Trading bot configuration.

All settings are centralized here with sensible defaults.
Override via .env file or environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env path relative to project root
_ENV_FILE = str(Path(__file__).parent.parent / ".env")


class BrokerConfig(BaseSettings):
    """OANDA broker connection settings."""
    model_config = SettingsConfigDict(
        env_prefix="OANDA_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    account_id: str = ""
    api_token: str = ""
    environment: str = "practice"  # "practice" or "live"

    # API endpoints (set in model_post_init based on environment)
    rest_url: str = ""
    stream_url: str = ""

    def model_post_init(self, __context):
        if not self.rest_url:
            if self.environment == "practice":
                self.rest_url = "https://api-fxpractice.oanda.com"
                self.stream_url = "https://stream-fxpractice.oanda.com"
            else:
                self.rest_url = "https://api-fxtrade.oanda.com"
                self.stream_url = "https://stream-fxtrade.oanda.com"


class RiskConfig(BaseSettings):
    """Risk management parameters. Conservative defaults."""
    model_config = SettingsConfigDict(
        env_prefix="RISK_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Per-trade risk — scaled for £100K demo
    max_risk_per_trade_pct: float = 0.5         # Risk 0.5% per trade (£500 on 100K)
    max_risk_per_trade_usd: float = 500.0       # Hard cap in GBP per trade

    # Position limits — expanded for 24/5 multi-pair trading
    max_open_positions: int = 10                # was 3 — now trading 23 pairs
    max_correlated_exposure: float = 1.5        # Correlated pairs count as 1.5x

    # Drawdown limits
    daily_loss_limit_pct: float = 3.0           # Stop trading if daily loss > 3% (£3K on 100K)
    max_drawdown_warning_pct: float = 5.0       # Alert at 5% drawdown from peak
    max_drawdown_critical_pct: float = 8.0      # Reduce size at 8%
    max_drawdown_halt_pct: float = 10.0         # Kill switch at 10% from peak

    # Spread protection
    max_spread_multiplier: float = 2.0          # Skip if spread > 2x average

    # Stop loss rules
    require_stop_loss: bool = True              # NEVER change this to False
    min_risk_reward_ratio: float = 1.5          # Minimum R:R to take a trade
    max_stop_loss_pips: float = 50.0            # No trade with SL > 50 pips

    # Time-based rules
    friday_reduce_size_pct: float = 50.0        # Reduce size after Friday 18:00 UTC
    no_trade_before_news_minutes: int = 30      # No new trades 30min before news
    no_trade_after_news_minutes: int = 30       # No new trades 30min after news


class StrategyConfig(BaseSettings):
    """Default strategy parameters."""
    model_config = SettingsConfigDict(
        env_prefix="STRATEGY_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Instruments to trade — 24/5 across Tokyo, London, New York
    instruments: list[str] = [
        # Majors (7) — always active, tightest spreads
        "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF",
        "AUD_USD", "USD_CAD", "NZD_USD",
        # JPY crosses — Tokyo session stars
        "EUR_JPY", "GBP_JPY", "AUD_JPY", "CAD_JPY",
        # EUR crosses — London session
        "EUR_GBP", "EUR_AUD", "EUR_CAD", "EUR_CHF",
        # GBP crosses — London session
        "GBP_AUD", "GBP_CHF", "GBP_CAD",
        # AUD/NZD crosses — Tokyo/Sydney
        "AUD_NZD", "AUD_CAD",
        # Other
        "CHF_JPY", "NZD_JPY", "CAD_CHF",
    ]

    # Timeframes to watch
    timeframes: list[str] = ["M5", "M15", "H1", "H4", "D"]

    # Strategy-specific defaults
    mean_reversion_enabled: bool = True
    trend_following_enabled: bool = True
    london_breakout_enabled: bool = True
    confluence_enabled: bool = True
    session_momentum_enabled: bool = True
    stat_arb_enabled: bool = True

    # Candle buffer size (how many candles to keep in memory per instrument/timeframe)
    candle_buffer_size: int = 300  # reduced from 500 — more pairs, less depth needed


class TelegramConfig(BaseSettings):
    """Telegram bot for monitoring and kill switch."""
    model_config = SettingsConfigDict(
        env_prefix="TELEGRAM_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    bot_token: str = ""
    chat_id: str = ""
    enabled: bool = False


class AppConfig(BaseSettings):
    """Top-level application config."""
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    trading_mode: str = "paper"  # "paper" or "live"
    log_level: str = "INFO"
    db_path: str = "db/trades.db"
    account_currency: str = "GBP"  # Account denomination

    # Main loop timing
    heartbeat_interval_sec: int = 60
    account_sync_interval_sec: int = 30
    equity_snapshot_interval_sec: int = 300  # 5 minutes

    # Candle polling (fallback if streaming gaps)
    candle_poll_interval_sec: int = 5

    broker: BrokerConfig = Field(default_factory=BrokerConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)


# Singleton config instance
_config: AppConfig | None = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def reload_config() -> AppConfig:
    global _config
    _config = AppConfig()
    return _config
