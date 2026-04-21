"""Carry — currency carry trade rule using OANDA financing rates.

Reference: Daniel/Hodrick/Lu "The Carry Trade: Risks and Drawdowns" (2017),
Burnside/Eichenbaum/Rebelo "Carry Trade and Momentum in Currency Markets".
Peer-reviewed Sharpe 0.4-1.0 with important tail-risk caveats.

What this rule does:
- Fetch OANDA's published financing (swap) rates for each instrument
- Compute the carry yield of going long each pair (base_rate - quote_rate)
  or short (reverse). OANDA returns these as `longRate` / `shortRate`.
- Cross-sectional rank the universe by carry yield
- Top quintile → long forecast, bottom quintile → short forecast, scaled

Timeframe: monthly rebalance. This rule emits forecasts on daily candle
closes but they change slowly — financing rates update infrequently and
cross-sectional ranks are stable over weeks.

Output: a `Forecast` (from strategies.ewmac), capped ±20.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING

from models import Candle
from strategies.base import Strategy
from strategies.ewmac import Forecast

if TYPE_CHECKING:
    from execution.broker import OandaBroker


class CarryStrategy(Strategy):
    """Cross-sectional carry rule — long high-yield, short low-yield."""

    name = "carry"
    timeframes = ["D"]
    instruments = [
        "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD",
        "USD_CAD", "NZD_USD", "USD_CHF", "EUR_JPY",
    ]

    forecast_cap: float = 20.0
    # Scalar: forecast = z_score * scalar, capped at cap.
    # 10.0 makes an average 1-sigma divergence produce a 10-forecast
    # (matching Carver's "average absolute forecast ≈ 10" convention).
    forecast_scalar: float = 10.0

    # Refresh financing rates every N hours — they change slowly
    refresh_hours: float = 24.0

    def __init__(self, broker: Optional["OandaBroker"] = None):
        super().__init__()
        self.broker = broker
        # Cache: {instrument: (long_rate, short_rate)}
        self._rates: dict[str, tuple[float, float]] = {}
        self._last_refresh: Optional[datetime] = None

    def set_broker(self, broker: "OandaBroker"):
        """Wire the broker post-init (strategies init before broker in main.py)."""
        self.broker = broker

    def _refresh_rates(self):
        """Pull the latest financing rates for all instruments in our universe."""
        if self.broker is None:
            return
        new_rates: dict[str, tuple[float, float]] = {}
        for inst in self.instruments:
            fin = self.broker.get_instrument_financing(inst)
            if fin is None:
                # Keep prior rate if we have one
                if inst in self._rates:
                    new_rates[inst] = self._rates[inst]
                continue
            new_rates[inst] = (fin["long_rate"], fin["short_rate"])
        self._rates = new_rates
        self._last_refresh = datetime.now(timezone.utc)
        self.logger.info(
            f"Carry rates refreshed: {len(new_rates)}/{len(self.instruments)} "
            f"instruments — {sorted(new_rates.keys())}"
        )

    def _rates_stale(self) -> bool:
        if self._last_refresh is None:
            return True
        age = datetime.now(timezone.utc) - self._last_refresh
        return age.total_seconds() > self.refresh_hours * 3600

    def on_candle(
        self,
        instrument: str,
        timeframe: str,
        candles: list[Candle],
    ) -> Optional[Forecast]:
        if timeframe != "D":
            return None

        if self._rates_stale():
            self._refresh_rates()

        # Need at least 3 rates for a meaningful cross-section
        if len(self._rates) < 3:
            return None

        if instrument not in self._rates:
            return None

        # Cross-section: use the larger magnitude of (long, short) as
        # "carry yield going the winning direction". Some pairs are only
        # attractive one way.
        yields = {}
        for inst, (long_rate, short_rate) in self._rates.items():
            # Prefer the side with higher expected carry
            if long_rate >= -short_rate:
                yields[inst] = (long_rate, "long")
            else:
                yields[inst] = (-short_rate, "short")

        # Compute z-score of THIS instrument's yield across the universe
        values = [y[0] for y in yields.values()]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = variance ** 0.5
        if std <= 0:
            return None

        my_yield, my_side = yields[instrument]
        z = (my_yield - mean) / std

        # Signed forecast: positive z → carry this way; direction from my_side
        signed_z = z if my_side == "long" else -z
        scaled = signed_z * self.forecast_scalar
        capped = max(-self.forecast_cap, min(scaled, self.forecast_cap))

        if abs(capped) < 1.0:
            # Too weak to act on — return neutral forecast (skip to reduce churn)
            return None

        forecast = Forecast(
            strategy=self.name,
            instrument=instrument,
            value=capped,
            metadata={
                "long_rate": self._rates[instrument][0],
                "short_rate": self._rates[instrument][1],
                "yield_used": my_yield,
                "preferred_side": my_side,
                "z_score": z,
                "universe_size": len(self._rates),
            },
        )
        self.trade_count += 1
        self.logger.info(
            f"Forecast: {instrument} carry={capped:+.1f} "
            f"(yield={my_yield:.4f}, side={my_side}, z={z:+.2f})"
        )
        return forecast
