"""Portfolio-level forecast combination + volatility targeting.

Reference: Rob Carver, *Systematic Trading* (ch. 8-10) and pysystemtrade
(github.com/robcarver17/pysystemtrade). This is the canonical approach for
multi-rule, multi-instrument systematic trend/carry systems.

Flow on each daily candle close:
  forecasts (per rule, per instrument)
    → combined forecast per instrument (weighted average × diversification multiplier)
    → notional exposure per instrument (forecast × target volatility)
    → units per instrument (notional / price × pip size handling)
    → diff against current position → target order(s)

Key parameters (Carver defaults, can be tuned):
- target annual volatility: 20% (we use 20%, Carver uses 25%)
- forecast scalar chosen so E[|forecast|] ≈ 10 (handled in the rule itself)
- forecast cap: ±20
- idm (instrument diversification multiplier): 1.2 for 2 rules
- instrument weights: equal by default across the 8 pairs

This module is DATA STRUCTURES + MATH. It does not place orders; it emits
`TargetPosition` objects that the live loop converts into Signal events or
order diffs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from strategies.ewmac import Forecast


@dataclass
class TargetPosition:
    """Desired position in units for an instrument.

    Positive = long, negative = short, 0 = flat. The executor compares against
    current broker position and issues the diff as market orders with ATR-
    based stops.
    """
    instrument: str
    target_units: float
    combined_forecast: float
    rule_forecasts: dict[str, float]  # {rule_name: forecast_value}
    target_notional_ccy: float  # in account currency
    timestamp: datetime


class VolTargetPortfolio:
    """Aggregates per-rule forecasts into one target position per instrument."""

    # --- Tunable knobs ---

    # Annual volatility target as fraction of equity (0.20 = 20%/yr). Carver
    # uses 0.25; 0.20 leaves margin for model error. At 20% annual vol and
    # ~252 trading days, daily portfolio vol ≈ 1.26%.
    annual_vol_target: float = 0.20

    # Forecast at which we size to our full target vol. ±20 = capped max,
    # ±10 = average strength per Carver's scaling.
    forecast_avg_magnitude: float = 10.0

    # Instrument Diversification Multiplier. Boosts exposure because
    # diversification means realised vol is less than sum of parts. Carver's
    # tabulated value for ~8 liquid FX pairs is ≈1.3; we use 1.2 to be
    # conservative.
    idm: float = 1.2

    # Forecast Diversification Multiplier. Boosts the combined forecast
    # because averaging uncorrelated rules reduces its variance. For 2 rules
    # with ~0.1 correlation, Carver's table gives ~1.35; use 1.2 conservative.
    fdm: float = 1.2

    # Trading days per year for volatility annualisation math.
    trading_days: int = 252

    def __init__(
        self,
        instruments: list[str],
        rule_weights: Optional[dict[str, float]] = None,
        instrument_weights: Optional[dict[str, float]] = None,
    ):
        self.instruments = list(instruments)
        # Default equal rule weights if not specified
        self.rule_weights = rule_weights or {"ewmac": 0.5, "carry": 0.5}
        # Default equal instrument weights
        n = len(self.instruments)
        self.instrument_weights = instrument_weights or {
            inst: 1.0 / n for inst in self.instruments
        }
        # Rolling forecast storage: {instrument: {rule: value}}
        self._forecasts: dict[str, dict[str, float]] = {
            inst: {} for inst in self.instruments
        }

    def add_forecast(self, forecast: Forecast):
        """Record a rule's forecast for an instrument."""
        if forecast.instrument not in self._forecasts:
            self._forecasts[forecast.instrument] = {}
        self._forecasts[forecast.instrument][forecast.strategy] = forecast.value

    def _combined_forecast(self, instrument: str) -> float:
        """Weighted sum of rule forecasts × FDM, capped."""
        rules = self._forecasts.get(instrument, {})
        if not rules:
            return 0.0
        total = 0.0
        total_weight = 0.0
        for rule, value in rules.items():
            w = self.rule_weights.get(rule, 0.0)
            total += value * w
            total_weight += w
        if total_weight <= 0:
            return 0.0
        # Re-normalise (in case some rules missing) then apply FDM
        combined = (total / total_weight) * self.fdm
        # Cap at ±20
        return max(-20.0, min(combined, 20.0))

    def target_position(
        self,
        instrument: str,
        equity: float,
        price: float,
        daily_price_vol: float,
        pip_size: float,
    ) -> Optional[TargetPosition]:
        """Compute target position for an instrument.

        Args:
            instrument: e.g. "EUR_USD"
            equity: current account equity in account currency
            price: current mid price
            daily_price_vol: avg absolute daily price change (ATR-like, in quote ccy)
            pip_size: pip size for the instrument (0.0001 or 0.01)
        """
        combined = self._combined_forecast(instrument)
        rule_forecasts = dict(self._forecasts.get(instrument, {}))

        # Target notional per instrument:
        #   target_vol_ccy = equity * annual_vol_target / sqrt(trading_days)
        #   instrument_notional = target_vol_ccy * idm * inst_weight * (forecast / avg_magnitude)
        #   units = instrument_notional / daily_price_vol
        daily_vol_target = equity * self.annual_vol_target / (self.trading_days ** 0.5)
        inst_weight = self.instrument_weights.get(instrument, 0.0)
        if inst_weight <= 0 or equity <= 0 or daily_price_vol <= 0:
            return None

        # Forecast-scaled notional in account currency
        forecast_ratio = combined / self.forecast_avg_magnitude
        instrument_notional = daily_vol_target * self.idm * inst_weight * forecast_ratio

        # Convert notional to units. For an FX pair QUOTED in account ccy
        # (e.g. account=GBP, pair=GBP_USD), 1 unit moves daily_price_vol *
        # quote_ccy_to_account_ccy ≈ daily_price_vol per unit in account ccy
        # (simplification — accurate when account ccy matches quote ccy).
        # For strict correctness we'd multiply by the FX rate of quote→account
        # at this moment. TODO in backtest harness.
        if daily_price_vol <= 0:
            return None
        target_units = instrument_notional / daily_price_vol

        return TargetPosition(
            instrument=instrument,
            target_units=target_units,
            combined_forecast=combined,
            rule_forecasts=rule_forecasts,
            target_notional_ccy=instrument_notional,
            timestamp=datetime.now(timezone.utc),
        )

    def clear_forecasts(self, instrument: Optional[str] = None):
        """Reset stored forecasts. Call between periods if desired."""
        if instrument:
            self._forecasts[instrument] = {}
        else:
            self._forecasts = {inst: {} for inst in self.instruments}

    def get_state(self) -> dict:
        """Debug/monitoring snapshot."""
        return {
            "annual_vol_target": self.annual_vol_target,
            "idm": self.idm,
            "fdm": self.fdm,
            "rule_weights": self.rule_weights,
            "instruments": list(self._forecasts.keys()),
            "latest_forecasts": {
                inst: dict(rules)
                for inst, rules in self._forecasts.items()
                if rules
            },
        }
