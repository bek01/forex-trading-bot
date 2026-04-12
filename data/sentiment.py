"""Market sentiment and positioning data from multiple free sources.

Sources:
1. OANDA Order/Position Book — retail positioning (requires API key)
2. CFTC COT Reports — institutional/speculative positioning (free, no key)
3. DXY (US Dollar Index) via Yahoo Finance — USD trend (free, no key)

All methods handle errors gracefully and return neutral values on failure.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# --- Cache TTLs (seconds) ---
OANDA_BOOK_TTL = 1200      # 20 minutes
COT_TTL = 86400             # 24 hours
DXY_TTL = 300               # 5 minutes

# --- CFTC SODA endpoint ---
CFTC_COT_URL = "https://publicreporting.cftc.gov/resource/jun7-fc8e.json"

# --- Yahoo Finance DXY chart ---
YAHOO_DXY_URL = "https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB"

# Map currency names to CFTC commodity_name search terms
_CURRENCY_TO_CFTC = {
    "EUR": "EURO FX",
    "GBP": "BRITISH POUND",
    "JPY": "JAPANESE YEN",
    "AUD": "AUSTRALIAN DOLLAR",
    "CAD": "CANADIAN DOLLAR",
    "CHF": "SWISS FRANC",
    "USD": "U.S. DOLLAR INDEX",
}

# Map OANDA instrument names for order book API
# OANDA uses the same format we do (EUR_USD) so usually no mapping needed.


class _CacheEntry:
    """Simple TTL cache entry."""

    __slots__ = ("data", "timestamp", "ttl")

    def __init__(self, data, ttl: float):
        self.data = data
        self.timestamp = time.monotonic()
        self.ttl = ttl

    @property
    def expired(self) -> bool:
        return (time.monotonic() - self.timestamp) > self.ttl


class SentimentData:
    """Aggregated sentiment/positioning data from multiple sources."""

    def __init__(self, oanda_api_token: str = "", oanda_base_url: str = ""):
        """
        Args:
            oanda_api_token: OANDA API bearer token for order/position book.
            oanda_base_url: OANDA REST API base URL (e.g. https://api-fxpractice.oanda.com).
        """
        self._oanda_token = oanda_api_token
        self._oanda_url = oanda_base_url.rstrip("/") if oanda_base_url else ""
        self._cache: dict[str, _CacheEntry] = {}
        self._timeout = 10.0

    # ------------------------------------------------------------------
    # OANDA Order/Position Book
    # ------------------------------------------------------------------

    def _oanda_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._oanda_token}",
            "Content-Type": "application/json",
        }

    def get_oanda_positioning(self, instrument: str) -> Optional[dict]:
        """Fetch OANDA order book to get retail long/short percentages.

        Returns:
            {
                "pct_long": float,   # 0-100
                "pct_short": float,  # 0-100
                "bias": str,         # "LONG", "SHORT", or "NEUTRAL"
                "source": "oanda_order_book",
                "instrument": str,
            }
            or None on failure.
        """
        if not self._oanda_token or not self._oanda_url:
            return None

        cache_key = f"oanda_book_{instrument}"
        cached = self._cache.get(cache_key)
        if cached and not cached.expired:
            return cached.data

        url = f"{self._oanda_url}/v3/instruments/{instrument}/orderBook"
        try:
            resp = requests.get(url, headers=self._oanda_headers(), timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            logger.warning(f"OANDA order book failed for {instrument}: {exc}")
            return None

        try:
            buckets = data.get("orderBook", {}).get("buckets", [])
            if not buckets:
                return None

            total_long = sum(float(b.get("longCountPercent", 0)) for b in buckets)
            total_short = sum(float(b.get("shortCountPercent", 0)) for b in buckets)
            total = total_long + total_short

            if total <= 0:
                return None

            pct_long = (total_long / total) * 100.0
            pct_short = (total_short / total) * 100.0

            # Determine bias — needs meaningful imbalance
            if pct_long > 55:
                bias = "LONG"
            elif pct_short > 55:
                bias = "SHORT"
            else:
                bias = "NEUTRAL"

            result = {
                "pct_long": round(pct_long, 1),
                "pct_short": round(pct_short, 1),
                "bias": bias,
                "source": "oanda_order_book",
                "instrument": instrument,
            }
            self._cache[cache_key] = _CacheEntry(result, OANDA_BOOK_TTL)
            return result

        except (KeyError, TypeError, ZeroDivisionError) as exc:
            logger.warning(f"Error parsing OANDA order book for {instrument}: {exc}")
            return None

    # ------------------------------------------------------------------
    # CFTC COT Reports
    # ------------------------------------------------------------------

    def get_cot_positioning(self, currency: str) -> Optional[dict]:
        """Fetch latest CFTC Commitment of Traders data for a currency.

        Args:
            currency: Currency code like "EUR", "GBP", "JPY".

        Returns:
            {
                "net_speculative": int,      # net non-commercial contracts (positive = net long)
                "direction": str,            # "LONG", "SHORT", or "NEUTRAL"
                "noncomm_long": int,
                "noncomm_short": int,
                "source": "cftc_cot",
                "currency": str,
                "report_date": str,
            }
            or None on failure.
        """
        currency = currency.upper()
        cftc_name = _CURRENCY_TO_CFTC.get(currency)
        if not cftc_name:
            return None

        cache_key = f"cot_{currency}"
        cached = self._cache.get(cache_key)
        if cached and not cached.expired:
            return cached.data

        params = {
            "$where": f"commodity_name like '%{cftc_name}%'",
            "$order": "report_date_as_yyyy_mm_dd DESC",
            "$limit": 2,  # latest + previous for weekly change
        }

        try:
            resp = requests.get(CFTC_COT_URL, params=params, timeout=self._timeout)
            resp.raise_for_status()
            rows = resp.json()
        except (requests.RequestException, ValueError) as exc:
            logger.warning(f"CFTC COT fetch failed for {currency}: {exc}")
            return None

        if not rows:
            return None

        try:
            latest = rows[0]
            noncomm_long = int(float(latest.get("noncomm_positions_long_all", 0)))
            noncomm_short = int(float(latest.get("noncomm_positions_short_all", 0)))
            net = noncomm_long - noncomm_short

            # Weekly change (if we have previous week)
            change_weekly = 0
            if len(rows) >= 2:
                prev = rows[1]
                prev_long = int(float(prev.get("noncomm_positions_long_all", 0)))
                prev_short = int(float(prev.get("noncomm_positions_short_all", 0)))
                prev_net = prev_long - prev_short
                change_weekly = net - prev_net

            if net > 0:
                direction = "LONG"
            elif net < 0:
                direction = "SHORT"
            else:
                direction = "NEUTRAL"

            result = {
                "net_speculative": net,
                "direction": direction,
                "change_weekly": change_weekly,
                "noncomm_long": noncomm_long,
                "noncomm_short": noncomm_short,
                "source": "cftc_cot",
                "currency": currency,
                "report_date": latest.get("report_date_as_yyyy_mm_dd", ""),
            }
            self._cache[cache_key] = _CacheEntry(result, COT_TTL)
            return result

        except (KeyError, TypeError, ValueError) as exc:
            logger.warning(f"Error parsing COT data for {currency}: {exc}")
            return None

    # ------------------------------------------------------------------
    # DXY (US Dollar Index) trend
    # ------------------------------------------------------------------

    def get_dxy_trend(self) -> str:
        """Get current DXY trend from Yahoo Finance 15m intraday data.

        Returns:
            "UP", "DOWN", or "FLAT".
        """
        cache_key = "dxy_trend"
        cached = self._cache.get(cache_key)
        if cached and not cached.expired:
            return cached.data

        params = {"range": "1d", "interval": "15m"}
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
        }

        try:
            resp = requests.get(
                YAHOO_DXY_URL, params=params, headers=headers, timeout=self._timeout
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            logger.warning(f"DXY fetch failed: {exc}")
            return "FLAT"

        try:
            result_data = data["chart"]["result"][0]
            closes = result_data["indicators"]["quote"][0]["close"]

            # Filter out None values
            closes = [c for c in closes if c is not None]
            if len(closes) < 4:
                return "FLAT"

            # Compare recent 4 candles vs earlier 4 candles for short-term trend
            recent_avg = sum(closes[-4:]) / 4
            earlier_avg = sum(closes[-8:-4]) / 4 if len(closes) >= 8 else sum(closes[:4]) / 4

            pct_change = ((recent_avg - earlier_avg) / earlier_avg) * 100

            if pct_change > 0.05:
                trend = "UP"
            elif pct_change < -0.05:
                trend = "DOWN"
            else:
                trend = "FLAT"

            self._cache[cache_key] = _CacheEntry(trend, DXY_TTL)
            return trend

        except (KeyError, TypeError, IndexError, ZeroDivisionError) as exc:
            logger.warning(f"Error parsing DXY data: {exc}")
            return "FLAT"

    # ------------------------------------------------------------------
    # Combined sentiment score
    # ------------------------------------------------------------------

    def get_sentiment_score(self, instrument: str) -> float:
        """Combined sentiment score from all sources.

        Returns:
            Float from -1.0 (strongly bearish) to +1.0 (strongly bullish).
            0.0 = neutral or data unavailable.

        The score is a weighted average of:
            - OANDA retail positioning (contrarian): weight 0.3
            - CFTC COT speculative positioning: weight 0.4
            - DXY trend (for USD pairs only): weight 0.3
        """
        scores: list[tuple[float, float]] = []  # (score, weight)

        # Extract base and quote currencies
        parts = instrument.replace("-", "_").split("_")
        if len(parts) != 2:
            return 0.0
        base_ccy, quote_ccy = parts[0].upper(), parts[1].upper()

        # --- OANDA positioning (contrarian signal) ---
        oanda = self.get_oanda_positioning(instrument)
        if oanda is not None:
            # Contrarian: if retail is heavily long, that's bearish for the pair
            imbalance = (oanda["pct_long"] - 50.0) / 50.0  # -1 to +1
            contrarian_score = -imbalance  # flip it
            scores.append((contrarian_score, 0.3))

        # --- CFTC COT (base currency) ---
        cot_base = self.get_cot_positioning(base_ccy)
        if cot_base is not None and cot_base["net_speculative"] != 0:
            # Normalize: use a rough scale of 200k contracts as "max"
            net = cot_base["net_speculative"]
            cot_score = max(-1.0, min(1.0, net / 200000.0))
            scores.append((cot_score, 0.4))

        # --- CFTC COT (quote currency — inverted) ---
        cot_quote = self.get_cot_positioning(quote_ccy)
        if cot_quote is not None and cot_quote["net_speculative"] != 0:
            net = cot_quote["net_speculative"]
            cot_score_quote = max(-1.0, min(1.0, net / 200000.0))
            # Quote currency strength is bearish for the pair
            scores.append((-cot_score_quote, 0.2))

        # --- DXY trend (only relevant for USD pairs) ---
        if "USD" in (base_ccy, quote_ccy):
            dxy = self.get_dxy_trend()
            if dxy != "FLAT":
                dxy_score = 0.5 if dxy == "UP" else -0.5
                # If USD is quote currency, DXY up = pair down
                # If USD is base currency, DXY up = pair up
                if quote_ccy == "USD":
                    dxy_score = -dxy_score
                scores.append((dxy_score, 0.3))

        if not scores:
            return 0.0

        # Weighted average
        total_weight = sum(w for _, w in scores)
        if total_weight <= 0:
            return 0.0

        weighted_sum = sum(s * w for s, w in scores)
        combined = weighted_sum / total_weight

        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, round(combined, 3)))
