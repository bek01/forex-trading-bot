"""Economic calendar — fetches high-impact events from ForexFactory (free API).

Used by RiskManager to avoid trading around major news releases.
No API key required.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ForexFactory free calendar endpoint (current week)
FF_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# Cache TTL in seconds (1 hour)
CACHE_TTL_SEC = 3600

# Map ForexFactory currency codes to OANDA instrument components
# e.g. "USD" appears in EUR_USD, USD_JPY, etc.
_CURRENCY_TO_INSTRUMENTS = {
    "USD": ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "USD_CHF"],
    "EUR": ["EUR_USD", "EUR_GBP"],
    "GBP": ["GBP_USD", "EUR_GBP"],
    "JPY": ["USD_JPY"],
    "AUD": ["AUD_USD"],
    "CAD": ["USD_CAD"],
    "CHF": ["USD_CHF"],
}

# Known date formats used by ForexFactory feed
_DATE_FORMATS = [
    "%m-%d-%Y %I:%M%p",   # 04-12-2026 08:30am
    "%m-%d-%Y %H:%M",     # 04-12-2026 08:30
    "%Y-%m-%dT%H:%M:%S",  # 2026-04-12T08:30:00
    "%Y-%m-%d %H:%M:%S",  # 2026-04-12 08:30:00
    "%m-%d-%Y",            # 04-12-2026  (date only, no time)
]


def _parse_event_datetime(date_str: str) -> Optional[datetime]:
    """Parse the date string from ForexFactory into a timezone-aware UTC datetime.

    ForexFactory times are in Eastern Time (ET). We convert to UTC.
    """
    if not date_str:
        return None

    date_str = date_str.strip()

    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(date_str, fmt)
            # ForexFactory uses Eastern Time — offset is -4 (EDT) or -5 (EST).
            # Use -4 as a reasonable approximation during most of the trading year.
            eastern_offset = timedelta(hours=-4)
            dt = dt.replace(tzinfo=timezone(eastern_offset))
            return dt.astimezone(timezone.utc)
        except (ValueError, TypeError):
            continue

    return None


def _extract_currency_from_instrument(instrument: str) -> list[str]:
    """Extract currency codes from an OANDA instrument like 'EUR_USD'."""
    parts = instrument.replace("-", "_").split("_")
    return [p.upper() for p in parts if len(p) == 3]


class EconomicCalendar:
    """Fetches and caches the weekly economic calendar from ForexFactory."""

    def __init__(self):
        self._events: list[dict] = []
        self._last_fetch: float = 0.0
        self._fetch_timeout_sec: float = 10.0

    def refresh(self) -> list[dict]:
        """Fetch this week's calendar from ForexFactory.

        Caches for 1 hour. Returns the cached list on network errors.
        """
        now = time.monotonic()
        if self._events and (now - self._last_fetch) < CACHE_TTL_SEC:
            return self._events

        try:
            resp = requests.get(FF_CALENDAR_URL, timeout=self._fetch_timeout_sec)
            resp.raise_for_status()
            raw_events = resp.json()
        except (requests.RequestException, ValueError) as exc:
            logger.warning(f"Failed to fetch economic calendar: {exc}")
            return self._events  # return stale cache on failure

        parsed: list[dict] = []
        for ev in raw_events:
            dt = _parse_event_datetime(ev.get("date", ""))
            if dt is None:
                continue

            parsed.append({
                "title": ev.get("title", ""),
                "currency": ev.get("country", "").upper(),
                "datetime": dt,
                "impact": ev.get("impact", "Low"),
                "forecast": ev.get("forecast", ""),
                "previous": ev.get("previous", ""),
            })

        # Sort by datetime
        parsed.sort(key=lambda e: e["datetime"])
        self._events = parsed
        self._last_fetch = now
        logger.info(f"Economic calendar refreshed: {len(parsed)} events this week")
        return self._events

    def get_upcoming_high_impact(self, minutes_ahead: int = 60) -> list[dict]:
        """Return high-impact events occurring within the next N minutes."""
        self.refresh()
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(minutes=minutes_ahead)

        upcoming = []
        for ev in self._events:
            if ev["impact"] != "High":
                continue
            if now <= ev["datetime"] <= cutoff:
                upcoming.append(ev)

        if upcoming:
            titles = ", ".join(e["title"] for e in upcoming)
            logger.info(
                f"High-impact events in next {minutes_ahead}min: {titles}"
            )

        return upcoming

    def is_near_high_impact(
        self,
        currency: str,
        minutes_before: int = 30,
        minutes_after: int = 30,
    ) -> bool:
        """Check if a currency has a high-impact event within the given window.

        Args:
            currency: Currency code like "USD", "EUR", or an OANDA instrument
                      like "EUR_USD" (both currencies will be checked).
            minutes_before: Minutes before the event to flag.
            minutes_after: Minutes after the event to flag.

        Returns:
            True if there is a high-impact event nearby for this currency.
        """
        self.refresh()

        # If an instrument was passed, extract both currencies
        currencies_to_check: set[str] = set()
        if "_" in currency:
            currencies_to_check.update(_extract_currency_from_instrument(currency))
        else:
            currencies_to_check.add(currency.upper())

        now = datetime.now(timezone.utc)
        window_start = now - timedelta(minutes=minutes_after)
        window_end = now + timedelta(minutes=minutes_before)

        for ev in self._events:
            if ev["impact"] != "High":
                continue
            if ev["currency"] not in currencies_to_check:
                continue
            if window_start <= ev["datetime"] <= window_end:
                logger.info(
                    f"Near high-impact event: {ev['title']} ({ev['currency']}) "
                    f"at {ev['datetime'].strftime('%H:%M UTC')}"
                )
                return True

        return False

    def get_next_event(self, currency: str) -> Optional[dict]:
        """Get the next upcoming event for a currency (any impact level).

        Args:
            currency: Currency code like "USD" or instrument like "EUR_USD".

        Returns:
            The next event dict, or None if nothing upcoming.
        """
        self.refresh()

        currencies_to_check: set[str] = set()
        if "_" in currency:
            currencies_to_check.update(_extract_currency_from_instrument(currency))
        else:
            currencies_to_check.add(currency.upper())

        now = datetime.now(timezone.utc)
        for ev in self._events:
            if ev["datetime"] < now:
                continue
            if ev["currency"] in currencies_to_check:
                return ev

        return None
