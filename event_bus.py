"""Event bus for decoupled communication between components.

Same pattern as the Polymarket bot — proven reliable.
Events are processed synchronously to maintain ordering guarantees.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class Event(str, Enum):
    # Market data events
    TICK = "TICK"                          # New price tick
    CANDLE_CLOSE = "CANDLE_CLOSE"          # Candle completed
    SPREAD_WARNING = "SPREAD_WARNING"      # Spread exceeds threshold

    # Trading events
    SIGNAL = "SIGNAL"                      # Strategy generated a signal
    ORDER_SUBMITTED = "ORDER_SUBMITTED"    # Order sent to broker
    ORDER_FILLED = "ORDER_FILLED"          # Order was filled
    ORDER_REJECTED = "ORDER_REJECTED"      # Order was rejected
    ORDER_CANCELLED = "ORDER_CANCELLED"    # Order was cancelled
    POSITION_OPENED = "POSITION_OPENED"    # Position opened
    POSITION_CLOSED = "POSITION_CLOSED"    # Position closed (SL/TP/manual)

    # Risk events
    RISK_BLOCKED = "RISK_BLOCKED"          # Risk manager blocked a trade
    DRAWDOWN_WARNING = "DRAWDOWN_WARNING"  # Approaching drawdown limit
    DRAWDOWN_CRITICAL = "DRAWDOWN_CRITICAL"  # Critical drawdown level
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"  # Daily loss limit hit
    KILL_SWITCH = "KILL_SWITCH"            # Emergency shutdown

    # System events
    BOT_STARTED = "BOT_STARTED"
    BOT_STOPPED = "BOT_STOPPED"
    CONNECTION_LOST = "CONNECTION_LOST"
    CONNECTION_RESTORED = "CONNECTION_RESTORED"
    HEARTBEAT = "HEARTBEAT"
    ERROR = "ERROR"

    # Account events
    ACCOUNT_UPDATE = "ACCOUNT_UPDATE"      # Account state refreshed
    EQUITY_SNAPSHOT = "EQUITY_SNAPSHOT"     # Periodic equity recording


class EventBus:
    """Simple synchronous event bus with priority support."""

    def __init__(self):
        self._handlers: dict[Event, list[tuple[int, Callable]]] = defaultdict(list)
        self._event_count: dict[Event, int] = defaultdict(int)

    def subscribe(self, event: Event, handler: Callable, priority: int = 10):
        """Subscribe to an event. Lower priority number = called first."""
        self._handlers[event].append((priority, handler))
        self._handlers[event].sort(key=lambda x: x[0])
        logger.debug(f"Subscribed {handler.__qualname__} to {event.value} (priority={priority})")

    def unsubscribe(self, event: Event, handler: Callable):
        """Remove a handler from an event."""
        self._handlers[event] = [
            (p, h) for p, h in self._handlers[event] if h != handler
        ]

    def emit(self, event: Event, data: Any = None):
        """Emit an event to all subscribers."""
        self._event_count[event] += 1
        handlers = self._handlers.get(event, [])

        if not handlers:
            return

        for priority, handler in handlers:
            try:
                handler(data)
            except Exception as e:
                logger.error(
                    f"Error in handler {handler.__qualname__} for {event.value}: {e}",
                    exc_info=True,
                )
                # Don't let one handler crash others
                continue

    def get_stats(self) -> dict[str, int]:
        """Return event emission counts."""
        return {e.value: c for e, c in self._event_count.items() if c > 0}

    def reset(self):
        """Clear all handlers and counts."""
        self._handlers.clear()
        self._event_count.clear()


# Global event bus instance
bus = EventBus()
