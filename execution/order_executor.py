"""Order executor — bridges signals from strategies to the broker.

Responsibilities:
1. Receive approved signals from risk manager
2. Convert Signal → Order with correct position size
3. Submit to broker
4. Track order state
5. Emit events for fills/rejections
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from config import AppConfig
from event_bus import Event, bus
from execution.broker import OandaBroker
from models import Order, OrderStatus, OrderType, Position, PositionStatus, Signal, Side

logger = logging.getLogger(__name__)


class OrderExecutor:
    """Executes trading signals by placing orders with the broker."""

    def __init__(self, broker: OandaBroker, config: AppConfig):
        self.broker = broker
        self.config = config
        self.pending_orders: dict[str, Order] = {}
        self._order_history: list[Order] = []

    def execute_signal(self, signal: Signal, units: float) -> Optional[Order]:
        """
        Execute a signal by placing an order with the broker.

        Args:
            signal: The trading signal
            units: Position size (already calculated by risk manager)

        Returns:
            Order object with updated status, or None if failed
        """
        order = Order(
            signal_id=signal.id,
            instrument=signal.instrument,
            side=signal.side,
            order_type=OrderType.MARKET,  # market orders for simplicity
            units=units,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            strategy=signal.strategy,
        )

        logger.info(
            f"Executing: {order.side.value} {order.units:.0f} {order.instrument} "
            f"SL={order.stop_loss:.5f} TP={order.take_profit:.5f} "
            f"[{order.strategy}]"
        )

        # Submit to broker
        order = self.broker.place_order(order)
        self._order_history.append(order)

        # Emit events based on result
        if order.status == OrderStatus.FILLED:
            bus.emit(Event.ORDER_FILLED, order)

            # Create and emit position
            position = Position(
                instrument=order.instrument,
                side=order.side,
                units=order.units,
                entry_price=order.fill_price or 0,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
                strategy=order.strategy,
                broker_trade_id=order.broker_order_id,
                opened_at=order.fill_time or datetime.now(timezone.utc),
            )
            bus.emit(Event.POSITION_OPENED, position)
            return order

        elif order.status == OrderStatus.SUBMITTED:
            self.pending_orders[order.id] = order
            bus.emit(Event.ORDER_SUBMITTED, order)
            return order

        elif order.status == OrderStatus.REJECTED:
            bus.emit(Event.ORDER_REJECTED, order)
            logger.warning(f"Order rejected: {order.instrument} [{order.strategy}]")
            return order

        return None

    def check_pending_orders(self):
        """Poll broker for updates on pending orders."""
        # For market orders this is rarely needed, but for limit orders
        # we'd check fill status here
        pass

    def cancel_all_orders(self) -> int:
        """Cancel all pending orders. Returns count cancelled."""
        cancelled = 0
        for order_id, order in list(self.pending_orders.items()):
            if order.broker_order_id:
                if self.broker.cancel_order(order.broker_order_id):
                    order.status = OrderStatus.CANCELLED
                    bus.emit(Event.ORDER_CANCELLED, order)
                    del self.pending_orders[order_id]
                    cancelled += 1
        return cancelled

    def get_stats(self) -> dict:
        """Order execution statistics."""
        total = len(self._order_history)
        filled = sum(1 for o in self._order_history if o.status == OrderStatus.FILLED)
        rejected = sum(1 for o in self._order_history if o.status == OrderStatus.REJECTED)
        return {
            "total_orders": total,
            "filled": filled,
            "rejected": rejected,
            "pending": len(self.pending_orders),
            "fill_rate": f"{filled/total*100:.1f}%" if total > 0 else "N/A",
        }
