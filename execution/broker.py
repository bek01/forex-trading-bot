"""OANDA broker adapter — handles all API communication.

Supports both practice (paper) and live environments.
Uses OANDA v20 REST API for orders/account and streaming for prices.

Key OANDA API concepts:
- Account ID format: "101-001-12345678-001"
- Instruments use underscore: "EUR_USD" not "EURUSD"
- Units: positive = buy, negative = sell
- Prices: 5 decimal places for most pairs, 3 for JPY
- Streaming: separate endpoint, long-lived HTTP connection
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import httpx

from config import BrokerConfig
from models import (
    AccountState, Candle, Order, OrderStatus, OrderType,
    Position, PositionStatus, Side, Tick,
)

logger = logging.getLogger(__name__)

# OANDA timeframe mapping
TIMEFRAME_MAP = {
    "M1": "M1", "M5": "M5", "M15": "M15", "M30": "M30",
    "H1": "H1", "H4": "H4", "D": "D", "W": "W", "M": "M",
}


class OandaBroker:
    """OANDA v20 REST API adapter."""

    def __init__(self, config: BrokerConfig):
        self.config = config
        self.base_url = config.rest_url
        self.stream_url = config.stream_url
        self.account_id = config.account_id
        self.headers = {
            "Authorization": f"Bearer {config.api_token}",
            "Content-Type": "application/json",
            "Accept-Datetime-Format": "RFC3339",
        }
        self.client = httpx.Client(
            base_url=self.base_url,
            headers=self.headers,
            timeout=30.0,
        )
        self._last_transaction_id = "0"

    # --- Account ---

    def get_account_state(self) -> AccountState:
        """Fetch current account summary."""
        resp = self._get(f"/v3/accounts/{self.account_id}/summary")
        acct = resp["account"]

        state = AccountState(
            balance=float(acct["balance"]),
            equity=float(acct["NAV"]),
            margin_used=float(acct["marginUsed"]),
            margin_available=float(acct["marginAvailable"]),
            unrealized_pnl=float(acct["unrealizedPL"]),
            open_position_count=int(acct["openPositionCount"]),
            timestamp=datetime.now(timezone.utc),
        )
        return state

    # --- Pricing ---

    def get_price(self, instrument: str) -> Tick:
        """Get current bid/ask for an instrument."""
        resp = self._get(
            f"/v3/accounts/{self.account_id}/pricing",
            params={"instruments": instrument},
        )
        price = resp["prices"][0]

        bid = float(price["bids"][0]["price"])
        ask = float(price["asks"][0]["price"])

        # Calculate spread in pips
        pip_size = 0.01 if "JPY" in instrument else 0.0001
        spread_pips = (ask - bid) / pip_size

        return Tick(
            instrument=instrument,
            bid=bid,
            ask=ask,
            spread=spread_pips,
            timestamp=datetime.now(timezone.utc),
        )

    def get_prices(self, instruments: list[str]) -> list[Tick]:
        """Get current prices for multiple instruments."""
        resp = self._get(
            f"/v3/accounts/{self.account_id}/pricing",
            params={"instruments": ",".join(instruments)},
        )
        ticks = []
        for price in resp["prices"]:
            inst = price["instrument"]
            bid = float(price["bids"][0]["price"])
            ask = float(price["asks"][0]["price"])
            pip_size = 0.01 if "JPY" in inst else 0.0001
            ticks.append(Tick(
                instrument=inst,
                bid=bid, ask=ask,
                spread=(ask - bid) / pip_size,
                timestamp=datetime.now(timezone.utc),
            ))
        return ticks

    # --- Candles ---

    def get_candles(
        self,
        instrument: str,
        timeframe: str,
        count: int = 500,
    ) -> list[Candle]:
        """Fetch historical candles."""
        granularity = TIMEFRAME_MAP.get(timeframe, timeframe)
        resp = self._get(
            f"/v3/instruments/{instrument}/candles",
            params={
                "granularity": granularity,
                "count": min(count, 5000),
                "price": "M",  # mid prices
            },
        )

        candles = []
        for c in resp.get("candles", []):
            mid = c["mid"]
            candles.append(Candle(
                instrument=instrument,
                timeframe=timeframe,
                timestamp=datetime.fromisoformat(c["time"].replace("Z", "+00:00")),
                open=float(mid["o"]),
                high=float(mid["h"]),
                low=float(mid["l"]),
                close=float(mid["c"]),
                volume=int(c["volume"]),
                complete=c["complete"],
            ))
        return candles

    # --- Orders ---

    def place_order(self, order: Order) -> Order:
        """
        Place an order with OANDA.
        Always includes stop loss (enforced by risk manager).
        """
        # Build order body
        units = order.units if order.side == Side.BUY else -order.units

        order_body: dict = {
            "type": order.order_type.value,
            "instrument": order.instrument,
            "units": str(int(units)),
            "timeInForce": "FOK" if order.order_type == OrderType.MARKET else "GTC",
        }

        # Price for limit orders
        if order.order_type == OrderType.LIMIT and order.price:
            order_body["price"] = str(order.price)

        # Stop loss (MANDATORY)
        if order.stop_loss:
            order_body["stopLossOnFill"] = {
                "price": str(round(order.stop_loss, 5)),
                "timeInForce": "GTC",
            }

        # Take profit
        if order.take_profit:
            order_body["takeProfitOnFill"] = {
                "price": str(round(order.take_profit, 5)),
            }

        logger.info(
            f"Placing {order.order_type.value} {order.side.value} "
            f"{abs(units)} {order.instrument} "
            f"SL={order.stop_loss} TP={order.take_profit}"
        )

        try:
            resp = self._post(
                f"/v3/accounts/{self.account_id}/orders",
                json={"order": order_body},
            )

            # Check for fill
            if "orderFillTransaction" in resp:
                fill = resp["orderFillTransaction"]
                order.status = OrderStatus.FILLED
                order.broker_order_id = fill["id"]
                order.fill_price = float(fill["price"])
                order.fill_time = datetime.fromisoformat(
                    fill["time"].replace("Z", "+00:00")
                )
                logger.info(
                    f"Order FILLED: {order.instrument} @ {order.fill_price} "
                    f"(id={order.broker_order_id})"
                )

            elif "orderCreateTransaction" in resp:
                create = resp["orderCreateTransaction"]
                order.status = OrderStatus.SUBMITTED
                order.broker_order_id = create["id"]
                logger.info(f"Order SUBMITTED: id={order.broker_order_id}")

            elif "orderRejectTransaction" in resp:
                reject = resp["orderRejectTransaction"]
                order.status = OrderStatus.REJECTED
                reason = reject.get("rejectReason", "unknown")
                logger.warning(f"Order REJECTED: {reason}")

        except Exception as e:
            order.status = OrderStatus.REJECTED
            logger.error(f"Order placement failed: {e}")

        return order

    def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel a pending order."""
        try:
            self._put(
                f"/v3/accounts/{self.account_id}/orders/{broker_order_id}/cancel"
            )
            logger.info(f"Order {broker_order_id} cancelled")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {broker_order_id}: {e}")
            return False

    # --- Positions ---

    def get_open_positions(self) -> list[Position]:
        """Get all open positions from broker."""
        resp = self._get(f"/v3/accounts/{self.account_id}/openPositions")
        positions = []

        for pos in resp.get("positions", []):
            instrument = pos["instrument"]

            # OANDA separates long/short sides
            for side_key, side_val in [("long", Side.BUY), ("short", Side.SELL)]:
                side_data = pos[side_key]
                units = float(side_data.get("units", 0))
                if units == 0:
                    continue

                positions.append(Position(
                    instrument=instrument,
                    side=side_val,
                    units=abs(units),
                    entry_price=float(side_data.get("averagePrice", 0)),
                    unrealized_pnl=float(side_data.get("unrealizedPL", 0)),
                    status=PositionStatus.OPEN,
                    broker_trade_id=side_data.get("tradeIDs", [""])[0],
                ))

        return positions

    def close_position(self, instrument: str, units: Optional[float] = None) -> bool:
        """Close a position (fully or partially)."""
        try:
            body = {}
            if units:
                body["units"] = str(int(units))
            else:
                body["units"] = "ALL"

            self._put(
                f"/v3/accounts/{self.account_id}/positions/{instrument}/close",
                json=body,
            )
            logger.info(f"Closed position: {instrument} ({units or 'ALL'} units)")
            return True
        except Exception as e:
            logger.error(f"Failed to close {instrument}: {e}")
            return False

    def close_trade(self, trade_id: str, units: Optional[float] = None) -> bool:
        """Close a specific trade by ID."""
        try:
            body = {"units": str(int(units)) if units else "ALL"}
            self._put(
                f"/v3/accounts/{self.account_id}/trades/{trade_id}/close",
                json=body,
            )
            logger.info(f"Closed trade {trade_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to close trade {trade_id}: {e}")
            return False

    def close_all_positions(self) -> int:
        """Emergency: close ALL open positions. Returns count closed."""
        positions = self.get_open_positions()
        closed = 0
        for pos in positions:
            if self.close_position(pos.instrument):
                closed += 1
        logger.warning(f"Closed {closed}/{len(positions)} positions (emergency)")
        return closed

    # --- Streaming ---

    def stream_prices(self, instruments: list[str]):
        """
        Generator that yields Tick objects from OANDA's streaming API.
        This is a long-lived HTTP connection — run in a thread.
        """
        params = {"instruments": ",".join(instruments)}
        url = f"{self.stream_url}/v3/accounts/{self.account_id}/pricing/stream"

        with httpx.stream(
            "GET", url,
            headers=self.headers,
            params=params,
            timeout=None,
        ) as response:
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("type") == "PRICE":
                        inst = data["instrument"]
                        bid = float(data["bids"][0]["price"])
                        ask = float(data["asks"][0]["price"])
                        pip_size = 0.01 if "JPY" in inst else 0.0001

                        yield Tick(
                            instrument=inst,
                            bid=bid, ask=ask,
                            spread=(ask - bid) / pip_size,
                            timestamp=datetime.fromisoformat(
                                data["time"].replace("Z", "+00:00")
                            ),
                        )
                    elif data.get("type") == "HEARTBEAT":
                        pass  # connection alive
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    # --- HTTP helpers ---

    def _get(self, path: str, params: dict = None) -> dict:
        resp = self.client.get(path, params=params)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, json: dict = None) -> dict:
        resp = self.client.post(path, json=json)
        resp.raise_for_status()
        return resp.json()

    def _put(self, path: str, json: dict = None) -> dict:
        resp = self.client.put(path, json=json)
        resp.raise_for_status()
        return resp.json()

    def close(self):
        """Clean up HTTP client."""
        self.client.close()
