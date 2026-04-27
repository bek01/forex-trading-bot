#!/usr/bin/env python3
"""Reconcile orphan positions/orders against the OANDA broker.

A position row with status='OPEN' is an orphan if the broker shows the
trade is not actually open (state=CLOSED, or the trade ID returns 404).
Same idea for orders that never reached a terminal state.

Defaults to dry-run. Pass --apply to write changes.

Usage:
    python -m scripts.reconcile_orphans                # both accounts, dry-run
    python -m scripts.reconcile_orphans --apply        # both accounts, write
    python -m scripts.reconcile_orphans --account live # one account only
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

ACCOUNTS = {
    "live": {
        "env": ROOT / ".env.live",
        "db":  ROOT / "db" / "trades_live.db",
    },
    "demo": {
        "env": ROOT / ".env",
        "db":  ROOT / "db" / "trades.db",
    },
}

ORDER_TERMINAL = ("FILLED", "CANCELLED", "REJECTED", "EXPIRED")


def load_account(env_path: Path) -> tuple[str, str, str]:
    load_dotenv(env_path, override=True)
    acct = os.getenv("OANDA_ACCOUNT_ID")
    tok = (os.getenv("OANDA_API_TOKEN") or os.getenv("OANDA_API_KEY")
           or os.getenv("OANDA_TOKEN"))
    env = os.getenv("OANDA_ENVIRONMENT", "practice")
    host = "https://api-fxtrade.oanda.com" if env == "live" \
           else "https://api-fxpractice.oanda.com"
    if not acct or not tok:
        raise RuntimeError(f"Missing OANDA creds in {env_path}")
    return acct, tok, host


def fetch_trade(client: httpx.Client, host: str, acct: str, tid: str) -> Optional[dict]:
    """Return broker trade dict, or None on 404 (orphan)."""
    r = client.get(f"{host}/v3/accounts/{acct}/trades/{tid}")
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json().get("trade")


def fetch_order(client: httpx.Client, host: str, acct: str, oid: str) -> Optional[dict]:
    r = client.get(f"{host}/v3/accounts/{acct}/orders/{oid}")
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json().get("order")


def reconcile_one(label: str, env_path: Path, db_path: Path, apply: bool) -> dict:
    print(f"\n{'='*70}\n{label.upper()}  ({env_path.name} → {db_path.name})\n{'='*70}")
    acct, tok, host = load_account(env_path)
    print(f"OANDA account: {acct}  host: {host}")

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    pos_open = cur.execute("""
        SELECT id, instrument, side, broker_trade_id, opened_at, status
          FROM positions WHERE status='OPEN'
        ORDER BY opened_at
    """).fetchall()

    ord_pending = cur.execute(f"""
        SELECT id, signal_id, instrument, side, status, broker_order_id, created_at
          FROM orders
         WHERE status NOT IN {ORDER_TERMINAL}
        ORDER BY created_at
    """).fetchall()

    print(f"DB positions OPEN: {len(pos_open)}")
    print(f"DB orders pending: {len(ord_pending)}")

    pos_changes = []   # list of (db_row, action, broker_state, exit_price, realized_pnl, close_time, reason)
    ord_changes = []   # list of (db_row, action, broker_state)

    headers = {"Authorization": f"Bearer {tok}"}
    with httpx.Client(timeout=15, headers=headers) as client:
        for row in pos_open:
            btid = str(row["broker_trade_id"] or "")
            if not btid:
                pos_changes.append((row, "SKIP_NO_BTID", None, None, None, None, None))
                continue
            try:
                trade = fetch_trade(client, host, acct, btid)
            except Exception as e:
                pos_changes.append((row, f"ERROR: {e}", None, None, None, None, None))
                continue
            if trade is None:
                # 404 → orphan, broker has no record
                close_time = datetime.now(timezone.utc).isoformat()
                pos_changes.append((row, "MARK_CLOSED_ORPHAN_404",
                                    "NOT_FOUND", 0.0, 0.0, close_time, "broker_orphan"))
            elif trade.get("state") == "CLOSED":
                close_price = float(trade.get("averageClosePrice") or 0.0)
                realized = float(trade.get("realizedPL") or 0.0)
                close_time = trade.get("closeTime") or datetime.now(timezone.utc).isoformat()
                pos_changes.append((row, "MARK_CLOSED_FROM_BROKER",
                                    "CLOSED", close_price, realized, close_time, "reconcile_late"))
            else:
                pos_changes.append((row, "STILL_OPEN_AT_BROKER",
                                    trade.get("state"), None, None, None, None))

        for row in ord_pending:
            oid = str(row["broker_order_id"] or "")
            if not oid:
                ord_changes.append((row, "SKIP_NO_BROKER_OID", None))
                continue
            try:
                order = fetch_order(client, host, acct, oid)
            except Exception as e:
                ord_changes.append((row, f"ERROR: {e}", None))
                continue
            if order is None:
                ord_changes.append((row, "MARK_CANCELLED_ORPHAN_404", "NOT_FOUND"))
            else:
                state = order.get("state", "UNKNOWN")
                if state in ORDER_TERMINAL:
                    ord_changes.append((row, f"MARK_{state}_FROM_BROKER", state))
                else:
                    ord_changes.append((row, "STILL_PENDING_AT_BROKER", state))

    # Print plan
    print("\n--- Position plan ---")
    for row, action, bstate, exit_price, realized, close_time, reason in pos_changes:
        print(f"  [{action}] db_id={row['id'][:8]} {row['instrument']:8s} "
              f"{row['side']:4s} btid={row['broker_trade_id']} "
              f"opened={(row['opened_at'] or '')[:19]} broker_state={bstate} "
              f"exit={exit_price} pnl={realized}")

    print("\n--- Order plan ---")
    for row, action, bstate in ord_changes:
        print(f"  [{action}] db_id={row['id'][:8] if isinstance(row['id'], str) else row['id']} "
              f"{row['instrument']:8s} {row['side'] or '':4s} "
              f"db_status={row['status']} broker_oid={row['broker_order_id']} broker_state={bstate}")

    if not apply:
        print("\n(dry-run — no DB writes)")
        return {"positions": len(pos_changes), "orders": len(ord_changes)}

    # Apply
    pos_written = 0
    for row, action, bstate, exit_price, realized, close_time, reason in pos_changes:
        if action.startswith("MARK_CLOSED"):
            cur.execute("""UPDATE positions
                              SET status='CLOSED', exit_price=?, realized_pnl=?,
                                  close_reason=?, closed_at=?
                            WHERE id=? AND status='OPEN'""",
                        (exit_price, realized, reason, close_time, row["id"]))
            pos_written += cur.rowcount

    ord_written = 0
    for row, action, bstate in ord_changes:
        if action.startswith("MARK_"):
            new_state = bstate if bstate in ORDER_TERMINAL else "CANCELLED"
            cur.execute("""UPDATE orders
                              SET status=?
                            WHERE id=? AND status NOT IN ('FILLED','CANCELLED','REJECTED','EXPIRED')""",
                        (new_state, row["id"]))
            ord_written += cur.rowcount

    con.commit()
    print(f"\nApplied: positions updated={pos_written}, orders updated={ord_written}")
    return {"positions": pos_written, "orders": ord_written}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry-run)")
    ap.add_argument("--account", choices=["live", "demo", "all"], default="all")
    args = ap.parse_args()

    targets = list(ACCOUNTS.keys()) if args.account == "all" else [args.account]
    for label in targets:
        cfg = ACCOUNTS[label]
        if not cfg["env"].exists():
            print(f"skip {label}: {cfg['env']} not found")
            continue
        if not cfg["db"].exists():
            print(f"skip {label}: {cfg['db']} not found")
            continue
        reconcile_one(label, cfg["env"], cfg["db"], args.apply)


if __name__ == "__main__":
    main()
