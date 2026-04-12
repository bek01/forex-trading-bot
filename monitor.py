#!/usr/bin/env python3
"""
Trading Bot Monitor — Dashboard, health checks, and reports.

Usage:
    python monitor.py              # Full dashboard
    python monitor.py health       # Quick health check (for cron)
    python monitor.py trades       # Recent trades
    python monitor.py equity       # Equity history
    python monitor.py strategies   # Strategy performance breakdown
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

from db.database import Database
from config import get_config


def dashboard(db: Database):
    """Full monitoring dashboard."""
    print("=" * 60)
    print("  TRADING BOT DASHBOARD")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    # Overall stats
    stats = db.get_overall_stats()
    total = stats["total_trades"]
    wins = stats["wins"] or 0
    losses = stats["losses"] or 0
    pnl = stats["total_pnl"] or 0
    wr = wins / total * 100 if total > 0 else 0

    print(f"\n📊 OVERALL PERFORMANCE")
    print(f"  Total Trades: {total}")
    print(f"  Wins/Losses:  {wins}/{losses} ({wr:.1f}% WR)")
    print(f"  Total P&L:    ${pnl:+,.2f}")
    print(f"  Avg P&L:      ${stats['avg_pnl'] or 0:+.2f}/trade")

    # Open positions
    open_pos = db.get_open_positions()
    print(f"\n📈 OPEN POSITIONS ({len(open_pos)})")
    if open_pos:
        for p in open_pos:
            print(f"  {p['side']:4s} {p['instrument']:8s} @ {p['entry_price']:.5f} "
                  f"SL={p['stop_loss']:.5f} [{p['strategy']}]")
    else:
        print("  None")

    # Recent closed trades
    recent = db.get_closed_positions(limit=10)
    print(f"\n📋 RECENT TRADES (last 10)")
    for t in recent:
        emoji = "✅" if (t["realized_pnl"] or 0) > 0 else "❌"
        print(f"  {emoji} {t['side']:4s} {t['instrument']:8s} "
              f"${t['realized_pnl'] or 0:+.2f} — {t['close_reason'] or 'N/A'} "
              f"[{t['strategy']}]")

    # Daily P&L
    daily = db.get_daily_pnl(days=7)
    if daily:
        print(f"\n📅 DAILY P&L (last 7 days)")
        for d in daily:
            emoji = "📈" if (d["realized_pnl"] or 0) >= 0 else "📉"
            print(f"  {emoji} {d['date']}: ${d['realized_pnl'] or 0:+.2f} "
                  f"({d['trades_closed'] or 0} trades, "
                  f"W:{d['wins'] or 0}/L:{d['losses'] or 0})")

    # Equity history
    equity = db.get_equity_history(hours=24)
    if equity:
        latest = equity[-1]
        print(f"\n💰 EQUITY (latest)")
        print(f"  Balance:  ${latest['balance']:,.2f}")
        print(f"  Equity:   ${latest['equity']:,.2f}")
        print(f"  Drawdown: {latest['drawdown_pct']:.1f}%")
        print(f"  Peak:     ${latest['peak_equity']:,.2f}")


def health_check(db: Database):
    """Quick health check for cron monitoring."""
    equity = db.get_equity_history(hours=1)
    if not equity:
        print("WARNING: No equity snapshots in last hour — bot may be down")
        sys.exit(1)

    latest = equity[-1]
    dd = latest["drawdown_pct"]

    if dd > 10:
        print(f"CRITICAL: Drawdown {dd:.1f}%")
        sys.exit(2)
    elif dd > 5:
        print(f"WARNING: Drawdown {dd:.1f}%")
        sys.exit(1)
    else:
        print(f"OK: equity=${latest['equity']:.2f}, dd={dd:.1f}%")


def show_strategies(db: Database):
    """Show per-strategy performance."""
    print("\n📊 STRATEGY PERFORMANCE")
    print("-" * 60)

    for strategy in ["mean_reversion", "trend_following", "london_breakout"]:
        stats = db.get_strategy_stats(strategy)
        if stats["total_trades"] == 0:
            continue
        print(f"\n  {strategy.upper()}")
        print(f"    Trades: {stats['total_trades']} "
              f"(W:{stats['wins']} L:{stats['losses']} WR:{stats['win_rate']:.1f}%)")
        print(f"    P&L:    ${stats['total_pnl']:+,.2f}")
        print(f"    Avg:    W=${stats['avg_win']:.2f} L=${stats['avg_loss']:.2f}")
        print(f"    PF:     {stats['profit_factor']:.2f}")
        print(f"    Best:   ${stats['best_trade']:.2f} | Worst: ${stats['worst_trade']:.2f}")


def main():
    config = get_config()
    db = Database(config.db_path)

    cmd = sys.argv[1] if len(sys.argv) > 1 else "dashboard"

    try:
        if cmd == "health":
            health_check(db)
        elif cmd == "trades":
            recent = db.get_closed_positions(limit=20)
            for t in recent:
                emoji = "✅" if (t["realized_pnl"] or 0) > 0 else "❌"
                print(f"{emoji} {t['closed_at'] or '?':20s} {t['side']:4s} "
                      f"{t['instrument']:8s} ${t['realized_pnl'] or 0:+.2f} "
                      f"[{t['strategy']}] {t['close_reason'] or ''}")
        elif cmd == "equity":
            equity = db.get_equity_history(hours=48)
            for e in equity:
                print(f"{e['timestamp']:20s} eq=${e['equity']:,.2f} "
                      f"dd={e['drawdown_pct']:.1f}% "
                      f"daily=${e['daily_pnl']:+.2f}")
        elif cmd == "strategies":
            show_strategies(db)
        else:
            dashboard(db)
    finally:
        db.close()


if __name__ == "__main__":
    main()
