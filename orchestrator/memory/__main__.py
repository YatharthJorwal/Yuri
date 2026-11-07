"""
Maintenance CLI for memory.db. Run from orchestrator/:

    python -m memory --facts                  # live facts (add --all to include superseded/forgotten)
    python -m memory --events 30              # the last 30 audit events
    python -m memory --restore 12             # undo a supersede or a forget of fact 12
    python -m memory --pin 12 / --unpin 12    # pinned facts are only replaced by an explicit correction
    python -m memory --import-luna PATH       # one-time, additive copy from Luna's memory.db

Nothing here runs automatically; the app never imports Luna's data on its own.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys

from . import migrate, store


def _when(ts: float | None) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "-"


def _status(fact: dict) -> str:
    marker = fact["superseded_by"]
    if marker is None:
        return "live"
    return "forgotten" if marker == migrate.FORGOTTEN else f"replaced by #{marker}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m memory", description="Inspect and repair memory.db")
    ap.add_argument("--facts", action="store_true", help="list facts")
    ap.add_argument("--all", action="store_true", help="with --facts: include superseded and forgotten")
    ap.add_argument("--events", type=int, metavar="N", help="show the last N audit events")
    ap.add_argument("--restore", type=int, metavar="ID", help="restore a superseded/forgotten fact")
    ap.add_argument("--pin", type=int, metavar="ID")
    ap.add_argument("--unpin", type=int, metavar="ID")
    ap.add_argument("--import-luna", metavar="PATH", help="copy facts and episodes from Luna's memory.db")
    args = ap.parse_args(argv)

    did_something = False
    if args.import_luna:
        migrate.import_luna(args.import_luna)
        did_something = True
    if args.restore is not None:
        ok = store.restore_fact(args.restore)
        if ok:
            store.log_event("restore", "owner", {"fact_id": args.restore})
        print(f"fact #{args.restore}: {'restored (its vector returns on the next writer pass)' if ok else 'nothing to restore'}")
        did_something = True
    for fact_id, pinned in ((args.pin, True), (args.unpin, False)):
        if fact_id is not None:
            store.set_pinned(fact_id, pinned)
            store.log_event("pin" if pinned else "unpin", "owner", {"fact_id": fact_id})
            print(f"fact #{fact_id}: {'pinned' if pinned else 'unpinned'}")
            did_something = True
    if args.facts:
        for f in store.get_fact_rows(include_superseded=args.all):
            pin = " [pinned]" if f["pinned"] else ""
            print(f"#{f['id']:<4} {_status(f):<18} conf={f['confidence']:.2f} {f['source']:<11}{pin} {f['content']}")
        did_something = True
    if args.events:
        for e in reversed(store.get_events(limit=args.events)):
            print(f"{_when(e['ts'])}  {e['kind']:<17} {e['actor']:<8} {e['detail']}")
        did_something = True
    if not did_something:
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
