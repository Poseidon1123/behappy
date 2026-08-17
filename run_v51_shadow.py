from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from shadow.engine import load_state, load_verified_bundle, process_closed_bars


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen v5.1 in SHADOW mode (no broker orders)")
    parser.add_argument("--bundle", type=Path, default=Path("models/v51_shadow_bundle.joblib"))
    parser.add_argument("--state", type=Path, default=Path("shadow_state/v51_state.json"))
    parser.add_argument("--events", type=Path, default=Path("shadow_logs/v51_events.csv"))
    parser.add_argument("--bars", type=int, default=10000)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def _cycle(args: argparse.Namespace) -> dict:
    from mt5.market_data import MarketData
    from mt5.mt5_connector import MT5Connector

    bundle, manifest = load_verified_bundle(args.bundle)
    state = load_state(args.state, bundle, manifest)
    snapshot = bundle["snapshot_manifest"]
    with MT5Connector():
        bars = MarketData().get_bars_chunked(snapshot["symbol"], snapshot["timeframe"], count=args.bars, chunk_size=min(5000, args.bars), require_full_count=False)
    return process_closed_bars(bars, bundle, manifest, state, args.events, args.state)


def main() -> None:
    args = _arguments()
    if args.bars < 1000:
        raise ValueError("--bars must be at least 1000 for causal feature context")
    if args.poll_seconds < 5:
        raise ValueError("--poll-seconds must be at least 5")
    print("V5.1 SHADOW/PAPER — NO BROKER ORDERS")
    while True:
        print(json.dumps(_cycle(args), indent=2))
        if not args.watch:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
