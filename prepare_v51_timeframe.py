from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and freeze one v5.1 timeframe bundle")
    parser.add_argument("--timeframe", required=True, choices=["M1", "M5", "M15", "M30", "H1", "H4"])
    parser.add_argument("--bars", type=int, default=30000)
    parser.add_argument("--chunk-size", type=int, default=5000)
    args = parser.parse_args()
    if args.bars < 15000:
        raise ValueError("At least 15,000 bars are required to prepare a timeframe bundle")
    timeframe = args.timeframe.lower()
    snapshot = Path("snapshots") / f"xauusd_sc_{timeframe}_{args.bars}.csv"
    bundle = Path("models") / f"v51_shadow_{timeframe}.joblib"
    subprocess.run(
        [
            sys.executable,
            "export_mt5_snapshot.py",
            "--timeframe", args.timeframe,
            "--bars", str(args.bars),
            "--chunk-size", str(args.chunk_size),
            "--output", str(snapshot),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "freeze_v51_candidate.py",
            "--data", str(snapshot),
            "--output", str(bundle),
        ],
        check=True,
    )
    print(f"READY: {args.timeframe} bundle saved to {bundle}")
    print("STATUS: DEVELOPMENT / DEMO ONLY — run timeframe-specific validation before judging performance")
    print(f"VALIDATE: {sys.executable} run_v51_experiment.py --data {snapshot}")


if __name__ == "__main__":
    main()
