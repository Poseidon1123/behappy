from __future__ import annotations

import argparse
from pathlib import Path

import MetaTrader5 as mt5
import yaml

from data.snapshot import save_snapshot
from mt5.market_data import MarketData
from mt5.mt5_connector import MT5Connector


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze closed MT5 bars for reproducible research")
    parser.add_argument("--bars", type=int, default=30000)
    parser.add_argument("--output", type=Path, default=Path("snapshots/xauusd_sc_m15_30000.csv"))
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    trading = config.get("trading", {})
    symbol = str(trading.get("symbol", "XAUUSD.sc"))
    timeframe = str(trading.get("timeframe", "M15"))

    print(f"Downloading {args.bars:,} CLOSED bars: {symbol} {timeframe}")
    with MT5Connector():
        frame = MarketData().get_bars(
            symbol=symbol,
            timeframe=timeframe,
            count=args.bars,
            include_current_bar=False,
        )
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"Could not read symbol info for {symbol}")
        csv_file, manifest_file, manifest = save_snapshot(
            frame,
            args.output,
            symbol=symbol,
            timeframe=timeframe,
            point=float(info.point),
            contract_size=float(info.trade_contract_size),
        )

    print(f"CSV:      {csv_file}")
    print(f"Manifest: {manifest_file}")
    print(f"Rows:     {manifest['rows']:,}")
    print(f"Period:   {manifest['first_bar_utc']} -> {manifest['last_bar_utc']}")
    print(f"SHA-256:  {manifest['sha256']}")
    print("Snapshot is frozen. Do not edit the CSV after this point.")


if __name__ == "__main__":
    main()
