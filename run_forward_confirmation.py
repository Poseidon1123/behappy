from __future__ import annotations

import json
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd
import yaml

from backtest.engine import BacktestConfig
from backtest.forward_confirmation import load_frozen_manifest, run_forward_confirmation
from mt5.market_data import MarketData
from mt5.mt5_connector import MT5Connector


def load_config(path: str = "config/config.yaml") -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    cfg = load_config()
    manifest, lock_hash = load_frozen_manifest()

    trading = cfg.get("trading", {})
    symbol = str(trading.get("symbol", "XAUUSD.sc"))
    timeframe = str(trading.get("timeframe", "M15"))

    if symbol != manifest["symbol"] or timeframe != manifest["timeframe"]:
        raise RuntimeError(
            f"Frozen v3.1 expects {manifest['symbol']} {manifest['timeframe']}, "
            f"but runtime config is {symbol} {timeframe}."
        )

    # 30k bars is enough for the frozen 12k history plus the post-cutoff confirmation window.
    bars = 30000
    print("FROZEN v3.1 FORWARD CONFIRMATION")
    print("No optimizer. No threshold sweep. No feature changes.")
    print(f"Freeze SHA256: {lock_hash}")
    print(f"Development cutoff: {manifest['development_outer_test_end_utc']}")
    print(f"Collecting {bars:,} closed bars: {symbol} {timeframe}")

    with MT5Connector():
        market = MarketData()
        df = market.get_bars(
            symbol=symbol,
            timeframe=timeframe,
            count=bars,
            include_current_bar=False,
        )
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"Could not read symbol info for {symbol}")
        point = float(info.point)
        contract_size = float(info.trade_contract_size)

    backtest_cfg = BacktestConfig(
        initial_balance=float(manifest["initial_balance"]),
        fixed_lot=float(manifest["fixed_lot"]),
        buy_threshold=0.5,  # ignored; selected from frozen inner process
        sell_threshold=0.5,  # ignored; selected from frozen inner process
        min_probability_edge=float(manifest["min_probability_edge"]),
        horizon=int(manifest["horizon"]),
        take_profit_pct=float(manifest["take_profit_pct"]),
        stop_loss_pct=float(manifest["stop_loss_pct"]),
        slippage_points=float(manifest["slippage_points"]),
        commission_per_lot_round_turn=float(manifest["commission_per_lot_round_turn"]),
        point=point,
        contract_size=contract_size,
    )

    trades, predictions, side_summary, scores, details, summary = run_forward_confirmation(
        df,
        backtest_cfg,
        manifest,
    )
    summary["freeze_sha256"] = lock_hash

    report_dir = Path("reports/forward_confirmation_v3_1")
    report_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(report_dir / "trades.csv", index=False)
    predictions.to_csv(report_dir / "predictions.csv", index=False)
    side_summary.to_csv(report_dir / "side_summary.csv", index=False)
    scores.to_csv(report_dir / "frozen_inner_candidate_scores.csv", index=False)
    details.to_csv(report_dir / "frozen_inner_split_diagnostics.csv", index=False)
    with (report_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (report_dir / "freeze_manifest_snapshot.json").open("w", encoding="utf-8") as f:
        json.dump({"freeze_sha256": lock_hash, **manifest}, f, indent=2, ensure_ascii=False)

    print("\nLOCKED PRE-CONFIRMATION SELECTION")
    print(f"BUY  : {summary['buy_architecture']} threshold={summary['buy_threshold']:.2f}")
    print(f"SELL : {summary['sell_architecture']} threshold={summary['sell_threshold']:.2f}")
    print(f"Train: {summary['train_start_utc']} -> {summary['train_end_utc']}")
    print(f"Test : {summary['confirmation_start_utc']} -> {summary['confirmation_end_utc']}")
    print(f"Bars : {summary['confirmation_bars']}")

    print("\nFORWARD CONFIRMATION SUMMARY")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\nBUY / SELL BREAKDOWN")
    print(side_summary.to_string(index=False))

    print(f"\nReports saved under {report_dir}/")
    print(
        "IMPORTANT: This confirmation window is now consumed once inspected. "
        "Do not tune v3.1 using these results and still call it forward confirmation."
    )


if __name__ == "__main__":
    main()
