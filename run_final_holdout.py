from __future__ import annotations

import hashlib
import json
from pathlib import Path

import MetaTrader5 as mt5
import yaml

from backtest.engine import BacktestConfig
from backtest.holdout import (
    LOCKED_CANDIDATES,
    build_untouched_holdout_predictions,
    evaluate_locked_candidates,
)
from mt5.market_data import MarketData
from mt5.mt5_connector import MT5Connector


def load_config(path: str = "config/config.yaml") -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    cfg = load_config()
    trading = cfg.get("trading", {})
    bt = cfg.get("backtest", {})
    holdout = cfg.get("final_holdout", {})

    symbol = str(trading.get("symbol", "XAUUSD.sc"))
    timeframe = str(trading.get("timeframe", "M15"))
    bars = int(holdout.get("bars", 75000))

    print("FINAL UNTOUCHED HOLDOUT TEST")
    print("No optimizer. No threshold sweep. Locked candidates:")
    for c in LOCKED_CANDIDATES:
        print(f"  - {c.name}: {c.method} threshold={c.threshold:.3f}")
    print(f"\nCollecting {bars:,} closed bars: {symbol} {timeframe}")

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
        initial_balance=float(bt.get("initial_balance", 1000.0)),
        fixed_lot=float(bt.get("fixed_lot", 0.01)),
        buy_threshold=0.60,  # ignored by locked candidate replay
        sell_threshold=0.60,
        min_probability_edge=float(bt.get("min_probability_edge", 0.08)),
        horizon=int(bt.get("horizon", 8)),
        take_profit_pct=float(bt.get("take_profit_pct", 0.006)),
        stop_loss_pct=float(bt.get("stop_loss_pct", 0.003)),
        slippage_points=float(bt.get("slippage_points", 2.0)),
        commission_per_lot_round_turn=float(bt.get("commission_per_lot_round_turn", 0.0)),
        point=point,
        contract_size=contract_size,
    )

    predictions, manifest = build_untouched_holdout_predictions(
        raw_df=df,
        cfg=backtest_cfg,
        recent_bars_excluded=int(holdout.get("recent_bars_excluded", 50000)),
        train_bars=int(holdout.get("train_bars", 12000)),
        calibration_bars=int(holdout.get("calibration_bars", 2000)),
        holdout_bars=int(holdout.get("holdout_bars", 5000)),
        purge_bars=int(holdout.get("purge_bars", backtest_cfg.horizon)),
    )

    results, side_results, trade_logs = evaluate_locked_candidates(
        predictions,
        backtest_cfg,
    )

    manifest["symbol"] = symbol
    manifest["timeframe"] = timeframe
    manifest["initial_balance"] = backtest_cfg.initial_balance
    manifest["fixed_lot"] = backtest_cfg.fixed_lot
    manifest["take_profit_pct"] = backtest_cfg.take_profit_pct
    manifest["stop_loss_pct"] = backtest_cfg.stop_loss_pct
    manifest["slippage_points"] = backtest_cfg.slippage_points
    manifest["commission_per_lot_round_turn"] = backtest_cfg.commission_per_lot_round_turn
    manifest_json = json.dumps(manifest, sort_keys=True, ensure_ascii=False)
    manifest["lock_hash_sha256"] = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()

    report_dir = Path("reports/final_holdout")
    report_dir.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(report_dir / "predictions.csv", index=False)
    results.to_csv(report_dir / "candidate_results.csv", index=False)
    side_results.to_csv(report_dir / "candidate_side_results.csv", index=False)
    for name, trades in trade_logs.items():
        trades.to_csv(report_dir / f"trades_{name}.csv", index=False)

    with (report_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\nLOCKED HOLDOUT WINDOW")
    print(f"  Train       : {manifest['train_start']} -> {manifest['train_end']}")
    print(f"  Calibration : {manifest['calibration_start']} -> {manifest['calibration_end']}")
    print(f"  HOLDOUT     : {manifest['holdout_start']} -> {manifest['holdout_end']}")
    print(f"  Excluded dev: {manifest['excluded_recent_start']} -> {manifest['excluded_recent_end']}")
    print(f"  Lock SHA256 : {manifest['lock_hash_sha256']}")

    print("\nFINAL HOLDOUT RESULTS (no optimization):")
    display_cols = [
        "candidate", "method", "threshold", "trades", "win_rate",
        "profit_factor", "net_profit", "expectancy", "max_drawdown_pct",
        "buy_profit_factor", "buy_net_profit", "sell_profit_factor", "sell_net_profit",
    ]
    print(results[display_cols].to_string(index=False))

    print("\nReports saved under reports/final_holdout/")
    print("IMPORTANT: Do not tune thresholds/models using this holdout. Treat it as consumed after this run.")


if __name__ == "__main__":
    main()
