from __future__ import annotations

import json
from pathlib import Path

import MetaTrader5 as mt5
import yaml

from backtest.engine import BacktestConfig
from backtest.meta_labeling import run_meta_labeling_walk_forward_v4
from mt5.market_data import MarketData
from mt5.mt5_connector import MT5Connector


def main() -> None:
    config = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8")) or {}
    trading = config.get("trading", {})
    backtest = config.get("backtest", {})
    nested = config.get("nested_robust", {})
    meta = config.get("meta_labeling", {})

    symbol = str(trading.get("symbol", "XAUUSD.sc"))
    timeframe = str(trading.get("timeframe", "M15"))
    bars = int(meta.get("bars", nested.get("bars", 30000)))
    meta_gate_threshold = float(meta.get("gate_threshold", 0.55))
    min_meta_samples = int(meta.get("min_meta_samples", 120))
    thresholds = [float(x) for x in nested.get(
        "thresholds", [0.50, 0.55, 0.60, 0.65, 0.70, 0.72, 0.75, 0.80]
    )]

    print(f"Collecting {bars:,} closed bars for v4 Meta-Labeling: {symbol} {timeframe}")
    print(f"Meta gate threshold is FIXED at {meta_gate_threshold:.2f} for this baseline test.")
    print("Primary OOF probabilities only are allowed for meta-model training.")

    with MT5Connector():
        market = MarketData()
        raw_df = market.get_bars(
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

    cfg = BacktestConfig(
        initial_balance=float(backtest.get("initial_balance", 1000.0)),
        fixed_lot=float(backtest.get("fixed_lot", 0.01)),
        buy_threshold=float(backtest.get("buy_threshold", 0.72)),
        sell_threshold=float(backtest.get("sell_threshold", 0.72)),
        min_probability_edge=float(backtest.get("min_probability_edge", 0.08)),
        horizon=int(backtest.get("horizon", 8)),
        take_profit_pct=float(backtest.get("take_profit_pct", 0.006)),
        stop_loss_pct=float(backtest.get("stop_loss_pct", 0.003)),
        slippage_points=float(backtest.get("slippage_points", 2.0)),
        commission_per_lot_round_turn=float(backtest.get("commission_per_lot_round_turn", 0.0)),
        point=point,
        contract_size=contract_size,
    )

    baseline, gated, folds, meta_training, summary = run_meta_labeling_walk_forward_v4(
        raw_df,
        cfg,
        thresholds=thresholds,
        outer_train_bars=int(nested.get("outer_train_bars", 12000)),
        inner_validation_bars=int(nested.get("inner_validation_bars", 1500)),
        inner_splits=int(nested.get("inner_splits", 3)),
        outer_test_bars=int(nested.get("outer_test_bars", 2000)),
        step_bars=int(nested.get("step_bars", 2000)),
        purge_bars=int(nested.get("purge_bars", cfg.horizon)),
        min_total_inner_trades=int(nested.get("min_total_inner_trades", 45)),
        meta_gate_threshold=meta_gate_threshold,
        min_meta_samples=min_meta_samples,
    )

    out = Path("reports/meta_labeling_v4")
    out.mkdir(parents=True, exist_ok=True)
    baseline.to_csv(out / "baseline_trades.csv", index=False)
    gated.to_csv(out / "gated_trades.csv", index=False)
    folds.to_csv(out / "fold_comparison.csv", index=False)
    meta_training.to_csv(out / "meta_training_samples.csv", index=False)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print("\nV4 META-LABELING: BASELINE VS GATED")
    print(json.dumps({"baseline": summary["baseline"], "gated": summary["gated"]}, indent=2, ensure_ascii=False, default=str))
    print("\nBASELINE BUY / SELL")
    for row in summary["baseline_sides"]:
        print(row)
    print("\nGATED BUY / SELL")
    for row in summary["gated_sides"]:
        print(row)
    print("\nFOLD COMPARISON")
    if not folds.empty:
        print(folds.to_string(index=False))
    print("\nMETA TRAINING AVAILABILITY")
    if not meta_training.empty:
        print(meta_training.to_string(index=False))
    print("\nReports saved under reports/meta_labeling_v4/")
    print("IMPORTANT: this is development walk-forward only. Do not reuse consumed confirmation windows as clean v4 holdouts.")


if __name__ == "__main__":
    main()
