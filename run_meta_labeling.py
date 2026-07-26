from __future__ import annotations

import json
from pathlib import Path

import yaml

from backtest.engine import BacktestConfig
from backtest.meta_labeling import run_meta_labeling_walk_forward_v4
from core.mt5_client import MT5Client


def main() -> None:
    config = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
    trading = config["trading"]
    backtest = config["backtest"]
    nested = config["nested_walk_forward"]
    meta = config.get("meta_labeling", {})

    symbol = str(trading["symbol"])
    timeframe = str(trading["timeframe"])
    bars = int(meta.get("bars", nested.get("bars", 30000)))
    meta_gate_threshold = float(meta.get("gate_threshold", 0.55))
    min_meta_samples = int(meta.get("min_meta_samples", 120))

    client = MT5Client()
    client.connect()
    info = client.symbol_info(symbol)
    cfg = BacktestConfig(
        initial_balance=float(backtest["initial_balance"]),
        fixed_lot=float(backtest["fixed_lot"]),
        buy_threshold=float(backtest["buy_threshold"]),
        sell_threshold=float(backtest["sell_threshold"]),
        min_probability_edge=float(backtest["min_probability_edge"]),
        horizon=int(backtest["horizon"]),
        take_profit_pct=float(backtest["take_profit_pct"]),
        stop_loss_pct=float(backtest["stop_loss_pct"]),
        slippage_points=float(backtest["slippage_points"]),
        commission_per_lot_round_turn=float(backtest["commission_per_lot_round_turn"]),
        point=float(info.point),
        contract_size=float(info.trade_contract_size),
    )

    thresholds = [float(x) for x in nested["threshold_candidates"]]
    print(f"Collecting {bars:,} closed bars for v4 Meta-Labeling: {symbol} {timeframe}")
    print(f"Meta gate threshold is FIXED at {meta_gate_threshold:.2f} for this baseline test.")
    print("Primary OOF probabilities only are allowed for meta-model training.")
    raw_df = client.get_closed_bars(symbol, timeframe, bars)

    baseline, gated, folds, meta_training, summary = run_meta_labeling_walk_forward_v4(
        raw_df,
        cfg,
        thresholds=thresholds,
        outer_train_bars=int(nested["outer_train_bars"]),
        inner_validation_bars=int(config.get("nested_robust", {}).get("inner_validation_bars", 1500)),
        inner_splits=int(config.get("nested_robust", {}).get("inner_splits", 3)),
        outer_test_bars=int(nested["outer_test_bars"]),
        step_bars=int(nested["step_bars"]),
        purge_bars=int(nested["purge_bars"]),
        min_total_inner_trades=int(config.get("nested_robust", {}).get("min_total_inner_trades", 45)),
        meta_gate_threshold=meta_gate_threshold,
        min_meta_samples=min_meta_samples,
    )

    out = Path("reports/meta_labeling_v4")
    out.mkdir(parents=True, exist_ok=True)
    baseline.to_csv(out / "baseline_trades.csv", index=False)
    gated.to_csv(out / "gated_trades.csv", index=False)
    folds.to_csv(out / "fold_comparison.csv", index=False)
    meta_training.to_csv(out / "meta_training_samples.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\nV4 META-LABELING: BASELINE VS GATED")
    print(json.dumps({"baseline": summary["baseline"], "gated": summary["gated"]}, indent=2, default=str))
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
    print("IMPORTANT: this is development walk-forward only. Do not reuse the consumed forward-confirmation window as a clean holdout for v4.")


if __name__ == "__main__":
    main()
