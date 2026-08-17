from __future__ import annotations

import json
from pathlib import Path

import MetaTrader5 as mt5
import yaml

from backtest.engine import BacktestConfig
from backtest.meta_labeling_v42 import run_meta_labeling_walk_forward_v42
from mt5.market_data import MarketData
from mt5.mt5_connector import MT5Connector


def load_config(path: str = "config/config.yaml") -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    config = load_config()
    trading = config.get("trading", {})
    backtest = config.get("backtest", {})
    nested = config.get("nested_robust", {})
    meta = config.get("meta_labeling_v42", config.get("meta_labeling_v41", {}))

    symbol = str(trading.get("symbol", "XAUUSD.sc"))
    timeframe = str(trading.get("timeframe", "M15"))
    bars = int(meta.get("bars", nested.get("bars", 30000)))
    gate_threshold = float(meta.get("gate_threshold", 0.55))
    buy_harvest_threshold = float(meta.get("buy_harvest_threshold", 0.45))
    sell_harvest_threshold = float(meta.get("sell_harvest_threshold", 0.45))
    min_meta_samples = int(meta.get("min_meta_samples", 250))
    thresholds = [float(x) for x in nested.get(
        "thresholds", [0.50, 0.55, 0.60, 0.65, 0.70, 0.72, 0.75, 0.80]
    )]

    print(f"Collecting {bars:,} closed bars for v4.2 Hybrid SELL Gate: {symbol} {timeframe}")
    print(f"Initial balance: {float(backtest.get('initial_balance', 1000.0)):.2f}")
    print(f"SELL gate threshold is FIXED at {gate_threshold:.2f}; BUY gate is disabled")
    print(f"Broad OOF harvesting retained: BUY>={buy_harvest_threshold:.2f}, SELL>={sell_harvest_threshold:.2f}")

    with MT5Connector():
        market = MarketData()
        df = market.get_bars(symbol=symbol, timeframe=timeframe, count=bars, include_current_bar=False)
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

    baseline, v41, v42, folds, meta_training, summary = run_meta_labeling_walk_forward_v42(
        df,
        cfg,
        thresholds=thresholds,
        outer_train_bars=int(nested.get("outer_train_bars", 12000)),
        inner_validation_bars=int(nested.get("inner_validation_bars", 1500)),
        inner_splits=int(nested.get("inner_splits", 3)),
        outer_test_bars=int(nested.get("outer_test_bars", 2000)),
        step_bars=int(nested.get("step_bars", 2000)),
        purge_bars=int(nested.get("purge_bars", cfg.horizon)),
        min_total_inner_trades=int(nested.get("min_total_inner_trades", 45)),
        meta_gate_threshold=gate_threshold,
        buy_harvest_threshold=buy_harvest_threshold,
        sell_harvest_threshold=sell_harvest_threshold,
        min_meta_samples=min_meta_samples,
    )

    out = Path("reports/meta_labeling_v4_2")
    out.mkdir(parents=True, exist_ok=True)
    baseline.to_csv(out / "baseline_trades.csv", index=False)
    v41.to_csv(out / "v41_all_gated_trades.csv", index=False)
    v42.to_csv(out / "v42_hybrid_trades.csv", index=False)
    folds.to_csv(out / "fold_comparison.csv", index=False)
    meta_training.to_csv(out / "meta_training_availability.csv", index=False)
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nV4.2 HYBRID: BASELINE VS V4.1 ALL-GATED VS V4.2 SELL-ONLY")
    print(json.dumps(
        {
            "baseline": summary["baseline"],
            "v4.1_all_gated": summary["v41_all_gated"],
            "v4.2_hybrid": summary["v42_hybrid"],
        },
        indent=2,
        ensure_ascii=False,
    ))
    for label, key in (
        ("BASELINE BUY / SELL", "baseline_sides"),
        ("V4.1 ALL-GATED BUY / SELL", "v41_sides"),
        ("V4.2 HYBRID BUY / SELL", "v42_sides"),
    ):
        print(f"\n{label}")
        for row in summary[key]:
            print(row)
    print("\nFOLD COMPARISON")
    if not folds.empty:
        print(folds.to_string(index=False))
    print("\nMETA TRAINING AVAILABILITY")
    if not meta_training.empty:
        print(meta_training.to_string(index=False))
    print("\nReports saved under reports/meta_labeling_v4_2/")
    print("IMPORTANT: development walk-forward only; consumed confirmation windows remain off-limits.")


if __name__ == "__main__":
    main()
