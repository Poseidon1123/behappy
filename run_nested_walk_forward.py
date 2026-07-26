from __future__ import annotations

import json
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd
import yaml

from backtest.analysis import analyze_sides
from backtest.engine import BacktestConfig
from backtest.nested_walk_forward import run_nested_walk_forward_v3
from mt5.market_data import MarketData
from mt5.mt5_connector import MT5Connector


def load_config(path: str = "config/config.yaml") -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    cfg = load_config()
    trading = cfg.get("trading", {})
    bt = cfg.get("backtest", {})
    nested = cfg.get("nested_walk_forward", {})

    symbol = str(trading.get("symbol", "XAUUSD.sc"))
    timeframe = str(trading.get("timeframe", "M15"))
    bars = int(nested.get("bars", 30000))
    thresholds = [
        float(x)
        for x in nested.get(
            "threshold_candidates",
            [0.50, 0.55, 0.60, 0.65, 0.70, 0.72, 0.75, 0.80],
        )
    ]

    print(f"Collecting {bars:,} closed bars for Nested Walk-Forward v3: {symbol} {timeframe}")
    print("BUY architecture: fixed A_m15_baseline")
    print("SELL candidates: A_m15_baseline / B_m15_raw_h1 / E_m15_regime_only")
    print(f"Threshold candidates: {thresholds}")

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
        initial_balance=float(bt.get("initial_balance", 10000.0)),
        fixed_lot=float(bt.get("fixed_lot", 0.01)),
        buy_threshold=float(bt.get("buy_threshold", 0.72)),
        sell_threshold=float(bt.get("sell_threshold", 0.72)),
        min_probability_edge=float(bt.get("min_probability_edge", 0.08)),
        horizon=int(bt.get("horizon", 8)),
        take_profit_pct=float(bt.get("take_profit_pct", 0.006)),
        stop_loss_pct=float(bt.get("stop_loss_pct", 0.003)),
        slippage_points=float(bt.get("slippage_points", 2.0)),
        commission_per_lot_round_turn=float(bt.get("commission_per_lot_round_turn", 0.0)),
        point=point,
        contract_size=contract_size,
    )

    trades, folds, selections, inner_diagnostics, summary = run_nested_walk_forward_v3(
        df,
        backtest_cfg,
        thresholds=thresholds,
        outer_train_bars=int(nested.get("outer_train_bars", 12000)),
        inner_validation_bars=int(nested.get("inner_validation_bars", 2000)),
        outer_test_bars=int(nested.get("outer_test_bars", 2000)),
        step_bars=int(nested.get("step_bars", 2000)),
        purge_bars=int(nested.get("purge_bars", backtest_cfg.horizon)),
        min_inner_trades=int(nested.get("min_inner_trades", 20)),
    )

    report_dir = Path("reports/nested_v3")
    report_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(report_dir / "outer_trades.csv", index=False)
    folds.to_csv(report_dir / "outer_folds.csv", index=False)
    selections.to_csv(report_dir / "inner_selections.csv", index=False)
    inner_diagnostics.to_csv(report_dir / "inner_threshold_diagnostics.csv", index=False)

    side_summary = analyze_sides(trades)
    side_summary.to_csv(report_dir / "outer_side_summary.csv", index=False)

    if not selections.empty:
        sell_counts = (
            selections["sell_architecture"]
            .value_counts()
            .rename_axis("sell_architecture")
            .reset_index(name="selected_folds")
        )
    else:
        sell_counts = pd.DataFrame(columns=["sell_architecture", "selected_folds"])
    sell_counts.to_csv(report_dir / "sell_architecture_selection_counts.csv", index=False)

    with (report_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nNESTED WALK-FORWARD v3 OUTER-TEST SUMMARY")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\nOUTER BUY / SELL BREAKDOWN")
    print(side_summary.to_string(index=False))

    print("\nINNER SELECTIONS USED BEFORE EACH OUTER TEST")
    selection_cols = [
        "outer_fold",
        "buy_architecture",
        "buy_threshold",
        "sell_architecture",
        "sell_threshold",
        "outer_test_start",
        "outer_test_end",
    ]
    if not selections.empty:
        print(selections[selection_cols].to_string(index=False))

    print("\nSELL ARCHITECTURE SELECTION FREQUENCY")
    print(sell_counts.to_string(index=False))

    print("\nOUTER FOLD PERFORMANCE")
    if not folds.empty:
        fold_cols = [
            "outer_fold",
            "buy_threshold",
            "sell_architecture",
            "sell_threshold",
            "trades",
            "win_rate",
            "profit_factor",
            "net_profit",
            "buy_net_profit",
            "sell_net_profit",
        ]
        print(folds[fold_cols].to_string(index=False))

    print("\nReports saved under reports/nested_v3/")
    print("IMPORTANT: architecture/threshold selection occurs only inside historical inner validation; outer test is never used for selection.")


if __name__ == "__main__":
    main()
