from __future__ import annotations

import json
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd
import yaml

from backtest.ablation import FEATURE_GROUPS, run_ablation_suite
from backtest.engine import BacktestConfig
from mt5.market_data import MarketData
from mt5.mt5_connector import MT5Connector


def load_config(path: str = "config/config.yaml") -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _best_rows(sweep: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if sweep.empty:
        return pd.DataFrame()

    for group_name, part in sweep.groupby("feature_group", sort=False):
        valid_pf = part.dropna(subset=["profit_factor"])
        if not valid_pf.empty:
            row = valid_pf.loc[[valid_pf["profit_factor"].idxmax()]].copy()
            row["selection"] = "best_profit_factor"
            rows.append(row)

        row = part.loc[[part["net_profit"].idxmax()]].copy()
        row["selection"] = "best_net_profit"
        rows.append(row)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    cfg = load_config()
    trading = cfg.get("trading", {})
    bt = cfg.get("backtest", {})
    wf = cfg.get("walk_forward", {})

    symbol = str(trading.get("symbol", "XAUUSD.sc"))
    timeframe = str(trading.get("timeframe", "M15"))
    bars = int(wf.get("bars", 30000))
    thresholds = [
        float(x)
        for x in wf.get(
            "raw_threshold_sweep",
            [0.50, 0.55, 0.60, 0.65, 0.70, 0.72, 0.75, 0.80, 0.85, 0.90],
        )
    ]

    print(f"Collecting {bars:,} closed bars for feature ablation: {symbol} {timeframe}")
    print("\nFeature groups:")
    for name, group in FEATURE_GROUPS.items():
        print(f"  - {name}: BUY={len(group['BUY'])} features, SELL={len(group['SELL'])} features")

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

    summaries, sides, sweeps, folds, trade_logs = run_ablation_suite(
        df,
        backtest_cfg,
        thresholds=thresholds,
        train_bars=int(wf.get("train_bars", 12000)),
        calibration_bars=int(wf.get("calibration_bars", 2000)),
        test_bars=int(wf.get("test_bars", 2000)),
        step_bars=int(wf.get("step_bars", 2000)),
        purge_bars=int(wf.get("purge_bars", backtest_cfg.horizon)),
    )

    best = _best_rows(sweeps)

    report_dir = Path("reports/feature_ablation")
    report_dir.mkdir(parents=True, exist_ok=True)
    summaries.to_csv(report_dir / "summary_fixed_threshold.csv", index=False)
    sides.to_csv(report_dir / "side_summary_fixed_threshold.csv", index=False)
    sweeps.to_csv(report_dir / "threshold_sweep.csv", index=False)
    best.to_csv(report_dir / "best_thresholds.csv", index=False)
    folds.to_csv(report_dir / "fold_summary.csv", index=False)

    for name, trades in trade_logs.items():
        trades.to_csv(report_dir / f"trades_{name}.csv", index=False)

    feature_manifest = {
        name: {"BUY": group["BUY"], "SELL": group["SELL"]}
        for name, group in FEATURE_GROUPS.items()
    }
    with (report_dir / "feature_groups.json").open("w", encoding="utf-8") as f:
        json.dump(feature_manifest, f, indent=2, ensure_ascii=False)

    print("\nFIXED-THRESHOLD ABLATION SUMMARY")
    summary_cols = [
        "feature_group", "trades", "win_rate", "profit_factor", "net_profit",
        "expectancy", "max_drawdown_pct", "buy_trades", "sell_trades",
    ]
    print(summaries[summary_cols].to_string(index=False))

    print("\nBUY / SELL BREAKDOWN AT THE SAME FIXED THRESHOLD")
    side_cols = [
        "feature_group", "side", "trades", "win_rate", "profit_factor",
        "net_profit", "expectancy",
    ]
    print(sides[side_cols].to_string(index=False))

    print("\nBEST RAW THRESHOLD PER FEATURE GROUP (development diagnostic only)")
    best_cols = [
        "feature_group", "selection", "threshold", "trades", "win_rate",
        "profit_factor", "net_profit", "expectancy", "buy_net_profit", "sell_net_profit",
    ]
    if not best.empty:
        print(best[best_cols].to_string(index=False))

    print("\nReports saved under reports/feature_ablation/")
    print("IMPORTANT: This is development/model-selection analysis. Do not reuse the consumed 2024 final holdout to choose the winning group.")


if __name__ == "__main__":
    main()
