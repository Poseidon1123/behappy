from __future__ import annotations

import json
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd
import yaml

from backtest.analysis import (
    analyze_sides,
    compare_calibration_methods,
)
from backtest.engine import BacktestConfig
from backtest.walk_forward import run_walk_forward_backtest
from mt5.market_data import MarketData
from mt5.mt5_connector import MT5Connector


def load_config(path: str = "config/config.yaml") -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _records(df: pd.DataFrame):
    if df.empty:
        return []
    return json.loads(df.to_json(orient="records"))


def main() -> None:
    cfg = load_config()
    trading = cfg.get("trading", {})
    bt = cfg.get("backtest", {})
    wf = cfg.get("walk_forward", {})

    symbol = str(trading.get("symbol", "XAUUSD.sc"))
    timeframe = str(trading.get("timeframe", "M15"))
    bars = int(wf.get("bars", 30000))
    baseline_method = str(wf.get("baseline_calibration_method", "raw")).lower()

    print(f"Collecting {bars:,} closed bars for calibrated walk-forward: {symbol} {timeframe}")

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

    trades, equity, folds, predictions, summary = run_walk_forward_backtest(
        raw_df=df,
        backtest_config=backtest_cfg,
        train_bars=int(wf.get("train_bars", 12000)),
        calibration_bars=int(wf.get("calibration_bars", 2000)),
        test_bars=int(wf.get("test_bars", 2000)),
        step_bars=int(wf.get("step_bars", 2000)),
        purge_bars=int(wf.get("purge_bars", backtest_cfg.horizon)),
        calibration_method=baseline_method,
    )

    thresholds = wf.get(
        "threshold_sweep",
        [0.50, 0.55, 0.60, 0.65, 0.70, 0.72, 0.75, 0.80, 0.85, 0.90],
    )
    bins = int(wf.get("calibration_bins", 10))

    sweep, calibration_bins, calibration_metrics = compare_calibration_methods(
        predictions,
        backtest_cfg,
        thresholds,
        bins=bins,
    )
    side_summary = analyze_sides(trades)

    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(report_dir / "walk_forward_trades.csv", index=False)
    equity.to_csv(report_dir / "walk_forward_equity.csv", index=False)
    folds.to_csv(report_dir / "walk_forward_folds.csv", index=False)
    predictions.to_csv(report_dir / "walk_forward_predictions.csv", index=False)
    side_summary.to_csv(report_dir / "walk_forward_side_summary.csv", index=False)
    sweep.to_csv(report_dir / "walk_forward_threshold_sweep.csv", index=False)
    calibration_bins.to_csv(report_dir / "walk_forward_calibration.csv", index=False)
    calibration_metrics.to_csv(report_dir / "walk_forward_calibration_metrics.csv", index=False)

    best_rows = []
    for method, part in sweep.groupby("method"):
        valid_pf = part.dropna(subset=["profit_factor"])
        if not valid_pf.empty:
            row = valid_pf.loc[[valid_pf["profit_factor"].idxmax()]].copy()
            row["selection"] = "best_profit_factor"
            best_rows.append(row)
        if not part.empty:
            row = part.loc[[part["net_profit"].idxmax()]].copy()
            row["selection"] = "best_net_profit"
            best_rows.append(row)
    best_df = pd.concat(best_rows, ignore_index=True) if best_rows else pd.DataFrame()
    best_df.to_csv(report_dir / "walk_forward_calibration_best.csv", index=False)

    analysis_summary = {
        "baseline": summary,
        "baseline_side_summary": _records(side_summary),
        "calibration_metrics": _records(calibration_metrics),
        "best_by_method": _records(best_df),
        "warning": (
            "Thresholds compared on these OOS folds are still model-selection information. "
            "Confirm any chosen method/threshold on a later untouched period before deployment."
        ),
    }

    with (report_dir / "walk_forward_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (report_dir / "walk_forward_analysis.json").open("w", encoding="utf-8") as f:
        json.dump(analysis_summary, f, indent=2, ensure_ascii=False)

    print("\nBaseline walk-forward summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\nCalibration quality (lower Brier/ECE is better):")
    print(calibration_metrics.to_string(index=False))

    print("\nBest threshold per method:")
    display_cols = [
        "method", "selection", "threshold", "trades", "win_rate",
        "profit_factor", "net_profit", "expectancy", "max_drawdown_pct",
        "buy_net_profit", "sell_net_profit",
    ]
    if not best_df.empty:
        print(best_df[display_cols].to_string(index=False))

    print("\nFull method/threshold comparison saved to reports/walk_forward_threshold_sweep.csv")
    print("Reports saved under reports/walk_forward_*.csv/json")


if __name__ == "__main__":
    main()
