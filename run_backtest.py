from __future__ import annotations

import json
from pathlib import Path

import MetaTrader5 as mt5
import yaml

from backtest.engine import BacktestConfig, run_backtest
from mt5.market_data import MarketData
from mt5.mt5_connector import MT5Connector


def load_config(path: str = "config/config.yaml") -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    cfg = load_config()
    trading = cfg.get("trading", {})
    bt_cfg = cfg.get("backtest", {})

    symbol = str(trading.get("symbol", "XAUUSD.sc"))
    timeframe = str(trading.get("timeframe", "M15"))
    bars = int(bt_cfg.get("bars", 10000))

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

    config = BacktestConfig(
        initial_balance=float(bt_cfg.get("initial_balance", 1000.0)),
        fixed_lot=float(bt_cfg.get("fixed_lot", 0.01)),
        buy_threshold=float(bt_cfg.get("buy_threshold", 0.72)),
        sell_threshold=float(bt_cfg.get("sell_threshold", 0.72)),
        min_probability_edge=float(bt_cfg.get("min_probability_edge", 0.08)),
        horizon=int(bt_cfg.get("horizon", 8)),
        take_profit_pct=float(bt_cfg.get("take_profit_pct", 0.006)),
        stop_loss_pct=float(bt_cfg.get("stop_loss_pct", 0.003)),
        slippage_points=float(bt_cfg.get("slippage_points", 2.0)),
        commission_per_lot_round_turn=float(
            bt_cfg.get("commission_per_lot_round_turn", 0.0)
        ),
        point=point,
        contract_size=contract_size,
    )

    trades, equity, summary = run_backtest(df, config=config)

    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(report_dir / "backtest_trades.csv", index=False)
    equity.to_csv(report_dir / "backtest_equity.csv", index=False)
    with (report_dir / "backtest_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nSaved:")
    print("  reports/backtest_trades.csv")
    print("  reports/backtest_equity.csv")
    print("  reports/backtest_summary.json")


if __name__ == "__main__":
    main()
