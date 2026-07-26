from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any, Iterable

import numpy as np
import pandas as pd

from backtest.engine import BacktestConfig, _decision, _simulate_trade, _summary


def _empty_side_summary(side: str) -> dict[str, Any]:
    return {
        "side": side,
        "trades": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0.0,
        "net_profit": 0.0,
        "profit_factor": None,
        "expectancy": 0.0,
        "total_costs": 0.0,
    }


def analyze_sides(trades: pd.DataFrame) -> pd.DataFrame:
    """Summarize BUY and SELL performance independently."""
    rows: list[dict[str, Any]] = []
    for side in ("BUY", "SELL"):
        part = trades.loc[trades.get("side", pd.Series(dtype=str)) == side].copy()
        if part.empty:
            rows.append(_empty_side_summary(side))
            continue

        pnl = part["net_pnl"].astype(float)
        gross_profit = float(pnl[pnl > 0].sum())
        gross_loss = float(-pnl[pnl < 0].sum())
        costs = float(
            part[["spread_cost", "slippage_cost", "commission_cost"]]
            .sum(axis=1)
            .sum()
        )
        rows.append(
            {
                "side": side,
                "trades": int(len(part)),
                "wins": int((pnl > 0).sum()),
                "losses": int((pnl <= 0).sum()),
                "win_rate": float((pnl > 0).mean()),
                "net_profit": float(pnl.sum()),
                "profit_factor": gross_profit / gross_loss if gross_loss > 0 else None,
                "expectancy": float(pnl.mean()),
                "total_costs": costs,
            }
        )
    return pd.DataFrame(rows)


def calibration_table(
    predictions: pd.DataFrame,
    bins: int = 10,
) -> pd.DataFrame:
    """Compare model probability with realized TP-before-SL frequency OOS."""
    if bins < 2:
        raise ValueError("bins must be at least 2")

    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, Any]] = []

    for side, probability_col, outcome_col in (
        ("BUY", "buy_probability", "buy_win"),
        ("SELL", "sell_probability", "sell_win"),
    ):
        part = predictions[[probability_col, outcome_col]].dropna().copy()
        if part.empty:
            continue
        part["bin"] = pd.cut(
            part[probability_col],
            bins=edges,
            include_lowest=True,
            right=True,
            duplicates="drop",
        )
        for interval, group in part.groupby("bin", observed=True):
            if group.empty:
                continue
            mean_probability = float(group[probability_col].mean())
            actual_win_rate = float(group[outcome_col].astype(float).mean())
            rows.append(
                {
                    "side": side,
                    "probability_bin": str(interval),
                    "samples": int(len(group)),
                    "mean_predicted_probability": mean_probability,
                    "actual_win_rate": actual_win_rate,
                    "calibration_error": actual_win_rate - mean_probability,
                }
            )

    return pd.DataFrame(rows)


def _run_predictions_at_threshold(
    predictions: pd.DataFrame,
    cfg: BacktestConfig,
    threshold: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replay OOS predictions using one symmetric BUY/SELL threshold."""
    threshold_cfg = replace(
        cfg,
        buy_threshold=float(threshold),
        sell_threshold=float(threshold),
    )

    trades: list[dict[str, Any]] = []
    balance = cfg.initial_balance

    for fold_id, fold in predictions.groupby("fold", sort=True):
        fold = fold.sort_values("local_index").reset_index(drop=True)
        if fold.empty:
            continue

        probability_by_index = {
            int(row.local_index): (float(row.buy_probability), float(row.sell_probability))
            for row in fold.itertuples()
        }
        first_i = int(fold["local_index"].min())
        last_i = int(fold["local_index"].max())
        i = first_i

        while i <= last_i:
            pair = probability_by_index.get(i)
            if pair is None:
                i += 1
                continue

            buy_p, sell_p = pair
            side = _decision(buy_p, sell_p, threshold_cfg)
            if side == "HOLD":
                i += 1
                continue

            trade, exit_index = _simulate_trade(
                df=fold,
                signal_index=i,
                side=side,
                buy_probability=buy_p,
                sell_probability=sell_p,
                cfg=threshold_cfg,
                balance=balance,
            )
            row = asdict(trade)
            row["fold"] = int(fold_id)
            row["threshold"] = float(threshold)
            trades.append(row)
            balance = trade.balance_after
            i = exit_index + 1

    trades_df = pd.DataFrame(trades)
    return trades_df, _summary(trades_df, cfg.initial_balance)


def threshold_sweep(
    predictions: pd.DataFrame,
    config: BacktestConfig,
    thresholds: Iterable[float],
) -> pd.DataFrame:
    """Evaluate multiple probability thresholds on identical OOS predictions."""
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        value = float(threshold)
        if not 0.0 < value < 1.0:
            raise ValueError(f"Invalid threshold: {value}")

        trades, summary = _run_predictions_at_threshold(predictions, config, value)
        side_df = analyze_sides(trades)
        buy = side_df.loc[side_df["side"] == "BUY"].iloc[0].to_dict()
        sell = side_df.loc[side_df["side"] == "SELL"].iloc[0].to_dict()

        rows.append(
            {
                "threshold": value,
                "trades": summary.get("trades", 0),
                "win_rate": summary.get("win_rate", 0.0),
                "profit_factor": summary.get("profit_factor"),
                "net_profit": summary.get("net_profit", 0.0),
                "expectancy": summary.get("expectancy", 0.0),
                "max_drawdown_pct": summary.get("max_drawdown_pct", 0.0),
                "total_costs": summary.get("total_costs", 0.0),
                "buy_trades": int(buy["trades"]),
                "buy_win_rate": float(buy["win_rate"]),
                "buy_profit_factor": buy["profit_factor"],
                "buy_net_profit": float(buy["net_profit"]),
                "sell_trades": int(sell["trades"]),
                "sell_win_rate": float(sell["win_rate"]),
                "sell_profit_factor": sell["profit_factor"],
                "sell_net_profit": float(sell["net_profit"]),
            }
        )

    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
