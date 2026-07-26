from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any, Iterable

import numpy as np
import pandas as pd

from backtest.calibration import calibration_metrics
from backtest.engine import BacktestConfig, _decision, _simulate_trade, _summary


CALIBRATION_METHODS = ("raw", "sigmoid", "isotonic")


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
    method: str = "raw",
) -> pd.DataFrame:
    if bins < 2:
        raise ValueError("bins must be at least 2")
    if method not in CALIBRATION_METHODS:
        raise ValueError(f"Unknown calibration method: {method}")

    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, Any]] = []

    for side, outcome_col in (("BUY", "buy_win"), ("SELL", "sell_win")):
        probability_col = f"{side.lower()}_probability_{method}"
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
                    "method": method,
                    "side": side,
                    "probability_bin": str(interval),
                    "samples": int(len(group)),
                    "mean_predicted_probability": mean_probability,
                    "actual_win_rate": actual_win_rate,
                    "calibration_error": actual_win_rate - mean_probability,
                }
            )

    return pd.DataFrame(rows)


def calibration_method_summary(
    predictions: pd.DataFrame,
    bins: int = 10,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in CALIBRATION_METHODS:
        for side, outcome_col in (("BUY", "buy_win"), ("SELL", "sell_win")):
            probability_col = f"{side.lower()}_probability_{method}"
            metrics = calibration_metrics(
                predictions[outcome_col],
                predictions[probability_col],
                bins=bins,
            )
            rows.append({"method": method, "side": side, **metrics})
    return pd.DataFrame(rows)


def add_cost_adjusted_breakeven(
    predictions: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    """Add a per-signal break-even probability including trading costs.

    The nominal two-outcome expectation is:
        p * (TP_profit - cost) + (1-p) * (-SL_loss - cost) = 0

    therefore:
        p_break_even = (SL_loss + cost) / (TP_profit + SL_loss)

    The cost estimate uses the next bar (the simulated entry bar), historical
    spread, configured round-trip slippage and commission. This is a decision
    aid; TIME exits and intrabar ambiguity can make realized break-even differ.
    """
    out = predictions.copy()
    out["breakeven_probability"] = np.nan
    out["estimated_round_trip_cost"] = np.nan
    out["nominal_tp_profit"] = np.nan
    out["nominal_sl_loss"] = np.nan

    for _, fold_index in out.groupby("fold", sort=True).groups.items():
        positions = list(fold_index)
        for pos in positions[:-1]:
            current = out.loc[pos]
            next_pos = pos + 1
            if next_pos not in out.index or out.loc[next_pos, "fold"] != current["fold"]:
                continue

            entry = out.loc[next_pos]
            entry_price = float(entry["open"])
            spread_points = float(entry.get("spread", 0.0))

            tp_profit = (
                entry_price
                * config.take_profit_pct
                * config.contract_size
                * config.fixed_lot
            )
            sl_loss = (
                entry_price
                * config.stop_loss_pct
                * config.contract_size
                * config.fixed_lot
            )
            spread_cost = (
                spread_points
                * config.point
                * config.contract_size
                * config.fixed_lot
            )
            slippage_cost = (
                2.0
                * config.slippage_points
                * config.point
                * config.contract_size
                * config.fixed_lot
            )
            commission_cost = (
                config.commission_per_lot_round_turn * config.fixed_lot
            )
            cost = spread_cost + slippage_cost + commission_cost
            denominator = tp_profit + sl_loss
            if denominator <= 0:
                continue

            p_be = (sl_loss + cost) / denominator
            out.loc[pos, "breakeven_probability"] = float(np.clip(p_be, 0.0, 1.0))
            out.loc[pos, "estimated_round_trip_cost"] = cost
            out.loc[pos, "nominal_tp_profit"] = tp_profit
            out.loc[pos, "nominal_sl_loss"] = sl_loss

    return out


def _run_predictions_at_threshold(
    predictions: pd.DataFrame,
    cfg: BacktestConfig,
    threshold: float,
    method: str = "raw",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if method not in CALIBRATION_METHODS:
        raise ValueError(f"Unknown calibration method: {method}")

    threshold_cfg = replace(
        cfg,
        buy_threshold=float(threshold),
        sell_threshold=float(threshold),
    )
    buy_col = f"buy_probability_{method}"
    sell_col = f"sell_probability_{method}"

    trades: list[dict[str, Any]] = []
    balance = cfg.initial_balance

    for fold_id, fold in predictions.groupby("fold", sort=True):
        fold = fold.sort_values("local_index").reset_index(drop=True)
        if fold.empty:
            continue

        probability_by_index: dict[int, tuple[float, float]] = {}
        for row in fold.itertuples():
            buy = getattr(row, buy_col)
            sell = getattr(row, sell_col)
            if not np.isfinite(buy) or not np.isfinite(sell):
                continue
            probability_by_index[int(row.local_index)] = (float(buy), float(sell))

        if not probability_by_index:
            continue
        i = min(probability_by_index)
        last_i = max(probability_by_index)

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
            row["calibration_method"] = method
            trades.append(row)
            balance = trade.balance_after
            i = exit_index + 1

    trades_df = pd.DataFrame(trades)
    return trades_df, _summary(trades_df, cfg.initial_balance)


def _run_predictions_at_edge_margin(
    predictions: pd.DataFrame,
    cfg: BacktestConfig,
    edge_margin: float,
    method: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replay calibrated OOS predictions above dynamic cost-adjusted break-even."""
    if method not in ("sigmoid", "isotonic"):
        raise ValueError("Cost-adjusted edge sweep is intended for calibrated probabilities")
    if edge_margin < 0:
        raise ValueError("edge_margin must be non-negative")

    buy_col = f"buy_probability_{method}"
    sell_col = f"sell_probability_{method}"
    trades: list[dict[str, Any]] = []
    balance = cfg.initial_balance

    for fold_id, fold in predictions.groupby("fold", sort=True):
        fold = fold.sort_values("local_index").reset_index(drop=True)
        if fold.empty:
            continue

        i = 0
        last_i = len(fold) - cfg.horizon - 1
        while i <= last_i:
            row = fold.iloc[i]
            buy_p = float(row.get(buy_col, np.nan))
            sell_p = float(row.get(sell_col, np.nan))
            p_be = float(row.get("breakeven_probability", np.nan))
            if not (np.isfinite(buy_p) and np.isfinite(sell_p) and np.isfinite(p_be)):
                i += 1
                continue

            required = p_be + edge_margin
            buy_edge = buy_p - required
            sell_edge = sell_p - required
            buy_ok = buy_edge >= 0.0
            sell_ok = sell_edge >= 0.0

            if not buy_ok and not sell_ok:
                i += 1
                continue
            if buy_ok and sell_ok:
                if abs(buy_edge - sell_edge) < 1e-12:
                    i += 1
                    continue
                side = "BUY" if buy_edge > sell_edge else "SELL"
            else:
                side = "BUY" if buy_ok else "SELL"

            trade, exit_index = _simulate_trade(
                df=fold,
                signal_index=i,
                side=side,
                buy_probability=buy_p,
                sell_probability=sell_p,
                cfg=cfg,
                balance=balance,
            )
            trade_row = asdict(trade)
            trade_row["fold"] = int(fold_id)
            trade_row["calibration_method"] = method
            trade_row["edge_margin"] = float(edge_margin)
            trade_row["breakeven_probability"] = p_be
            trade_row["required_probability"] = required
            trade_row["selected_probability_edge"] = (
                buy_p - p_be if side == "BUY" else sell_p - p_be
            )
            trades.append(trade_row)
            balance = trade.balance_after
            i = exit_index + 1

    trades_df = pd.DataFrame(trades)
    return trades_df, _summary(trades_df, cfg.initial_balance)


def threshold_sweep(
    predictions: pd.DataFrame,
    config: BacktestConfig,
    thresholds: Iterable[float],
    method: str = "raw",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        value = float(threshold)
        if not 0.0 < value < 1.0:
            raise ValueError(f"Invalid threshold: {value}")

        trades, summary = _run_predictions_at_threshold(
            predictions, config, value, method=method
        )
        side_df = analyze_sides(trades)
        buy = side_df.loc[side_df["side"] == "BUY"].iloc[0].to_dict()
        sell = side_df.loc[side_df["side"] == "SELL"].iloc[0].to_dict()

        rows.append(
            {
                "method": method,
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


def calibrated_edge_sweep(
    predictions: pd.DataFrame,
    config: BacktestConfig,
    edge_margins: Iterable[float],
) -> pd.DataFrame:
    """Compare Sigmoid/Isotonic using probability edge over dynamic break-even."""
    rows: list[dict[str, Any]] = []
    for method in ("sigmoid", "isotonic"):
        for margin in edge_margins:
            value = float(margin)
            trades, summary = _run_predictions_at_edge_margin(
                predictions, config, value, method
            )
            side_df = analyze_sides(trades)
            buy = side_df.loc[side_df["side"] == "BUY"].iloc[0].to_dict()
            sell = side_df.loc[side_df["side"] == "SELL"].iloc[0].to_dict()
            rows.append(
                {
                    "method": method,
                    "edge_margin": value,
                    "trades": summary.get("trades", 0),
                    "win_rate": summary.get("win_rate", 0.0),
                    "profit_factor": summary.get("profit_factor"),
                    "net_profit": summary.get("net_profit", 0.0),
                    "expectancy": summary.get("expectancy", 0.0),
                    "max_drawdown_pct": summary.get("max_drawdown_pct", 0.0),
                    "total_costs": summary.get("total_costs", 0.0),
                    "buy_trades": int(buy["trades"]),
                    "buy_net_profit": float(buy["net_profit"]),
                    "sell_trades": int(sell["trades"]),
                    "sell_net_profit": float(sell["net_profit"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["method", "edge_margin"]).reset_index(drop=True)


def compare_calibration_methods(
    predictions: pd.DataFrame,
    config: BacktestConfig,
    raw_thresholds: Iterable[float],
    calibrated_thresholds: Iterable[float],
    bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare methods using probability ranges appropriate to each scale."""
    sweeps: list[pd.DataFrame] = []
    sweeps.append(threshold_sweep(predictions, config, raw_thresholds, method="raw"))
    for method in ("sigmoid", "isotonic"):
        sweeps.append(
            threshold_sweep(predictions, config, calibrated_thresholds, method=method)
        )

    non_empty_sweeps = [df for df in sweeps if not df.empty]
    sweep_df = (
        pd.concat(non_empty_sweeps, ignore_index=True)
        if non_empty_sweeps
        else pd.DataFrame()
    )

    calibration_parts = [
        calibration_table(predictions, bins=bins, method=m)
        for m in CALIBRATION_METHODS
    ]
    non_empty_calibration = [df for df in calibration_parts if not df.empty]
    calibration_df = (
        pd.concat(non_empty_calibration, ignore_index=True)
        if non_empty_calibration
        else pd.DataFrame()
    )
    method_metrics = calibration_method_summary(predictions, bins=bins)
    return sweep_df, calibration_df, method_metrics
