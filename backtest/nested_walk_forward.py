from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pandas as pd

from ai.labeling import create_tp_sl_labels
from backtest.ablation import FEATURE_GROUPS
from backtest.analysis import analyze_sides
from backtest.engine import BacktestConfig, _simulate_trade, _summary
from backtest.walk_forward import _fit_binary_model
from data.feature_engineering import build_features


BUY_GROUP = "A_m15_baseline"
SELL_GROUPS = ("A_m15_baseline", "B_m15_raw_h1", "E_m15_regime_only")


def _dedupe(columns: list[str]) -> list[str]:
    return list(dict.fromkeys(columns))


def _side_only_replay(
    df: pd.DataFrame,
    probabilities: dict[int, float],
    *,
    side: str,
    threshold: float,
    cfg: BacktestConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate one side independently on an inner-validation block."""
    balance = cfg.initial_balance
    rows: list[dict[str, Any]] = []
    if not probabilities:
        empty = pd.DataFrame()
        return empty, _summary(empty, cfg.initial_balance)

    i = min(probabilities)
    last_i = max(probabilities)
    while i <= last_i:
        p = probabilities.get(i)
        if p is None or not np.isfinite(p) or p < threshold:
            i += 1
            continue

        buy_p = float(p) if side == "BUY" else 0.0
        sell_p = float(p) if side == "SELL" else 0.0
        trade, exit_i = _simulate_trade(
            df=df,
            signal_index=i,
            side=side,
            buy_probability=buy_p,
            sell_probability=sell_p,
            cfg=cfg,
            balance=balance,
        )
        rows.append(asdict(trade))
        balance = trade.balance_after
        i = exit_i + 1

    trades = pd.DataFrame(rows)
    return trades, _summary(trades, cfg.initial_balance)


def _selection_score(summary: dict[str, Any], min_trades: int) -> tuple[float, float, int]:
    """Rank inner candidates conservatively: PF first, then expectancy/trade count.

    Candidates with too few trades or non-positive expectancy are strongly penalized.
    """
    trades = int(summary.get("trades", 0) or 0)
    expectancy = float(summary.get("expectancy", 0.0) or 0.0)
    pf_raw = summary.get("profit_factor")
    pf = float(pf_raw) if pf_raw is not None and np.isfinite(pf_raw) else 0.0
    if trades < min_trades or expectancy <= 0.0:
        return (-1.0, expectancy, trades)
    return (pf, expectancy, trades)


def _choose_threshold_for_side(
    validation_df: pd.DataFrame,
    probabilities: np.ndarray,
    valid_indices: np.ndarray,
    *,
    side: str,
    thresholds: list[float],
    cfg: BacktestConfig,
    min_trades: int,
) -> tuple[float, pd.DataFrame]:
    prob_map = {
        int(idx): float(p)
        for idx, p in zip(valid_indices, probabilities)
        if np.isfinite(p)
    }
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        trades, summary = _side_only_replay(
            validation_df,
            prob_map,
            side=side,
            threshold=float(threshold),
            cfg=cfg,
        )
        rows.append(
            {
                "side": side,
                "threshold": float(threshold),
                "trades": int(summary.get("trades", 0)),
                "win_rate": float(summary.get("win_rate", 0.0) or 0.0),
                "profit_factor": summary.get("profit_factor"),
                "net_profit": float(summary.get("net_profit", 0.0) or 0.0),
                "expectancy": float(summary.get("expectancy", 0.0) or 0.0),
                "score": _selection_score(summary, min_trades),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError(f"No inner threshold candidates for {side}")
    best_index = max(range(len(rows)), key=lambda k: rows[k]["score"])
    return float(rows[best_index]["threshold"]), table.drop(columns=["score"])


def _outer_decision(
    buy_p: float,
    sell_p: float,
    *,
    buy_threshold: float,
    sell_threshold: float,
    min_probability_edge: float,
) -> str:
    buy_ok = buy_p >= buy_threshold
    sell_ok = sell_p >= sell_threshold
    if buy_ok and not sell_ok:
        return "BUY"
    if sell_ok and not buy_ok:
        return "SELL"
    if not buy_ok and not sell_ok:
        return "HOLD"

    # Compare excess confidence above each side's independently selected threshold.
    buy_excess = buy_p - buy_threshold
    sell_excess = sell_p - sell_threshold
    if buy_excess - sell_excess >= min_probability_edge:
        return "BUY"
    if sell_excess - buy_excess >= min_probability_edge:
        return "SELL"
    return "HOLD"


def _combined_outer_replay(
    test_df: pd.DataFrame,
    buy_prob: dict[int, float],
    sell_prob: dict[int, float],
    *,
    buy_threshold: float,
    sell_threshold: float,
    cfg: BacktestConfig,
    balance: float,
    outer_fold: int,
    sell_group: str,
) -> tuple[list[dict[str, Any]], float]:
    common = sorted(set(buy_prob).intersection(sell_prob))
    if not common:
        return [], balance

    rows: list[dict[str, Any]] = []
    i = common[0]
    last_i = common[-1]
    while i <= last_i:
        if i not in buy_prob or i not in sell_prob:
            i += 1
            continue
        bp = buy_prob[i]
        sp = sell_prob[i]
        side = _outer_decision(
            bp,
            sp,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            min_probability_edge=cfg.min_probability_edge,
        )
        if side == "HOLD":
            i += 1
            continue
        trade, exit_i = _simulate_trade(
            df=test_df,
            signal_index=i,
            side=side,
            buy_probability=bp,
            sell_probability=sp,
            cfg=cfg,
            balance=balance,
        )
        row = asdict(trade)
        row.update(
            {
                "outer_fold": outer_fold,
                "selected_buy_group": BUY_GROUP,
                "selected_sell_group": sell_group,
                "selected_buy_threshold": buy_threshold,
                "selected_sell_threshold": sell_threshold,
            }
        )
        rows.append(row)
        balance = trade.balance_after
        i = exit_i + 1
    return rows, balance


def run_nested_walk_forward_v3(
    raw_df: pd.DataFrame,
    cfg: BacktestConfig,
    *,
    thresholds: list[float],
    outer_train_bars: int = 12000,
    inner_validation_bars: int = 2000,
    outer_test_bars: int = 2000,
    step_bars: int | None = None,
    purge_bars: int | None = None,
    min_inner_trades: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Nested walk-forward v3.

    For every outer fold:
      1. BUY architecture is fixed to A_m15_baseline.
      2. Inner validation selects BUY threshold.
      3. Inner validation selects SELL architecture from A/B/E and SELL threshold.
      4. Selected models are refit on all outer-training history.
      5. The locked choices are evaluated once on the future outer-test block.

    Outer-test observations never participate in architecture/threshold selection.
    """
    thresholds = sorted({float(x) for x in thresholds})
    if not thresholds or any(not 0.0 < x < 1.0 for x in thresholds):
        raise ValueError("thresholds must contain values strictly between 0 and 1")
    if inner_validation_bars < 500:
        raise ValueError("inner_validation_bars must be at least 500")
    if outer_train_bars <= inner_validation_bars + 1000:
        raise ValueError("outer_train_bars is too small for inner fit/validation")

    step_bars = int(step_bars or outer_test_bars)
    purge_bars = int(purge_bars or cfg.horizon)
    if purge_bars < cfg.horizon:
        raise ValueError("purge_bars must be at least the label horizon")

    featured = build_features(raw_df).reset_index(drop=True)
    labeled = create_tp_sl_labels(
        featured,
        horizon=cfg.horizon,
        take_profit_pct=cfg.take_profit_pct,
        stop_loss_pct=cfg.stop_loss_pct,
    )

    buy_features = FEATURE_GROUPS[BUY_GROUP]["BUY"]
    sell_feature_map = {name: FEATURE_GROUPS[name]["SELL"] for name in SELL_GROUPS}
    all_required = _dedupe(
        buy_features + [c for columns in sell_feature_map.values() for c in columns]
    )

    n = len(featured)
    first_test_start = outer_train_bars + purge_bars
    if first_test_start + outer_test_bars > n:
        raise ValueError("Not enough bars for one nested outer fold")

    all_trades: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    inner_rows: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    balance = cfg.initial_balance
    outer_fold = 0
    test_start = first_test_start

    while test_start + outer_test_bars <= n:
        outer_fold += 1
        outer_train_end = test_start - purge_bars
        outer_train_start = max(0, outer_train_end - outer_train_bars)
        test_end = test_start + outer_test_bars

        inner_val_end = outer_train_end
        inner_val_start = inner_val_end - inner_validation_bars
        inner_fit_end = inner_val_start - purge_bars
        inner_fit_start = outer_train_start
        if inner_fit_end - inner_fit_start < 1500:
            test_start += step_bars
            continue

        inner_fit = labeled.iloc[inner_fit_start:inner_fit_end].copy().dropna(
            subset=all_required + ["buy_win", "sell_win"]
        )
        inner_val = featured.iloc[inner_val_start:inner_val_end].copy().reset_index(drop=True)
        outer_refit = labeled.iloc[outer_train_start:outer_train_end].copy().dropna(
            subset=all_required + ["buy_win", "sell_win"]
        )
        outer_test = featured.iloc[test_start:test_end].copy().reset_index(drop=True)

        if len(inner_fit) < 1500 or len(outer_refit) < 2000:
            test_start += step_bars
            continue

        y_buy_inner = inner_fit["buy_win"].astype(int)
        y_sell_inner = inner_fit["sell_win"].astype(int)
        y_buy_refit = outer_refit["buy_win"].astype(int)
        y_sell_refit = outer_refit["sell_win"].astype(int)
        if any(y.nunique() < 2 for y in (y_buy_inner, y_sell_inner, y_buy_refit, y_sell_refit)):
            test_start += step_bars
            continue

        # --- Inner BUY threshold selection: architecture fixed to M15 baseline.
        buy_inner_model = _fit_binary_model(inner_fit[buy_features], y_buy_inner)
        buy_valid_mask = inner_val[buy_features].notna().all(axis=1)
        buy_valid_idx = np.flatnonzero(buy_valid_mask.to_numpy())
        buy_valid_idx = buy_valid_idx[buy_valid_idx < len(inner_val) - cfg.horizon - 1]
        if len(buy_valid_idx) == 0:
            test_start += step_bars
            continue
        buy_inner_prob = buy_inner_model.predict_proba(inner_val.iloc[buy_valid_idx][buy_features])[:, 1]
        selected_buy_threshold, buy_table = _choose_threshold_for_side(
            inner_val,
            buy_inner_prob,
            buy_valid_idx,
            side="BUY",
            thresholds=thresholds,
            cfg=cfg,
            min_trades=min_inner_trades,
        )
        buy_table.insert(0, "outer_fold", outer_fold)
        buy_table.insert(1, "architecture", BUY_GROUP)
        inner_rows.append(buy_table)

        # --- Inner SELL architecture + threshold selection.
        sell_candidates: list[dict[str, Any]] = []
        sell_tables: list[pd.DataFrame] = []
        for sell_group, sell_features in sell_feature_map.items():
            model = _fit_binary_model(inner_fit[sell_features], y_sell_inner)
            mask = inner_val[sell_features].notna().all(axis=1)
            idx = np.flatnonzero(mask.to_numpy())
            idx = idx[idx < len(inner_val) - cfg.horizon - 1]
            if len(idx) == 0:
                continue
            prob = model.predict_proba(inner_val.iloc[idx][sell_features])[:, 1]
            selected_threshold, table = _choose_threshold_for_side(
                inner_val,
                prob,
                idx,
                side="SELL",
                thresholds=thresholds,
                cfg=cfg,
                min_trades=min_inner_trades,
            )
            table.insert(0, "outer_fold", outer_fold)
            table.insert(1, "architecture", sell_group)
            sell_tables.append(table)
            selected_row = table.loc[table["threshold"] == selected_threshold].iloc[0].to_dict()
            candidate_summary = {
                "trades": selected_row["trades"],
                "expectancy": selected_row["expectancy"],
                "profit_factor": selected_row["profit_factor"],
            }
            sell_candidates.append(
                {
                    "architecture": sell_group,
                    "threshold": selected_threshold,
                    "score": _selection_score(candidate_summary, min_inner_trades),
                    **selected_row,
                }
            )

        if not sell_candidates:
            test_start += step_bars
            continue
        inner_rows.extend(sell_tables)
        chosen_sell = max(sell_candidates, key=lambda row: row["score"])
        selected_sell_group = str(chosen_sell["architecture"])
        selected_sell_threshold = float(chosen_sell["threshold"])
        selected_sell_features = sell_feature_map[selected_sell_group]

        selection_rows.append(
            {
                "outer_fold": outer_fold,
                "buy_architecture": BUY_GROUP,
                "buy_threshold": selected_buy_threshold,
                "sell_architecture": selected_sell_group,
                "sell_threshold": selected_sell_threshold,
                "inner_fit_start": str(featured.iloc[inner_fit_start]["time"]),
                "inner_fit_end": str(featured.iloc[inner_fit_end - 1]["time"]),
                "inner_validation_start": str(featured.iloc[inner_val_start]["time"]),
                "inner_validation_end": str(featured.iloc[inner_val_end - 1]["time"]),
                "outer_test_start": str(featured.iloc[test_start]["time"]),
                "outer_test_end": str(featured.iloc[test_end - 1]["time"]),
            }
        )

        # --- Refit locked architectures on all outer-training history.
        buy_model = _fit_binary_model(outer_refit[buy_features], y_buy_refit)
        sell_model = _fit_binary_model(outer_refit[selected_sell_features], y_sell_refit)
        valid_mask = (
            outer_test[buy_features].notna().all(axis=1)
            & outer_test[selected_sell_features].notna().all(axis=1)
        )
        valid_idx = np.flatnonzero(valid_mask.to_numpy())
        valid_idx = valid_idx[valid_idx < len(outer_test) - cfg.horizon - 1]

        buy_map: dict[int, float] = {}
        sell_map: dict[int, float] = {}
        if len(valid_idx):
            buy_prob = buy_model.predict_proba(outer_test.iloc[valid_idx][buy_features])[:, 1]
            sell_prob = sell_model.predict_proba(outer_test.iloc[valid_idx][selected_sell_features])[:, 1]
            buy_map = {int(i): float(p) for i, p in zip(valid_idx, buy_prob)}
            sell_map = {int(i): float(p) for i, p in zip(valid_idx, sell_prob)}

        fold_initial_balance = balance
        fold_trades, balance = _combined_outer_replay(
            outer_test,
            buy_map,
            sell_map,
            buy_threshold=selected_buy_threshold,
            sell_threshold=selected_sell_threshold,
            cfg=cfg,
            balance=balance,
            outer_fold=outer_fold,
            sell_group=selected_sell_group,
        )
        all_trades.extend(fold_trades)
        fold_df = pd.DataFrame(fold_trades)
        fold_summary = _summary(fold_df, fold_initial_balance)
        sides = analyze_sides(fold_df)
        buy_side = sides.loc[sides["side"] == "BUY"].iloc[0].to_dict()
        sell_side = sides.loc[sides["side"] == "SELL"].iloc[0].to_dict()
        fold_rows.append(
            {
                "outer_fold": outer_fold,
                "buy_threshold": selected_buy_threshold,
                "sell_architecture": selected_sell_group,
                "sell_threshold": selected_sell_threshold,
                "trades": int(fold_summary.get("trades", 0)),
                "win_rate": float(fold_summary.get("win_rate", 0.0) or 0.0),
                "profit_factor": fold_summary.get("profit_factor"),
                "net_profit": float(fold_summary.get("net_profit", 0.0) or 0.0),
                "expectancy": float(fold_summary.get("expectancy", 0.0) or 0.0),
                "buy_trades": int(buy_side["trades"]),
                "buy_net_profit": float(buy_side["net_profit"]),
                "sell_trades": int(sell_side["trades"]),
                "sell_net_profit": float(sell_side["net_profit"]),
                "ending_balance": float(balance),
            }
        )
        test_start += step_bars

    trades = pd.DataFrame(all_trades)
    selections = pd.DataFrame(selection_rows)
    inner_diagnostics = pd.concat(inner_rows, ignore_index=True) if inner_rows else pd.DataFrame()
    folds = pd.DataFrame(fold_rows)
    summary = _summary(trades, cfg.initial_balance)
    summary.update(
        {
            "architecture": "nested_v3_buy_m15_sell_A_B_E",
            "outer_folds": int(len(folds)),
            "buy_architecture_fixed": BUY_GROUP,
            "sell_architecture_candidates": list(SELL_GROUPS),
            "threshold_candidates": thresholds,
            "outer_train_bars": int(outer_train_bars),
            "inner_validation_bars": int(inner_validation_bars),
            "outer_test_bars": int(outer_test_bars),
            "step_bars": int(step_bars),
            "purge_bars": int(purge_bars),
            "min_inner_trades": int(min_inner_trades),
            "outer_test_used_for_selection": False,
        }
    )
    return trades, folds, selections, inner_diagnostics, summary
