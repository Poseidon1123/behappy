from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from backtest.engine import BacktestConfig, _summary
from backtest.exits import EXIT_POLICIES, ExitPolicy, simulate_trade_with_exit_policy
from backtest.nested_robust import _outer_decision, _robust_score
from backtest.walk_forward import _fit_binary_model


def _replay_policy(
    frame: pd.DataFrame,
    buy_prob: dict[int, float],
    sell_prob: dict[int, float],
    *,
    buy_threshold: float,
    sell_threshold: float,
    cfg: BacktestConfig,
    policy: ExitPolicy,
) -> dict[str, Any]:
    common = sorted(set(buy_prob).intersection(sell_prob))
    if not common:
        return _summary(pd.DataFrame(), cfg.initial_balance)
    rows: list[dict[str, Any]] = []
    balance = cfg.initial_balance
    i, last_i = common[0], common[-1]
    while i <= last_i:
        if i not in buy_prob or i not in sell_prob:
            i += 1
            continue
        bp, sp = buy_prob[i], sell_prob[i]
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
        trade, exit_i = simulate_trade_with_exit_policy(
            frame,
            i,
            side,
            bp,
            sp,
            cfg,
            balance,
            policy,
        )
        rows.append(asdict(trade))
        balance = trade.balance_after
        i = exit_i + 1
    return _summary(pd.DataFrame(rows), cfg.initial_balance)


def select_exit_policy(
    labeled: pd.DataFrame,
    featured: pd.DataFrame,
    *,
    splits: list[tuple[int, int, int, int]],
    buy_features: list[str],
    sell_features: list[str],
    buy_threshold: float,
    sell_threshold: float,
    cfg: BacktestConfig,
    min_total_trades: int,
    policies: tuple[ExitPolicy, ...] = EXIT_POLICIES,
) -> tuple[ExitPolicy, pd.DataFrame, pd.DataFrame]:
    """Select an exit policy using inner validation only."""
    details: list[dict[str, Any]] = []
    required = list(dict.fromkeys(buy_features + sell_features))
    max_horizon = max(policy.horizon for policy in policies)

    for split_id, (fit_start, fit_end, val_start, val_end) in enumerate(splits, start=1):
        fit = labeled.iloc[fit_start:fit_end].copy().dropna(
            subset=required + ["buy_win", "sell_win"]
        )
        if len(fit) < 1500:
            continue
        y_buy = fit["buy_win"].astype(int)
        y_sell = fit["sell_win"].astype(int)
        if y_buy.nunique() < 2 or y_sell.nunique() < 2:
            continue
        buy_model = _fit_binary_model(fit[buy_features], y_buy)
        sell_model = _fit_binary_model(fit[sell_features], y_sell)
        val = featured.iloc[val_start:val_end].copy().reset_index(drop=True)
        mask = val[required + ["atr14_abs"]].notna().all(axis=1)
        idx = np.flatnonzero(mask.to_numpy())
        idx = idx[idx < len(val) - max_horizon - 1]
        if len(idx) == 0:
            continue
        bp = buy_model.predict_proba(val.iloc[idx][buy_features])[:, 1]
        sp = sell_model.predict_proba(val.iloc[idx][sell_features])[:, 1]
        buy_map = {int(i): float(p) for i, p in zip(idx, bp)}
        sell_map = {int(i): float(p) for i, p in zip(idx, sp)}

        for policy in policies:
            result = _replay_policy(
                val,
                buy_map,
                sell_map,
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold,
                cfg=cfg,
                policy=policy,
            )
            details.append(
                {
                    "inner_split": split_id,
                    "exit_policy_id": policy.policy_id,
                    "exit_mode": policy.mode,
                    "horizon": policy.horizon,
                    "take_profit_atr": policy.take_profit_atr,
                    "stop_loss_atr": policy.stop_loss_atr,
                    "trades": int(result.get("trades", 0)),
                    "win_rate": float(result.get("win_rate", 0.0) or 0.0),
                    "profit_factor": result.get("profit_factor"),
                    "net_profit": float(result.get("net_profit", 0.0) or 0.0),
                    "expectancy": float(result.get("expectancy", 0.0) or 0.0),
                }
            )

    detail_frame = pd.DataFrame(details)
    if detail_frame.empty:
        raise ValueError("No exit-policy candidates were evaluated")
    aggregates: list[dict[str, Any]] = []
    for policy in policies:
        rows = detail_frame.loc[detail_frame["exit_policy_id"] == policy.policy_id].copy()
        score = _robust_score(rows, min_total_trades)
        aggregates.append(
            {
                "exit_policy_id": policy.policy_id,
                "exit_mode": policy.mode,
                "horizon": policy.horizon,
                "take_profit_atr": policy.take_profit_atr,
                "stop_loss_atr": policy.stop_loss_atr,
                "inner_splits_used": len(rows),
                "total_trades": int(rows["trades"].sum()) if not rows.empty else 0,
                "mean_profit_factor": float(rows["profit_factor"].fillna(0).mean()) if not rows.empty else 0.0,
                "std_profit_factor": float(rows["profit_factor"].fillna(0).std(ddof=0)) if not rows.empty else 0.0,
                "mean_expectancy": float(rows["expectancy"].mean()) if not rows.empty else 0.0,
                "positive_split_ratio": float((rows["expectancy"] > 0).mean()) if not rows.empty else 0.0,
                "robust_score": score[0],
            }
        )
    aggregate_frame = pd.DataFrame(aggregates)
    best = aggregate_frame.sort_values(
        ["robust_score", "mean_expectancy", "mean_profit_factor", "total_trades"],
        ascending=False,
    ).iloc[0]
    selected = next(policy for policy in policies if policy.policy_id == best["exit_policy_id"])
    return selected, aggregate_frame, detail_frame
