from __future__ import annotations

from dataclasses import asdict
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
) -> dict[str, Any]:
    balance = cfg.initial_balance
    rows: list[dict[str, Any]] = []
    if not probabilities:
        return _summary(pd.DataFrame(), cfg.initial_balance)

    i = min(probabilities)
    last_i = max(probabilities)
    while i <= last_i:
        p = probabilities.get(i)
        if p is None or not np.isfinite(p) or p < threshold:
            i += 1
            continue
        trade, exit_i = _simulate_trade(
            df=df,
            signal_index=i,
            side=side,
            buy_probability=float(p) if side == "BUY" else 0.0,
            sell_probability=float(p) if side == "SELL" else 0.0,
            cfg=cfg,
            balance=balance,
        )
        rows.append(asdict(trade))
        balance = trade.balance_after
        i = exit_i + 1
    return _summary(pd.DataFrame(rows), cfg.initial_balance)


def _robust_score(rows: pd.DataFrame, min_total_trades: int) -> tuple[float, float, float, int]:
    if rows.empty:
        return (-1e9, -1e9, -1e9, 0)
    total_trades = int(rows["trades"].sum())
    mean_expectancy = float(rows["expectancy"].mean())
    std_expectancy = float(rows["expectancy"].std(ddof=0))
    pf = rows["profit_factor"].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    mean_pf = float(pf.mean())
    std_pf = float(pf.std(ddof=0))
    positive_ratio = float((rows["expectancy"] > 0).mean())

    if total_trades < min_total_trades:
        return (-1e6 + total_trades, mean_expectancy, mean_pf, total_trades)

    # Prefer stable, repeatedly positive candidates over one spectacular inner split.
    score = (
        1.20 * mean_pf
        + 40.0 * mean_expectancy
        + 0.35 * positive_ratio
        - 0.80 * std_pf
        - 30.0 * std_expectancy
    )
    return (score, mean_expectancy, mean_pf, total_trades)


def _build_inner_splits(
    outer_train_start: int,
    outer_train_end: int,
    *,
    inner_splits: int,
    inner_validation_bars: int,
    purge_bars: int,
    min_fit_bars: int,
) -> list[tuple[int, int, int, int]]:
    splits: list[tuple[int, int, int, int]] = []
    for offset in range(inner_splits, 0, -1):
        val_end = outer_train_end - (offset - 1) * inner_validation_bars
        val_start = val_end - inner_validation_bars
        fit_end = val_start - purge_bars
        fit_start = outer_train_start
        if fit_end - fit_start < min_fit_bars:
            continue
        splits.append((fit_start, fit_end, val_start, val_end))
    return splits


def _select_side_candidate(
    labeled: pd.DataFrame,
    featured: pd.DataFrame,
    *,
    side: str,
    architectures: list[str],
    thresholds: list[float],
    splits: list[tuple[int, int, int, int]],
    cfg: BacktestConfig,
    min_total_trades: int,
) -> tuple[str, float, pd.DataFrame, pd.DataFrame]:
    y_col = "buy_win" if side == "BUY" else "sell_win"
    all_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []

    for architecture in architectures:
        features = FEATURE_GROUPS[architecture][side]
        per_threshold: dict[float, list[dict[str, Any]]] = {t: [] for t in thresholds}

        for split_id, (fit_start, fit_end, val_start, val_end) in enumerate(splits, start=1):
            fit = labeled.iloc[fit_start:fit_end].copy().dropna(subset=features + [y_col])
            val = featured.iloc[val_start:val_end].copy().reset_index(drop=True)
            y = fit[y_col].astype(int)
            if len(fit) < 1500 or y.nunique() < 2:
                continue
            model = _fit_binary_model(fit[features], y)
            mask = val[features].notna().all(axis=1)
            idx = np.flatnonzero(mask.to_numpy())
            idx = idx[idx < len(val) - cfg.horizon - 1]
            if len(idx) == 0:
                continue
            prob = model.predict_proba(val.iloc[idx][features])[:, 1]
            prob_map = {int(i): float(p) for i, p in zip(idx, prob)}

            for threshold in thresholds:
                summary = _side_only_replay(
                    val,
                    prob_map,
                    side=side,
                    threshold=threshold,
                    cfg=cfg,
                )
                row = {
                    "side": side,
                    "architecture": architecture,
                    "threshold": threshold,
                    "inner_split": split_id,
                    "trades": int(summary.get("trades", 0)),
                    "win_rate": float(summary.get("win_rate", 0.0) or 0.0),
                    "profit_factor": summary.get("profit_factor"),
                    "net_profit": float(summary.get("net_profit", 0.0) or 0.0),
                    "expectancy": float(summary.get("expectancy", 0.0) or 0.0),
                }
                all_rows.append(row)
                per_threshold[threshold].append(row)

        for threshold, rows in per_threshold.items():
            frame = pd.DataFrame(rows)
            score = _robust_score(frame, min_total_trades)
            aggregate_rows.append(
                {
                    "side": side,
                    "architecture": architecture,
                    "threshold": threshold,
                    "inner_splits_used": int(len(frame)),
                    "total_trades": int(frame["trades"].sum()) if not frame.empty else 0,
                    "mean_profit_factor": float(frame["profit_factor"].fillna(0).mean()) if not frame.empty else 0.0,
                    "std_profit_factor": float(frame["profit_factor"].fillna(0).std(ddof=0)) if not frame.empty else 0.0,
                    "mean_expectancy": float(frame["expectancy"].mean()) if not frame.empty else 0.0,
                    "std_expectancy": float(frame["expectancy"].std(ddof=0)) if not frame.empty else 0.0,
                    "positive_split_ratio": float((frame["expectancy"] > 0).mean()) if not frame.empty else 0.0,
                    "robust_score": score[0],
                }
            )

    aggregates = pd.DataFrame(aggregate_rows)
    details = pd.DataFrame(all_rows)
    if aggregates.empty:
        raise ValueError(f"No robust inner candidates for {side}")
    best = aggregates.sort_values(
        ["robust_score", "mean_expectancy", "mean_profit_factor", "total_trades"],
        ascending=False,
    ).iloc[0]
    return str(best["architecture"]), float(best["threshold"]), aggregates, details


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
    buy_excess = buy_p - buy_threshold
    sell_excess = sell_p - sell_threshold
    if buy_excess - sell_excess >= min_probability_edge:
        return "BUY"
    if sell_excess - buy_excess >= min_probability_edge:
        return "SELL"
    return "HOLD"


def run_nested_robust_v31(
    raw_df: pd.DataFrame,
    cfg: BacktestConfig,
    *,
    thresholds: list[float],
    outer_train_bars: int = 12000,
    inner_validation_bars: int = 1500,
    inner_splits: int = 3,
    outer_test_bars: int = 2000,
    step_bars: int | None = None,
    purge_bars: int | None = None,
    min_total_inner_trades: int = 45,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    thresholds = sorted({float(t) for t in thresholds})
    step_bars = int(step_bars or outer_test_bars)
    purge_bars = int(purge_bars or cfg.horizon)
    if inner_splits < 2:
        raise ValueError("inner_splits must be at least 2")
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
    all_required = _dedupe(buy_features + [c for cols in sell_feature_map.values() for c in cols])

    n = len(featured)
    first_test_start = outer_train_bars + purge_bars
    if first_test_start + outer_test_bars > n:
        raise ValueError("Not enough bars for one robust nested fold")

    all_trades: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    aggregate_frames: list[pd.DataFrame] = []
    detail_frames: list[pd.DataFrame] = []
    balance = cfg.initial_balance
    outer_fold = 0
    test_start = first_test_start

    while test_start + outer_test_bars <= n:
        outer_fold += 1
        outer_train_end = test_start - purge_bars
        outer_train_start = max(0, outer_train_end - outer_train_bars)
        test_end = test_start + outer_test_bars
        splits = _build_inner_splits(
            outer_train_start,
            outer_train_end,
            inner_splits=inner_splits,
            inner_validation_bars=inner_validation_bars,
            purge_bars=purge_bars,
            min_fit_bars=1500,
        )
        if len(splits) < 2:
            test_start += step_bars
            continue

        buy_arch, buy_threshold, buy_agg, buy_detail = _select_side_candidate(
            labeled,
            featured,
            side="BUY",
            architectures=[BUY_GROUP],
            thresholds=thresholds,
            splits=splits,
            cfg=cfg,
            min_total_trades=min_total_inner_trades,
        )
        sell_arch, sell_threshold, sell_agg, sell_detail = _select_side_candidate(
            labeled,
            featured,
            side="SELL",
            architectures=list(SELL_GROUPS),
            thresholds=thresholds,
            splits=splits,
            cfg=cfg,
            min_total_trades=min_total_inner_trades,
        )
        for frame in (buy_agg, sell_agg):
            frame.insert(0, "outer_fold", outer_fold)
            aggregate_frames.append(frame)
        for frame in (buy_detail, sell_detail):
            frame.insert(0, "outer_fold", outer_fold)
            detail_frames.append(frame)

        selection_rows.append(
            {
                "outer_fold": outer_fold,
                "buy_architecture": buy_arch,
                "buy_threshold": buy_threshold,
                "sell_architecture": sell_arch,
                "sell_threshold": sell_threshold,
                "inner_splits_used": len(splits),
                "outer_test_start": str(featured.iloc[test_start]["time"]),
                "outer_test_end": str(featured.iloc[test_end - 1]["time"]),
            }
        )

        outer_refit = labeled.iloc[outer_train_start:outer_train_end].copy().dropna(
            subset=all_required + ["buy_win", "sell_win"]
        )
        test = featured.iloc[test_start:test_end].copy().reset_index(drop=True)
        y_buy = outer_refit["buy_win"].astype(int)
        y_sell = outer_refit["sell_win"].astype(int)
        if len(outer_refit) < 2000 or y_buy.nunique() < 2 or y_sell.nunique() < 2:
            test_start += step_bars
            continue

        buy_model = _fit_binary_model(outer_refit[buy_features], y_buy)
        sell_features = sell_feature_map[sell_arch]
        sell_model = _fit_binary_model(outer_refit[sell_features], y_sell)
        valid = test[buy_features].notna().all(axis=1) & test[sell_features].notna().all(axis=1)
        idx = np.flatnonzero(valid.to_numpy())
        idx = idx[idx < len(test) - cfg.horizon - 1]
        buy_map: dict[int, float] = {}
        sell_map: dict[int, float] = {}
        if len(idx):
            buy_p = buy_model.predict_proba(test.iloc[idx][buy_features])[:, 1]
            sell_p = sell_model.predict_proba(test.iloc[idx][sell_features])[:, 1]
            buy_map = {int(i): float(p) for i, p in zip(idx, buy_p)}
            sell_map = {int(i): float(p) for i, p in zip(idx, sell_p)}

        fold_initial = balance
        fold_trade_rows: list[dict[str, Any]] = []
        common = sorted(set(buy_map).intersection(sell_map))
        if common:
            i = common[0]
            last_i = common[-1]
            while i <= last_i:
                if i not in buy_map or i not in sell_map:
                    i += 1
                    continue
                side = _outer_decision(
                    buy_map[i],
                    sell_map[i],
                    buy_threshold=buy_threshold,
                    sell_threshold=sell_threshold,
                    min_probability_edge=cfg.min_probability_edge,
                )
                if side == "HOLD":
                    i += 1
                    continue
                trade, exit_i = _simulate_trade(
                    df=test,
                    signal_index=i,
                    side=side,
                    buy_probability=buy_map[i],
                    sell_probability=sell_map[i],
                    cfg=cfg,
                    balance=balance,
                )
                row = asdict(trade)
                row.update(
                    {
                        "outer_fold": outer_fold,
                        "selected_buy_threshold": buy_threshold,
                        "selected_sell_architecture": sell_arch,
                        "selected_sell_threshold": sell_threshold,
                    }
                )
                fold_trade_rows.append(row)
                all_trades.append(row)
                balance = trade.balance_after
                i = exit_i + 1

        fold_df = pd.DataFrame(fold_trade_rows)
        fold_summary = _summary(fold_df, fold_initial)
        sides = analyze_sides(fold_df)
        buy_side = sides.loc[sides["side"] == "BUY"].iloc[0].to_dict()
        sell_side = sides.loc[sides["side"] == "SELL"].iloc[0].to_dict()
        fold_rows.append(
            {
                "outer_fold": outer_fold,
                "buy_threshold": buy_threshold,
                "sell_architecture": sell_arch,
                "sell_threshold": sell_threshold,
                "trades": int(fold_summary.get("trades", 0)),
                "win_rate": float(fold_summary.get("win_rate", 0.0) or 0.0),
                "profit_factor": fold_summary.get("profit_factor"),
                "net_profit": float(fold_summary.get("net_profit", 0.0) or 0.0),
                "expectancy": float(fold_summary.get("expectancy", 0.0) or 0.0),
                "buy_net_profit": float(buy_side["net_profit"]),
                "sell_net_profit": float(sell_side["net_profit"]),
                "ending_balance": float(balance),
            }
        )
        test_start += step_bars

    trades = pd.DataFrame(all_trades)
    folds = pd.DataFrame(fold_rows)
    selections = pd.DataFrame(selection_rows)
    aggregates = pd.concat(aggregate_frames, ignore_index=True) if aggregate_frames else pd.DataFrame()
    details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    summary = _summary(trades, cfg.initial_balance)
    summary.update(
        {
            "architecture": "nested_v3_1_robust_multi_inner_split",
            "outer_folds": int(len(folds)),
            "buy_architecture_fixed": BUY_GROUP,
            "sell_architecture_candidates": list(SELL_GROUPS),
            "threshold_candidates": thresholds,
            "outer_train_bars": outer_train_bars,
            "inner_validation_bars": inner_validation_bars,
            "inner_splits": inner_splits,
            "outer_test_bars": outer_test_bars,
            "step_bars": step_bars,
            "purge_bars": purge_bars,
            "min_total_inner_trades": min_total_inner_trades,
            "outer_test_used_for_selection": False,
        }
    )
    return trades, folds, selections, aggregates, details, summary
