from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from ai.labeling import create_tp_sl_labels
from backtest.analysis import analyze_sides, threshold_sweep
from backtest.engine import BacktestConfig, _decision, _simulate_trade, _summary
from backtest.walk_forward import _fit_binary_model
from data.feature_engineering import (
    H1_CONTEXT_FEATURES,
    M15_COMMON_FEATURES,
    build_features,
)


RAW_H1_FEATURES = [
    "h1_ema_diff",
    "h1_ema_slope",
    "h1_rsi14",
    "h1_atr_ratio",
    "h1_adx14",
    "h1_volatility20",
    "h1_range_pos20",
    "distance_h1_ema20",
]

REGIME_FEATURES = [
    "regime_trending",
    "regime_ranging",
    "regime_high_vol",
    "regime_low_vol",
]

BUY_M15_ASYMMETRIC = ["lower_wick_ratio", "bullish_body"]
SELL_M15_ASYMMETRIC = ["upper_wick_ratio", "bearish_body"]
BUY_TREND_FLAGS = ["h1_trend_up", "m15_h1_bull_alignment"]
SELL_TREND_FLAGS = ["h1_trend_down", "m15_h1_bear_alignment"]


FEATURE_GROUPS: dict[str, dict[str, list[str]]] = {
    # A: closest comparable baseline: M15 only, but BUY/SELL keep their asymmetric
    # candle-shape variables.
    "A_m15_baseline": {
        "BUY": M15_COMMON_FEATURES + BUY_M15_ASYMMETRIC,
        "SELL": M15_COMMON_FEATURES + SELL_M15_ASYMMETRIC,
    },
    # B: add continuous/raw H1 context only. No regime or hand-built trend flags.
    "B_m15_raw_h1": {
        "BUY": M15_COMMON_FEATURES + BUY_M15_ASYMMETRIC + RAW_H1_FEATURES,
        "SELL": M15_COMMON_FEATURES + SELL_M15_ASYMMETRIC + RAW_H1_FEATURES,
    },
    # C: full v2 architecture currently used by the main pipeline.
    "C_m15_h1_regime_full": {
        "BUY": M15_COMMON_FEATURES + BUY_M15_ASYMMETRIC + H1_CONTEXT_FEATURES + BUY_TREND_FLAGS + ["distance_h1_ema20"],
        "SELL": M15_COMMON_FEATURES + SELL_M15_ASYMMETRIC + H1_CONTEXT_FEATURES + SELL_TREND_FLAGS + ["distance_h1_ema20"],
    },
    # D: H1 + regime, but remove the binary direction/alignment conclusions.
    "D_m15_h1_regime_no_alignment": {
        "BUY": M15_COMMON_FEATURES + BUY_M15_ASYMMETRIC + RAW_H1_FEATURES + REGIME_FEATURES,
        "SELL": M15_COMMON_FEATURES + SELL_M15_ASYMMETRIC + RAW_H1_FEATURES + REGIME_FEATURES,
    },
    # E: market-regime flags only on top of M15; no raw H1 indicators.
    "E_m15_regime_only": {
        "BUY": M15_COMMON_FEATURES + BUY_M15_ASYMMETRIC + REGIME_FEATURES,
        "SELL": M15_COMMON_FEATURES + SELL_M15_ASYMMETRIC + REGIME_FEATURES,
    },
}


def _dedupe(columns: list[str]) -> list[str]:
    return list(dict.fromkeys(columns))


for _group in FEATURE_GROUPS.values():
    _group["BUY"] = _dedupe(_group["BUY"])
    _group["SELL"] = _dedupe(_group["SELL"])


def _simulate_from_predictions(
    fold_df: pd.DataFrame,
    cfg: BacktestConfig,
    balance: float,
) -> tuple[list[dict[str, Any]], float]:
    probs = {
        int(row.local_index): (float(row.buy_probability_raw), float(row.sell_probability_raw))
        for row in fold_df.itertuples()
        if np.isfinite(row.buy_probability_raw) and np.isfinite(row.sell_probability_raw)
    }
    if not probs:
        return [], balance

    rows: list[dict[str, Any]] = []
    i = min(probs)
    last_i = max(probs)
    while i <= last_i:
        pair = probs.get(i)
        if pair is None:
            i += 1
            continue
        buy_p, sell_p = pair
        side = _decision(buy_p, sell_p, cfg)
        if side == "HOLD":
            i += 1
            continue
        trade, exit_i = _simulate_trade(
            df=fold_df,
            signal_index=i,
            side=side,
            buy_probability=buy_p,
            sell_probability=sell_p,
            cfg=cfg,
            balance=balance,
        )
        row = asdict(trade)
        row["fold"] = int(fold_df["fold"].iloc[0])
        rows.append(row)
        balance = trade.balance_after
        i = exit_i + 1
    return rows, balance


def run_feature_group_walk_forward(
    raw_df: pd.DataFrame,
    cfg: BacktestConfig,
    *,
    group_name: str,
    buy_features: list[str],
    sell_features: list[str],
    train_bars: int = 12000,
    calibration_bars: int = 2000,
    test_bars: int = 2000,
    step_bars: int | None = None,
    purge_bars: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run one raw-probability feature ablation on identical chronological folds.

    Calibration windows are retained in the layout so every ablation uses exactly
    the same train/test timestamps as the calibrated main walk-forward. They are
    intentionally not used for fitting here; ablation focuses on feature quality.
    """
    if group_name not in FEATURE_GROUPS:
        raise ValueError(f"Unknown feature group: {group_name}")

    step_bars = int(step_bars or test_bars)
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

    all_required = _dedupe(buy_features + sell_features)
    n = len(featured)
    first_test_start = train_bars + purge_bars + calibration_bars + purge_bars
    if first_test_start + test_bars > n:
        raise ValueError("Not enough bars for one ablation walk-forward fold")

    all_predictions: list[pd.DataFrame] = []
    all_trades: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    balance = cfg.initial_balance
    fold_id = 0
    test_start = first_test_start

    while test_start + test_bars <= n:
        fold_id += 1
        calibration_end = test_start - purge_bars
        calibration_start = calibration_end - calibration_bars
        train_end = calibration_start - purge_bars
        train_start = max(0, train_end - train_bars)
        test_end = test_start + test_bars

        train_slice = labeled.iloc[train_start:train_end].copy().dropna(
            subset=all_required + ["buy_win", "sell_win"]
        )
        test_slice = featured.iloc[test_start:test_end].copy().reset_index(drop=True)

        if len(train_slice) < 1500:
            test_start += step_bars
            continue

        y_buy = train_slice["buy_win"].astype(int)
        y_sell = train_slice["sell_win"].astype(int)
        if y_buy.nunique() < 2 or y_sell.nunique() < 2:
            test_start += step_bars
            continue

        buy_model = _fit_binary_model(train_slice[buy_features], y_buy)
        sell_model = _fit_binary_model(train_slice[sell_features], y_sell)

        valid_mask = (
            test_slice[buy_features].notna().all(axis=1)
            & test_slice[sell_features].notna().all(axis=1)
        )
        valid_indices = np.flatnonzero(valid_mask.to_numpy())
        valid_indices = valid_indices[valid_indices < len(test_slice) - cfg.horizon - 1]

        export = test_slice.copy()
        export["fold"] = fold_id
        export["local_index"] = np.arange(len(export), dtype=int)
        export["global_index"] = test_start + export["local_index"]
        export["feature_group"] = group_name
        export["buy_probability_raw"] = np.nan
        export["sell_probability_raw"] = np.nan

        if len(valid_indices):
            buy_prob = buy_model.predict_proba(test_slice.iloc[valid_indices][buy_features])[:, 1]
            sell_prob = sell_model.predict_proba(test_slice.iloc[valid_indices][sell_features])[:, 1]
            export.loc[valid_indices, "buy_probability_raw"] = buy_prob
            export.loc[valid_indices, "sell_probability_raw"] = sell_prob

        fold_trades, balance = _simulate_from_predictions(export, cfg, balance)
        all_trades.extend(fold_trades)
        all_predictions.append(export)

        fold_df = pd.DataFrame(fold_trades)
        fold_summary = _summary(fold_df, balance - float(fold_df["net_pnl"].sum()) if not fold_df.empty else balance)
        fold_rows.append(
            {
                "feature_group": group_name,
                "fold": fold_id,
                "train_start": str(featured.iloc[train_start]["time"]),
                "train_end": str(featured.iloc[train_end - 1]["time"]),
                "test_start": str(featured.iloc[test_start]["time"]),
                "test_end": str(featured.iloc[test_end - 1]["time"]),
                "train_samples": int(len(train_slice)),
                "test_bars": int(len(test_slice)),
                "trades": int(len(fold_df)),
                "net_profit": float(fold_df["net_pnl"].sum()) if not fold_df.empty else 0.0,
                "profit_factor": fold_summary.get("profit_factor"),
            }
        )
        test_start += step_bars

    trades = pd.DataFrame(all_trades)
    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    folds = pd.DataFrame(fold_rows)
    summary = _summary(trades, cfg.initial_balance)
    summary.update(
        {
            "feature_group": group_name,
            "buy_feature_count": len(buy_features),
            "sell_feature_count": len(sell_features),
            "walk_forward_folds": int(len(folds)),
            "out_of_sample_only": True,
        }
    )
    return trades, predictions, folds, summary


def run_ablation_suite(
    raw_df: pd.DataFrame,
    cfg: BacktestConfig,
    *,
    thresholds: list[float],
    train_bars: int,
    calibration_bars: int,
    test_bars: int,
    step_bars: int,
    purge_bars: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    summary_rows: list[dict[str, Any]] = []
    side_frames: list[pd.DataFrame] = []
    sweep_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []
    trade_logs: dict[str, pd.DataFrame] = {}

    for group_name, feature_sets in FEATURE_GROUPS.items():
        trades, predictions, folds, summary = run_feature_group_walk_forward(
            raw_df,
            cfg,
            group_name=group_name,
            buy_features=feature_sets["BUY"],
            sell_features=feature_sets["SELL"],
            train_bars=train_bars,
            calibration_bars=calibration_bars,
            test_bars=test_bars,
            step_bars=step_bars,
            purge_bars=purge_bars,
        )
        summary_rows.append(summary)
        trade_logs[group_name] = trades

        sides = analyze_sides(trades)
        sides.insert(0, "feature_group", group_name)
        side_frames.append(sides)

        # threshold_sweep expects calibrated naming plus fold/local_index; raw is present.
        sweep = threshold_sweep(predictions, cfg, thresholds, method="raw")
        sweep.insert(0, "feature_group", group_name)
        sweep_frames.append(sweep)

        if not folds.empty:
            fold_frames.append(folds)

    summaries = pd.DataFrame(summary_rows)
    sides = pd.concat(side_frames, ignore_index=True) if side_frames else pd.DataFrame()
    sweeps = pd.concat(sweep_frames, ignore_index=True) if sweep_frames else pd.DataFrame()
    folds = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    return summaries, sides, sweeps, folds, trade_logs
