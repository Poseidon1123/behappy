from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

from ai.labeling import create_tp_sl_labels
from backtest.ablation import FEATURE_GROUPS
from backtest.analysis import analyze_sides
from backtest.engine import BacktestConfig, _simulate_trade, _summary
from backtest.nested_robust import (
    BUY_GROUP,
    SELL_GROUPS,
    _build_inner_splits,
    _outer_decision,
    _select_side_candidate,
)
from backtest.walk_forward import _fit_binary_model
from data.feature_engineering import build_features


META_MARKET_FEATURES = [
    "ema_diff",
    "atr_ratio",
    "volatility20",
    "momentum3",
    "momentum5",
    "momentum10",
    "rsi14",
    "range_pos20",
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "hour_sin",
    "hour_cos",
    "london_session",
    "newyork_session",
    "spread_relative",
    "h1_ema_diff",
    "h1_ema_slope",
    "h1_rsi14",
    "h1_atr_ratio",
    "h1_adx14",
    "h1_volatility20",
    "h1_range_pos20",
    "distance_h1_ema20",
    "regime_trending",
    "regime_ranging",
    "regime_high_vol",
    "regime_low_vol",
]

META_FEATURES = [
    "primary_probability",
    "opposing_probability",
    "probability_gap",
    "probability_excess",
] + META_MARKET_FEATURES


def _fit_meta_model(X: pd.DataFrame, y: pd.Series) -> HistGradientBoostingClassifier:
    if y.nunique() < 2:
        raise ValueError("Meta labels need both classes")
    weights = compute_sample_weight(class_weight="balanced", y=y)
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(X, y, sample_weight=weights)
    return model


def _make_meta_rows(
    frame: pd.DataFrame,
    indices: np.ndarray,
    buy_prob: np.ndarray,
    sell_prob: np.ndarray,
    *,
    side: str,
    threshold: float,
    label_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, bp, sp in zip(indices, buy_prob, sell_prob):
        primary = float(bp if side == "BUY" else sp)
        opposing = float(sp if side == "BUY" else bp)
        if primary < threshold:
            continue
        source = frame.iloc[int(idx)]
        target = source.get(label_col)
        if pd.isna(target):
            continue
        row: dict[str, Any] = {
            "local_index": int(idx),
            "side": side,
            "primary_probability": primary,
            "opposing_probability": opposing,
            "probability_gap": primary - opposing,
            "probability_excess": primary - threshold,
            "meta_target": int(target),
        }
        for col in META_MARKET_FEATURES:
            row[col] = source.get(col, np.nan)
        rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        result.replace([np.inf, -np.inf], np.nan, inplace=True)
        result.dropna(subset=META_FEATURES + ["meta_target"], inplace=True)
    return result


def _generate_oof_meta_training(
    featured: pd.DataFrame,
    labeled: pd.DataFrame,
    *,
    train_start: int,
    train_end: int,
    buy_features: list[str],
    sell_features: list[str],
    buy_threshold: float,
    sell_threshold: float,
    cfg: BacktestConfig,
    oof_splits: int = 3,
    oof_validation_bars: int = 1500,
    purge_bars: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build meta training data from primary out-of-fold probabilities only."""
    splits = _build_inner_splits(
        train_start,
        train_end,
        inner_splits=oof_splits,
        inner_validation_bars=oof_validation_bars,
        purge_bars=purge_bars,
        min_fit_bars=1500,
    )
    buy_frames: list[pd.DataFrame] = []
    sell_frames: list[pd.DataFrame] = []

    for fit_start, fit_end, val_start, val_end in splits:
        required = list(dict.fromkeys(buy_features + sell_features))
        fit = labeled.iloc[fit_start:fit_end].copy().dropna(
            subset=required + ["buy_win", "sell_win"]
        )
        val_labeled = labeled.iloc[val_start:val_end].copy().reset_index(drop=True)
        if len(fit) < 1500:
            continue
        y_buy = fit["buy_win"].astype(int)
        y_sell = fit["sell_win"].astype(int)
        if y_buy.nunique() < 2 or y_sell.nunique() < 2:
            continue

        buy_model = _fit_binary_model(fit[buy_features], y_buy)
        sell_model = _fit_binary_model(fit[sell_features], y_sell)
        mask = val_labeled[buy_features].notna().all(axis=1) & val_labeled[sell_features].notna().all(axis=1)
        idx = np.flatnonzero(mask.to_numpy())
        idx = idx[idx < len(val_labeled) - cfg.horizon - 1]
        if len(idx) == 0:
            continue
        bp = buy_model.predict_proba(val_labeled.iloc[idx][buy_features])[:, 1]
        sp = sell_model.predict_proba(val_labeled.iloc[idx][sell_features])[:, 1]
        buy_frames.append(
            _make_meta_rows(
                val_labeled, idx, bp, sp,
                side="BUY", threshold=buy_threshold, label_col="buy_win"
            )
        )
        sell_frames.append(
            _make_meta_rows(
                val_labeled, idx, bp, sp,
                side="SELL", threshold=sell_threshold, label_col="sell_win"
            )
        )

    buy_meta = pd.concat([x for x in buy_frames if not x.empty], ignore_index=True) if any(not x.empty for x in buy_frames) else pd.DataFrame()
    sell_meta = pd.concat([x for x in sell_frames if not x.empty], ignore_index=True) if any(not x.empty for x in sell_frames) else pd.DataFrame()
    return buy_meta, sell_meta


def _meta_vector(row: pd.Series, primary: float, opposing: float, threshold: float) -> pd.DataFrame:
    values: dict[str, float] = {
        "primary_probability": float(primary),
        "opposing_probability": float(opposing),
        "probability_gap": float(primary - opposing),
        "probability_excess": float(primary - threshold),
    }
    for col in META_MARKET_FEATURES:
        values[col] = float(row[col])
    return pd.DataFrame([values], columns=META_FEATURES)


def _replay_with_meta_gate(
    test: pd.DataFrame,
    buy_prob: dict[int, float],
    sell_prob: dict[int, float],
    *,
    buy_threshold: float,
    sell_threshold: float,
    buy_meta_model: HistGradientBoostingClassifier | None,
    sell_meta_model: HistGradientBoostingClassifier | None,
    meta_gate_threshold: float,
    cfg: BacktestConfig,
    balance: float,
    outer_fold: int,
    sell_architecture: str,
    gate_enabled: bool,
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
        bp, sp = buy_prob[i], sell_prob[i]
        side = _outer_decision(
            bp, sp,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            min_probability_edge=cfg.min_probability_edge,
        )
        if side == "HOLD":
            i += 1
            continue

        meta_probability = 1.0
        if gate_enabled:
            primary = bp if side == "BUY" else sp
            opposing = sp if side == "BUY" else bp
            threshold = buy_threshold if side == "BUY" else sell_threshold
            model = buy_meta_model if side == "BUY" else sell_meta_model
            if model is None:
                i += 1
                continue
            x_meta = _meta_vector(test.iloc[i], primary, opposing, threshold)
            if x_meta.isna().any(axis=None):
                i += 1
                continue
            meta_probability = float(model.predict_proba(x_meta)[:, 1][0])
            if meta_probability < meta_gate_threshold:
                i += 1
                continue

        trade, exit_i = _simulate_trade(
            df=test,
            signal_index=i,
            side=side,
            buy_probability=bp,
            sell_probability=sp,
            cfg=cfg,
            balance=balance,
        )
        record = asdict(trade)
        record.update({
            "outer_fold": outer_fold,
            "sell_architecture": sell_architecture,
            "buy_threshold": buy_threshold,
            "sell_threshold": sell_threshold,
            "meta_gate_enabled": gate_enabled,
            "meta_probability": meta_probability,
            "meta_gate_threshold": meta_gate_threshold,
        })
        rows.append(record)
        balance = trade.balance_after
        i = exit_i + 1
    return rows, balance


def run_meta_labeling_walk_forward_v4(
    raw_df: pd.DataFrame,
    cfg: BacktestConfig,
    *,
    thresholds: list[float],
    outer_train_bars: int = 12000,
    inner_validation_bars: int = 1500,
    inner_splits: int = 3,
    outer_test_bars: int = 2000,
    step_bars: int = 2000,
    purge_bars: int = 8,
    min_total_inner_trades: int = 45,
    meta_gate_threshold: float = 0.55,
    min_meta_samples: int = 120,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compare frozen-style robust primary strategy with an OOF-trained meta gate."""
    featured = build_features(raw_df).reset_index(drop=True)
    labeled = create_tp_sl_labels(
        featured,
        horizon=cfg.horizon,
        take_profit_pct=cfg.take_profit_pct,
        stop_loss_pct=cfg.stop_loss_pct,
    )
    n = len(featured)
    first_test_start = outer_train_bars + purge_bars
    baseline_rows: list[dict[str, Any]] = []
    gated_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    meta_training_rows: list[dict[str, Any]] = []
    baseline_balance = cfg.initial_balance
    gated_balance = cfg.initial_balance
    outer_fold = 0
    test_start = first_test_start

    while test_start + outer_test_bars <= n:
        outer_fold += 1
        train_end = test_start - purge_bars
        train_start = train_end - outer_train_bars
        test_end = test_start + outer_test_bars
        if train_start < 0:
            test_start += step_bars
            continue
        splits = _build_inner_splits(
            train_start, train_end,
            inner_splits=inner_splits,
            inner_validation_bars=inner_validation_bars,
            purge_bars=purge_bars,
            min_fit_bars=1500,
        )
        if len(splits) < 2:
            test_start += step_bars
            continue

        buy_arch, buy_threshold, _, _ = _select_side_candidate(
            labeled, featured,
            side="BUY", architectures=[BUY_GROUP], thresholds=thresholds,
            splits=splits, cfg=cfg, min_total_trades=min_total_inner_trades,
        )
        sell_arch, sell_threshold, _, _ = _select_side_candidate(
            labeled, featured,
            side="SELL", architectures=list(SELL_GROUPS), thresholds=thresholds,
            splits=splits, cfg=cfg, min_total_trades=min_total_inner_trades,
        )
        buy_features = FEATURE_GROUPS[buy_arch]["BUY"]
        sell_features = FEATURE_GROUPS[sell_arch]["SELL"]

        buy_meta, sell_meta = _generate_oof_meta_training(
            featured, labeled,
            train_start=train_start, train_end=train_end,
            buy_features=buy_features, sell_features=sell_features,
            buy_threshold=buy_threshold, sell_threshold=sell_threshold,
            cfg=cfg, oof_splits=inner_splits,
            oof_validation_bars=inner_validation_bars,
            purge_bars=purge_bars,
        )
        buy_meta_model = None
        sell_meta_model = None
        if len(buy_meta) >= min_meta_samples and buy_meta["meta_target"].nunique() == 2:
            buy_meta_model = _fit_meta_model(buy_meta[META_FEATURES], buy_meta["meta_target"].astype(int))
        if len(sell_meta) >= min_meta_samples and sell_meta["meta_target"].nunique() == 2:
            sell_meta_model = _fit_meta_model(sell_meta[META_FEATURES], sell_meta["meta_target"].astype(int))
        meta_training_rows.append({
            "outer_fold": outer_fold,
            "buy_meta_samples": len(buy_meta),
            "sell_meta_samples": len(sell_meta),
            "buy_meta_ready": buy_meta_model is not None,
            "sell_meta_ready": sell_meta_model is not None,
            "buy_threshold": buy_threshold,
            "sell_architecture": sell_arch,
            "sell_threshold": sell_threshold,
        })

        required = list(dict.fromkeys(buy_features + sell_features))
        refit = labeled.iloc[train_start:train_end].copy().dropna(subset=required + ["buy_win", "sell_win"])
        y_buy = refit["buy_win"].astype(int)
        y_sell = refit["sell_win"].astype(int)
        if len(refit) < 2000 or y_buy.nunique() < 2 or y_sell.nunique() < 2:
            test_start += step_bars
            continue
        buy_model = _fit_binary_model(refit[buy_features], y_buy)
        sell_model = _fit_binary_model(refit[sell_features], y_sell)
        test = featured.iloc[test_start:test_end].copy().reset_index(drop=True)
        valid_mask = test[buy_features].notna().all(axis=1) & test[sell_features].notna().all(axis=1)
        idx = np.flatnonzero(valid_mask.to_numpy())
        idx = idx[idx < len(test) - cfg.horizon - 1]
        if len(idx) == 0:
            test_start += step_bars
            continue
        bp = buy_model.predict_proba(test.iloc[idx][buy_features])[:, 1]
        sp = sell_model.predict_proba(test.iloc[idx][sell_features])[:, 1]
        buy_map = {int(i): float(p) for i, p in zip(idx, bp)}
        sell_map = {int(i): float(p) for i, p in zip(idx, sp)}

        fold_baseline_start = baseline_balance
        fold_gated_start = gated_balance
        baseline_trades, baseline_balance = _replay_with_meta_gate(
            test, buy_map, sell_map,
            buy_threshold=buy_threshold, sell_threshold=sell_threshold,
            buy_meta_model=None, sell_meta_model=None,
            meta_gate_threshold=meta_gate_threshold, cfg=cfg,
            balance=baseline_balance, outer_fold=outer_fold,
            sell_architecture=sell_arch, gate_enabled=False,
        )
        gated_trades, gated_balance = _replay_with_meta_gate(
            test, buy_map, sell_map,
            buy_threshold=buy_threshold, sell_threshold=sell_threshold,
            buy_meta_model=buy_meta_model, sell_meta_model=sell_meta_model,
            meta_gate_threshold=meta_gate_threshold, cfg=cfg,
            balance=gated_balance, outer_fold=outer_fold,
            sell_architecture=sell_arch, gate_enabled=True,
        )
        baseline_rows.extend(baseline_trades)
        gated_rows.extend(gated_trades)
        bs = _summary(pd.DataFrame(baseline_trades), fold_baseline_start)
        gs = _summary(pd.DataFrame(gated_trades), fold_gated_start)
        fold_rows.append({
            "outer_fold": outer_fold,
            "buy_threshold": buy_threshold,
            "sell_architecture": sell_arch,
            "sell_threshold": sell_threshold,
            "baseline_trades": bs.get("trades", 0),
            "baseline_profit_factor": bs.get("profit_factor"),
            "baseline_net_profit": bs.get("net_profit", 0.0),
            "gated_trades": gs.get("trades", 0),
            "gated_profit_factor": gs.get("profit_factor"),
            "gated_net_profit": gs.get("net_profit", 0.0),
        })
        test_start += step_bars

    baseline = pd.DataFrame(baseline_rows)
    gated = pd.DataFrame(gated_rows)
    folds = pd.DataFrame(fold_rows)
    meta_training = pd.DataFrame(meta_training_rows)
    baseline_summary = _summary(baseline, cfg.initial_balance)
    gated_summary = _summary(gated, cfg.initial_balance)
    summary = {
        "architecture": "v4_oof_meta_labeling_trade_gate",
        "meta_gate_threshold": meta_gate_threshold,
        "outer_test_used_for_meta_training": False,
        "primary_probabilities_for_meta_training": "OOF only",
        "baseline": baseline_summary,
        "gated": gated_summary,
        "baseline_sides": analyze_sides(baseline).to_dict(orient="records"),
        "gated_sides": analyze_sides(gated).to_dict(orient="records"),
    }
    return baseline, gated, folds, meta_training, summary
