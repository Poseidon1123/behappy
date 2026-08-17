from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ai.labeling import create_tp_sl_labels
from backtest.ablation import FEATURE_GROUPS
from backtest.analysis import analyze_sides
from backtest.engine import BacktestConfig, _summary
from backtest.meta_labeling_v41 import (
    _fit_economic_meta_model,
    _generate_oof_economic_meta_training,
    _replay_v41,
)
from backtest.nested_robust import (
    BUY_GROUP,
    SELL_GROUPS,
    _build_inner_splits,
    _select_side_candidate,
)
from backtest.walk_forward import _fit_binary_model
from data.feature_engineering import build_features


def run_meta_labeling_walk_forward_v42(
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
    buy_harvest_threshold: float = 0.45,
    sell_harvest_threshold: float = 0.45,
    min_meta_samples: int = 250,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    """Compare baseline, v4.1 all-side gate, and v4.2 SELL-only gate.

    The three strategies replay every outer test independently. This preserves
    signal ordering: skipping a gated SELL may expose a later BUY opportunity.
    """
    featured = build_features(raw_df).reset_index(drop=True)
    labeled = create_tp_sl_labels(
        featured,
        horizon=cfg.horizon,
        take_profit_pct=cfg.take_profit_pct,
        stop_loss_pct=cfg.stop_loss_pct,
    )

    n = len(featured)
    test_start = outer_train_bars + purge_bars
    outer_fold = 0
    baseline_rows: list[dict[str, Any]] = []
    all_gated_rows: list[dict[str, Any]] = []
    hybrid_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    meta_training_rows: list[dict[str, Any]] = []
    baseline_balance = cfg.initial_balance
    all_gated_balance = cfg.initial_balance
    hybrid_balance = cfg.initial_balance

    while test_start + outer_test_bars <= n:
        outer_fold += 1
        train_end = test_start - purge_bars
        train_start = train_end - outer_train_bars
        test_end = test_start + outer_test_bars
        if train_start < 0:
            test_start += step_bars
            continue

        splits = _build_inner_splits(
            train_start,
            train_end,
            inner_splits=inner_splits,
            inner_validation_bars=inner_validation_bars,
            purge_bars=purge_bars,
            min_fit_bars=1500,
        )
        if len(splits) < 2:
            test_start += step_bars
            continue

        buy_arch, buy_threshold, _, _ = _select_side_candidate(
            labeled,
            featured,
            side="BUY",
            architectures=[BUY_GROUP],
            thresholds=thresholds,
            splits=splits,
            cfg=cfg,
            min_total_trades=min_total_inner_trades,
        )
        sell_arch, sell_threshold, _, _ = _select_side_candidate(
            labeled,
            featured,
            side="SELL",
            architectures=list(SELL_GROUPS),
            thresholds=thresholds,
            splits=splits,
            cfg=cfg,
            min_total_trades=min_total_inner_trades,
        )

        buy_features = FEATURE_GROUPS[buy_arch]["BUY"]
        sell_features = FEATURE_GROUPS[sell_arch]["SELL"]
        buy_meta, sell_meta = _generate_oof_economic_meta_training(
            featured,
            labeled,
            train_start=train_start,
            train_end=train_end,
            buy_features=buy_features,
            sell_features=sell_features,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            buy_harvest_threshold=buy_harvest_threshold,
            sell_harvest_threshold=sell_harvest_threshold,
            cfg=cfg,
            oof_splits=inner_splits,
            oof_validation_bars=inner_validation_bars,
            purge_bars=purge_bars,
        )

        buy_meta_model = None
        sell_meta_model = None
        if len(buy_meta) >= min_meta_samples and buy_meta["meta_target"].nunique() == 2:
            buy_meta_model = _fit_economic_meta_model(buy_meta, "BUY")
        if len(sell_meta) >= min_meta_samples and sell_meta["meta_target"].nunique() == 2:
            sell_meta_model = _fit_economic_meta_model(sell_meta, "SELL")

        meta_training_rows.append(
            {
                "outer_fold": outer_fold,
                "buy_meta_samples": len(buy_meta),
                "sell_meta_samples": len(sell_meta),
                "buy_positive_rate": float(buy_meta["meta_target"].mean()) if not buy_meta.empty else np.nan,
                "sell_positive_rate": float(sell_meta["meta_target"].mean()) if not sell_meta.empty else np.nan,
                "buy_mean_net_pnl": float(buy_meta["net_pnl"].mean()) if not buy_meta.empty else np.nan,
                "sell_mean_net_pnl": float(sell_meta["net_pnl"].mean()) if not sell_meta.empty else np.nan,
                "buy_meta_ready": buy_meta_model is not None,
                "sell_meta_ready": sell_meta_model is not None,
                "buy_threshold": buy_threshold,
                "sell_architecture": sell_arch,
                "sell_threshold": sell_threshold,
                "buy_harvest_threshold": buy_harvest_threshold,
                "sell_harvest_threshold": sell_harvest_threshold,
            }
        )

        required = list(dict.fromkeys(buy_features + sell_features))
        refit = labeled.iloc[train_start:train_end].copy().dropna(
            subset=required + ["buy_win", "sell_win"]
        )
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

        baseline_start = baseline_balance
        all_gated_start = all_gated_balance
        hybrid_start = hybrid_balance
        common = dict(
            test=test,
            buy_prob=buy_map,
            sell_prob=sell_map,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            buy_meta_model=buy_meta_model,
            sell_meta_model=sell_meta_model,
            meta_gate_threshold=meta_gate_threshold,
            cfg=cfg,
            outer_fold=outer_fold,
            sell_architecture=sell_arch,
        )
        baseline_trades, baseline_balance = _replay_v41(
            **common,
            balance=baseline_balance,
            gate_enabled=False,
            meta_version="v3.1_baseline",
        )
        all_gated_trades, all_gated_balance = _replay_v41(
            **common,
            balance=all_gated_balance,
            gate_enabled=True,
            meta_version="v4.1_economic_all_sides",
        )
        hybrid_trades, hybrid_balance = _replay_v41(
            **common,
            balance=hybrid_balance,
            gate_enabled=True,
            gate_sides=frozenset({"SELL"}),
            meta_version="v4.2_sell_only_gate",
        )
        baseline_rows.extend(baseline_trades)
        all_gated_rows.extend(all_gated_trades)
        hybrid_rows.extend(hybrid_trades)

        bs = _summary(pd.DataFrame(baseline_trades), baseline_start)
        ags = _summary(pd.DataFrame(all_gated_trades), all_gated_start)
        hs = _summary(pd.DataFrame(hybrid_trades), hybrid_start)
        fold_rows.append(
            {
                "outer_fold": outer_fold,
                "buy_threshold": buy_threshold,
                "sell_architecture": sell_arch,
                "sell_threshold": sell_threshold,
                "baseline_trades": bs.get("trades", 0),
                "baseline_profit_factor": bs.get("profit_factor"),
                "baseline_net_profit": bs.get("net_profit", 0.0),
                "v41_trades": ags.get("trades", 0),
                "v41_profit_factor": ags.get("profit_factor"),
                "v41_net_profit": ags.get("net_profit", 0.0),
                "v42_trades": hs.get("trades", 0),
                "v42_profit_factor": hs.get("profit_factor"),
                "v42_net_profit": hs.get("net_profit", 0.0),
            }
        )
        test_start += step_bars

    baseline = pd.DataFrame(baseline_rows)
    all_gated = pd.DataFrame(all_gated_rows)
    hybrid = pd.DataFrame(hybrid_rows)
    folds = pd.DataFrame(fold_rows)
    meta_training = pd.DataFrame(meta_training_rows)
    summary = {
        "architecture": "v4_2_hybrid_sell_only_economic_gate",
        "initial_balance": cfg.initial_balance,
        "meta_target": "realized_net_pnl_gt_zero_after_costs",
        "meta_gate_threshold": meta_gate_threshold,
        "buy_gate_enabled": False,
        "sell_gate_enabled": True,
        "outer_test_used_for_meta_training": False,
        "primary_probabilities_for_meta_training": "OOF only",
        "baseline": _summary(baseline, cfg.initial_balance),
        "v41_all_gated": _summary(all_gated, cfg.initial_balance),
        "v42_hybrid": _summary(hybrid, cfg.initial_balance),
        "baseline_sides": analyze_sides(baseline).to_dict(orient="records"),
        "v41_sides": analyze_sides(all_gated).to_dict(orient="records"),
        "v42_sides": analyze_sides(hybrid).to_dict(orient="records"),
    }
    return baseline, all_gated, hybrid, folds, meta_training, summary
