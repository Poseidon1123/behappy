from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ai.labeling import create_tp_sl_labels
from backtest.ablation import FEATURE_GROUPS
from backtest.drift import DRIFT_FEATURES, build_causal_drift_map
from backtest.exit_selection import select_exit_policy
from backtest.meta_labeling_v41 import (
    _fit_economic_meta_model,
    _generate_oof_economic_meta_training,
)
from backtest.nested_robust import (
    BUY_GROUP,
    SELL_GROUPS,
    _build_inner_splits,
    _select_side_candidate,
)
from backtest.walk_forward import _fit_binary_model
from data.feature_engineering import build_features


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_v51_bundle(
    raw_df,
    snapshot_manifest: dict[str, Any],
    cfg,
    *,
    thresholds: list[float],
    outer_train_bars: int,
    inner_validation_bars: int,
    inner_splits: int,
    purge_bars: int,
    min_total_inner_trades: int,
    meta_gate_threshold: float,
    buy_harvest_threshold: float,
    sell_harvest_threshold: float,
    min_meta_samples: int,
    drift_recent_window: int,
    drift_score_step: int,
    drift_calibration_step: int,
    drift_cutoff_quantile: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    timeframe = str(snapshot_manifest.get("timeframe", "M15"))
    featured = build_features(raw_df, base_timeframe=timeframe).reset_index(drop=True)
    labeled = create_tp_sl_labels(
        featured,
        horizon=cfg.horizon,
        take_profit_pct=cfg.take_profit_pct,
        stop_loss_pct=cfg.stop_loss_pct,
    )
    train_end = len(featured)
    train_start = train_end - outer_train_bars
    if train_start < 0:
        raise ValueError("Snapshot is shorter than outer_train_bars")
    splits = _build_inner_splits(
        train_start,
        train_end,
        inner_splits=inner_splits,
        inner_validation_bars=inner_validation_bars,
        purge_bars=purge_bars,
        min_fit_bars=1500,
    )
    if len(splits) < 2:
        raise ValueError("Not enough inner splits to freeze v5.1")

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
    if len(sell_meta) < min_meta_samples or sell_meta["meta_target"].nunique() != 2:
        raise ValueError("SELL meta model is not ready; refusing to freeze candidate")
    sell_meta_model = _fit_economic_meta_model(sell_meta, "SELL")

    selected_exit, exit_aggregates, _ = select_exit_policy(
        labeled,
        featured,
        splits=splits,
        buy_features=buy_features,
        sell_features=sell_features,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        cfg=cfg,
        min_total_trades=min_total_inner_trades,
    )
    required = list(dict.fromkeys(buy_features + sell_features))
    refit = labeled.iloc[train_start:train_end].copy().dropna(
        subset=required + ["buy_win", "sell_win"]
    )
    y_buy = refit["buy_win"].astype(int)
    y_sell = refit["sell_win"].astype(int)
    if y_buy.nunique() < 2 or y_sell.nunique() < 2:
        raise ValueError("Primary labels need both classes")
    buy_model = _fit_binary_model(refit[buy_features], y_buy)
    sell_model = _fit_binary_model(refit[sell_features], y_sell)

    drift = build_causal_drift_map(
        featured,
        train_start=train_start,
        train_end=train_end,
        test_start=train_end,
        test_end=train_end,
        recent_window=drift_recent_window,
        score_step=drift_score_step,
        calibration_step=drift_calibration_step,
        cutoff_quantile=drift_cutoff_quantile,
    )
    bundle = {
        "bundle_version": 1,
        "architecture": "v5.1_atr_adaptive_shadow",
        "execution_mode": "SHADOW_ONLY",
        "deployment_allowed": False,
        "research_status": "DEVELOPMENT_DEMO_ONLY",
        "buy_model": buy_model,
        "sell_model": sell_model,
        "sell_meta_model": sell_meta_model,
        "buy_features": buy_features,
        "sell_features": sell_features,
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
        "min_probability_edge": cfg.min_probability_edge,
        "meta_gate_threshold": meta_gate_threshold,
        "exit_policy": selected_exit,
        "drift_reference": featured.iloc[train_start:train_end][DRIFT_FEATURES].copy(),
        "drift_cutoff": drift.cutoff,
        "drift_recent_window": drift_recent_window,
        "drift_score_step": drift_score_step,
        "backtest_config": cfg,
        "snapshot_manifest": snapshot_manifest,
        "training_cutoff_utc": featured.iloc[-1]["time"].isoformat(),
        "raw_context_tail": raw_df.tail(
            max(
                1500,
                drift_recent_window + 300,
                {"M1": 6000, "M5": 1500, "M15": 1500, "M30": 1500, "H1": 1500, "H4": 1500}.get(timeframe.upper(), 1500),
            )
        ).copy(),
    }
    manifest = {
        "bundle_version": 1,
        "architecture": bundle["architecture"],
        "execution_mode": "SHADOW_ONLY",
        "deployment_allowed": False,
        "research_status": "DEVELOPMENT_DEMO_ONLY",
        "snapshot_sha256": snapshot_manifest["sha256"],
        "training_cutoff_utc": bundle["training_cutoff_utc"],
        "buy_architecture": buy_arch,
        "sell_architecture": sell_arch,
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
        "meta_gate_threshold": meta_gate_threshold,
        "sell_meta_samples": len(sell_meta),
        "sell_meta_positive_rate": float(sell_meta["meta_target"].mean()),
        "drift_cutoff": drift.cutoff,
        "drift_score_step": drift_score_step,
        "drift_cutoff_quantile": drift_cutoff_quantile,
        "exit_policy": {
            "policy_id": selected_exit.policy_id,
            "mode": selected_exit.mode,
            "horizon": selected_exit.horizon,
            "take_profit_atr": selected_exit.take_profit_atr,
            "stop_loss_atr": selected_exit.stop_loss_atr,
        },
        "exit_candidate_scores": exit_aggregates.replace({np.nan: None}).to_dict(orient="records"),
    }
    return bundle, manifest


def save_bundle(
    bundle: dict[str, Any],
    manifest: dict[str, Any],
    output_path: str | Path,
) -> tuple[Path, Path, dict[str, Any]]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    final_manifest = dict(manifest)
    final_manifest["bundle_sha256"] = _sha256(output)
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(final_manifest, indent=2), encoding="utf-8")
    return output, manifest_path, final_manifest
