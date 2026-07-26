from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd

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


def load_frozen_manifest(path: str | Path = "config/frozen_v3_1.json") -> tuple[dict[str, Any], str]:
    path = Path(path)
    raw = path.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    return manifest, sha256(raw).hexdigest()


def _validate_manifest(manifest: dict[str, Any], cfg: BacktestConfig) -> None:
    if manifest.get("architecture") != "nested_v3_1_robust_multi_inner_split":
        raise ValueError("Unsupported frozen architecture")
    if manifest.get("buy_architecture_fixed") != BUY_GROUP:
        raise ValueError("Frozen BUY architecture no longer matches code")
    if tuple(manifest.get("sell_architecture_candidates", [])) != SELL_GROUPS:
        raise ValueError("Frozen SELL candidates no longer match code")

    expected = {
        "horizon": cfg.horizon,
        "take_profit_pct": cfg.take_profit_pct,
        "stop_loss_pct": cfg.stop_loss_pct,
        "min_probability_edge": cfg.min_probability_edge,
        "fixed_lot": cfg.fixed_lot,
        "initial_balance": cfg.initial_balance,
        "slippage_points": cfg.slippage_points,
        "commission_per_lot_round_turn": cfg.commission_per_lot_round_turn,
    }
    for key, actual in expected.items():
        frozen = manifest.get(key)
        if frozen is None or abs(float(frozen) - float(actual)) > 1e-12:
            raise ValueError(f"Frozen setting mismatch for {key}: manifest={frozen}, runtime={actual}")


def _simulate_confirmation(
    test: pd.DataFrame,
    buy_map: dict[int, float],
    sell_map: dict[int, float],
    *,
    buy_threshold: float,
    sell_threshold: float,
    sell_architecture: str,
    cfg: BacktestConfig,
) -> pd.DataFrame:
    common = sorted(set(buy_map).intersection(sell_map))
    if not common:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    balance = cfg.initial_balance
    i = common[0]
    last_i = common[-1]

    while i <= last_i:
        if i not in buy_map or i not in sell_map:
            i += 1
            continue
        buy_p = buy_map[i]
        sell_p = sell_map[i]
        side = _outer_decision(
            buy_p,
            sell_p,
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
            buy_probability=buy_p,
            sell_probability=sell_p,
            cfg=cfg,
            balance=balance,
        )
        row = asdict(trade)
        row.update(
            {
                "selected_buy_architecture": BUY_GROUP,
                "selected_sell_architecture": sell_architecture,
                "selected_buy_threshold": buy_threshold,
                "selected_sell_threshold": sell_threshold,
                "confirmation_only": True,
            }
        )
        rows.append(row)
        balance = trade.balance_after
        i = exit_i + 1

    return pd.DataFrame(rows)


def run_forward_confirmation(
    raw_df: pd.DataFrame,
    cfg: BacktestConfig,
    manifest: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run one frozen v3.1 confirmation strictly after the development cutoff.

    Architecture candidates, threshold grid, inner-split count and scoring policy are
    read from the frozen manifest. No sweep/selection uses confirmation observations.
    """
    _validate_manifest(manifest, cfg)

    featured = build_features(raw_df).reset_index(drop=True)
    featured["time"] = pd.to_datetime(featured["time"], utc=True)
    labeled = create_tp_sl_labels(
        featured,
        horizon=cfg.horizon,
        take_profit_pct=cfg.take_profit_pct,
        stop_loss_pct=cfg.stop_loss_pct,
    )

    cutoff = pd.Timestamp(manifest["development_outer_test_end_utc"])
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")

    confirmation_idx = np.flatnonzero((featured["time"] > cutoff).to_numpy())
    if len(confirmation_idx) == 0:
        raise ValueError(
            "No closed bars exist after the frozen development cutoff. "
            "Do not move the cutoff forward to manufacture a result."
        )
    confirmation_start = int(confirmation_idx[0])

    purge_bars = int(manifest["purge_bars"])
    outer_train_bars = int(manifest["outer_train_bars"])
    inner_validation_bars = int(manifest["inner_validation_bars"])
    inner_splits = int(manifest["inner_splits"])
    min_total_inner_trades = int(manifest["min_total_inner_trades"])
    thresholds = [float(x) for x in manifest["threshold_candidates"]]

    train_end = confirmation_start - purge_bars
    train_start = train_end - outer_train_bars
    if train_start < 0:
        raise ValueError("Not enough pre-confirmation history for the frozen 12,000-bar training window")

    splits = _build_inner_splits(
        train_start,
        train_end,
        inner_splits=inner_splits,
        inner_validation_bars=inner_validation_bars,
        purge_bars=purge_bars,
        min_fit_bars=1500,
    )
    if len(splits) != inner_splits:
        raise ValueError("Could not construct all frozen robust inner splits before confirmation")

    buy_arch, buy_threshold, buy_scores, buy_details = _select_side_candidate(
        labeled,
        featured,
        side="BUY",
        architectures=[BUY_GROUP],
        thresholds=thresholds,
        splits=splits,
        cfg=cfg,
        min_total_trades=min_total_inner_trades,
    )
    sell_arch, sell_threshold, sell_scores, sell_details = _select_side_candidate(
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
    required = list(dict.fromkeys(buy_features + sell_features))

    refit = labeled.iloc[train_start:train_end].copy().dropna(
        subset=required + ["buy_win", "sell_win"]
    )
    if len(refit) < 2000:
        raise ValueError("Too few frozen pre-confirmation samples after feature/label filtering")

    y_buy = refit["buy_win"].astype(int)
    y_sell = refit["sell_win"].astype(int)
    if y_buy.nunique() < 2 or y_sell.nunique() < 2:
        raise ValueError("Frozen pre-confirmation training labels do not contain both classes")

    buy_model = _fit_binary_model(refit[buy_features], y_buy)
    sell_model = _fit_binary_model(refit[sell_features], y_sell)

    test = featured.iloc[confirmation_start:].copy().reset_index(drop=True)
    # Keep enough future candles inside the confirmation period for honest horizon-based exits.
    valid_mask = test[buy_features].notna().all(axis=1) & test[sell_features].notna().all(axis=1)
    valid_idx = np.flatnonzero(valid_mask.to_numpy())
    valid_idx = valid_idx[valid_idx < len(test) - cfg.horizon - 1]
    if len(valid_idx) == 0:
        raise ValueError("No valid forward-confirmation prediction rows after the frozen cutoff")

    buy_prob = buy_model.predict_proba(test.iloc[valid_idx][buy_features])[:, 1]
    sell_prob = sell_model.predict_proba(test.iloc[valid_idx][sell_features])[:, 1]
    buy_map = {int(i): float(p) for i, p in zip(valid_idx, buy_prob)}
    sell_map = {int(i): float(p) for i, p in zip(valid_idx, sell_prob)}

    predictions = test.copy()
    predictions["local_index"] = np.arange(len(predictions), dtype=int)
    predictions["buy_probability_raw"] = np.nan
    predictions["sell_probability_raw"] = np.nan
    predictions.loc[valid_idx, "buy_probability_raw"] = buy_prob
    predictions.loc[valid_idx, "sell_probability_raw"] = sell_prob
    predictions["selected_buy_threshold"] = buy_threshold
    predictions["selected_sell_threshold"] = sell_threshold
    predictions["selected_sell_architecture"] = sell_arch

    trades = _simulate_confirmation(
        test,
        buy_map,
        sell_map,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        sell_architecture=sell_arch,
        cfg=cfg,
    )
    side_summary = analyze_sides(trades)
    summary = _summary(trades, cfg.initial_balance)
    summary.update(
        {
            "architecture": manifest["architecture"],
            "freeze_name": manifest["freeze_name"],
            "development_cutoff_utc": str(cutoff),
            "confirmation_start_utc": str(test.iloc[0]["time"]),
            "confirmation_end_utc": str(test.iloc[-1]["time"]),
            "confirmation_bars": int(len(test)),
            "train_start_utc": str(featured.iloc[train_start]["time"]),
            "train_end_utc": str(featured.iloc[train_end - 1]["time"]),
            "purge_bars": purge_bars,
            "buy_architecture": buy_arch,
            "buy_threshold": buy_threshold,
            "sell_architecture": sell_arch,
            "sell_threshold": sell_threshold,
            "inner_splits": inner_splits,
            "confirmation_used_for_selection": False,
            "optimizer_used": False,
        }
    )

    scores = pd.concat(
        [buy_scores.assign(selection_side="BUY"), sell_scores.assign(selection_side="SELL")],
        ignore_index=True,
    )
    details = pd.concat(
        [buy_details.assign(selection_side="BUY"), sell_details.assign(selection_side="SELL")],
        ignore_index=True,
    )
    return trades, predictions, side_summary, scores, details, summary
