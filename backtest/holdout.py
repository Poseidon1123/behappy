from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ai.labeling import create_tp_sl_labels
from backtest.analysis import analyze_sides, run_prediction_candidate
from backtest.engine import BacktestConfig
from backtest.walk_forward import _fit_binary_model, _fit_calibrators
from data.feature_engineering import (
    BUY_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    SELL_FEATURE_COLUMNS,
    build_features,
)


@dataclass(frozen=True)
class LockedCandidate:
    name: str
    method: str
    threshold: float


# Historical v1 candidates are retained only for reproducibility of the already
# consumed holdout. Do not reuse that consumed period for v2 feature selection.
LOCKED_CANDIDATES: tuple[LockedCandidate, ...] = (
    LockedCandidate("raw_055", "raw", 0.55),
    LockedCandidate("raw_060", "raw", 0.60),
    LockedCandidate("raw_080", "raw", 0.80),
    LockedCandidate("sigmoid_020", "sigmoid", 0.20),
)


def build_untouched_holdout_predictions(
    raw_df: pd.DataFrame,
    cfg: BacktestConfig,
    *,
    recent_bars_excluded: int = 50000,
    train_bars: int = 12000,
    calibration_bars: int = 2000,
    holdout_bars: int = 5000,
    purge_bars: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    purge_bars = int(purge_bars or cfg.horizon)
    if purge_bars < cfg.horizon:
        raise ValueError("purge_bars must be at least the label horizon")
    if recent_bars_excluded < 1:
        raise ValueError("recent_bars_excluded must be positive")
    if holdout_bars < 500:
        raise ValueError("holdout_bars must be at least 500")

    featured = build_features(raw_df).reset_index(drop=True)
    labeled = create_tp_sl_labels(
        featured,
        horizon=cfg.horizon,
        take_profit_pct=cfg.take_profit_pct,
        stop_loss_pct=cfg.stop_loss_pct,
    )

    n = len(featured)
    eligible_end = n - recent_bars_excluded
    required = train_bars + purge_bars + calibration_bars + purge_bars + holdout_bars
    if eligible_end < required:
        raise ValueError(
            "Not enough untouched historical bars. "
            f"Need at least {required + recent_bars_excluded:,} total bars, got {n:,}."
        )

    holdout_end = eligible_end
    holdout_start = holdout_end - holdout_bars
    calibration_end = holdout_start - purge_bars
    calibration_start = calibration_end - calibration_bars
    train_end = calibration_start - purge_bars
    train_start = train_end - train_bars

    train_slice = labeled.iloc[train_start:train_end].copy().dropna(
        subset=FEATURE_COLUMNS + ["buy_win", "sell_win"]
    )
    calibration_slice = labeled.iloc[calibration_start:calibration_end].copy().dropna(
        subset=FEATURE_COLUMNS + ["buy_win", "sell_win"]
    )
    holdout_slice = featured.iloc[holdout_start:holdout_end].copy().reset_index(drop=True)

    if len(train_slice) < 1500:
        raise ValueError("Too few valid training samples in untouched segment")
    if len(calibration_slice) < 100:
        raise ValueError("Too few valid calibration samples in untouched segment")

    y_buy_train = train_slice["buy_win"].astype(int)
    y_sell_train = train_slice["sell_win"].astype(int)
    y_buy_cal = calibration_slice["buy_win"].astype(int)
    y_sell_cal = calibration_slice["sell_win"].astype(int)
    if any(y.nunique() < 2 for y in (y_buy_train, y_sell_train, y_buy_cal, y_sell_cal)):
        raise ValueError("Untouched fit/calibration blocks must contain both classes")

    buy_model = _fit_binary_model(train_slice[BUY_FEATURE_COLUMNS], y_buy_train)
    sell_model = _fit_binary_model(train_slice[SELL_FEATURE_COLUMNS], y_sell_train)
    buy_calibrators = _fit_calibrators(
        buy_model, calibration_slice[BUY_FEATURE_COLUMNS], y_buy_cal
    )
    sell_calibrators = _fit_calibrators(
        sell_model, calibration_slice[SELL_FEATURE_COLUMNS], y_sell_cal
    )

    export = holdout_slice.copy()
    export["fold"] = 1
    export["local_index"] = np.arange(len(export), dtype=int)
    export["global_index"] = holdout_start + export["local_index"]
    for side in ("buy", "sell"):
        for method in ("raw", "sigmoid", "isotonic"):
            export[f"{side}_probability_{method}"] = np.nan
    export["buy_win"] = np.nan
    export["sell_win"] = np.nan

    holdout_labeled = create_tp_sl_labels(
        holdout_slice,
        horizon=cfg.horizon,
        take_profit_pct=cfg.take_profit_pct,
        stop_loss_pct=cfg.stop_loss_pct,
    )
    if not holdout_labeled.empty:
        label_count = len(holdout_labeled)
        export.loc[: label_count - 1, "buy_win"] = holdout_labeled["buy_win"].to_numpy()
        export.loc[: label_count - 1, "sell_win"] = holdout_labeled["sell_win"].to_numpy()

    valid_mask = holdout_slice[FEATURE_COLUMNS].notna().all(axis=1)
    valid_indices = np.flatnonzero(valid_mask.to_numpy())
    valid_indices = valid_indices[valid_indices < len(holdout_slice) - cfg.horizon - 1]
    if len(valid_indices) == 0:
        raise ValueError("No valid feature rows in final holdout")

    buy_raw = buy_model.predict_proba(
        holdout_slice.iloc[valid_indices][BUY_FEATURE_COLUMNS]
    )[:, 1]
    sell_raw = sell_model.predict_proba(
        holdout_slice.iloc[valid_indices][SELL_FEATURE_COLUMNS]
    )[:, 1]
    for method in ("raw", "sigmoid", "isotonic"):
        export.loc[valid_indices, f"buy_probability_{method}"] = buy_calibrators[method].transform(buy_raw)
        export.loc[valid_indices, f"sell_probability_{method}"] = sell_calibrators[method].transform(sell_raw)

    manifest = {
        "locked": True,
        "optimizer_allowed": False,
        "architecture": "multi_timeframe_m15_h1_regime_side_specific_v2",
        "recent_bars_excluded": int(recent_bars_excluded),
        "train_bars": int(train_bars),
        "calibration_bars": int(calibration_bars),
        "holdout_bars": int(holdout_bars),
        "purge_bars": int(purge_bars),
        "train_start": str(featured.iloc[train_start]["time"]),
        "train_end": str(featured.iloc[train_end - 1]["time"]),
        "calibration_start": str(featured.iloc[calibration_start]["time"]),
        "calibration_end": str(featured.iloc[calibration_end - 1]["time"]),
        "holdout_start": str(featured.iloc[holdout_start]["time"]),
        "holdout_end": str(featured.iloc[holdout_end - 1]["time"]),
        "excluded_recent_start": str(featured.iloc[eligible_end]["time"]),
        "excluded_recent_end": str(featured.iloc[-1]["time"]),
        "candidates": [
            {"name": c.name, "method": c.method, "threshold": c.threshold}
            for c in LOCKED_CANDIDATES
        ],
    }
    return export, manifest


def evaluate_locked_candidates(
    predictions: pd.DataFrame,
    cfg: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    side_rows: list[pd.DataFrame] = []
    trade_logs: dict[str, pd.DataFrame] = {}

    for candidate in LOCKED_CANDIDATES:
        trades, summary = run_prediction_candidate(
            predictions,
            cfg,
            method=candidate.method,
            threshold=candidate.threshold,
        )
        side = analyze_sides(trades)
        side["candidate"] = candidate.name
        side["method"] = candidate.method
        side["threshold"] = candidate.threshold
        side_rows.append(side)
        trade_logs[candidate.name] = trades

        buy = side.loc[side["side"] == "BUY"].iloc[0]
        sell = side.loc[side["side"] == "SELL"].iloc[0]
        rows.append(
            {
                "candidate": candidate.name,
                "method": candidate.method,
                "threshold": candidate.threshold,
                **summary,
                "buy_profit_factor": buy["profit_factor"],
                "buy_net_profit": float(buy["net_profit"]),
                "sell_profit_factor": sell["profit_factor"],
                "sell_net_profit": float(sell["net_profit"]),
            }
        )

    return pd.DataFrame(rows), pd.concat(side_rows, ignore_index=True), trade_logs
