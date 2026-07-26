from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

from ai.labeling import create_tp_sl_labels
from backtest.calibration import ProbabilityCalibrator
from backtest.engine import BacktestConfig, _decision, _simulate_trade, _summary
from data.feature_engineering import FEATURE_COLUMNS, build_features


def _fit_binary_model(x_train: pd.DataFrame, y_train: pd.Series) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=250,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=42,
    )
    weights = compute_sample_weight(class_weight="balanced", y=y_train)
    model.fit(x_train, y_train, sample_weight=weights)
    return model


def _fit_calibrators(
    model: HistGradientBoostingClassifier,
    x_cal: pd.DataFrame,
    y_cal: pd.Series,
) -> dict[str, ProbabilityCalibrator]:
    raw = model.predict_proba(x_cal)[:, 1]
    calibrators: dict[str, ProbabilityCalibrator] = {
        "raw": ProbabilityCalibrator("raw").fit(raw, y_cal.to_numpy()),
        "sigmoid": ProbabilityCalibrator("sigmoid").fit(raw, y_cal.to_numpy()),
        "isotonic": ProbabilityCalibrator("isotonic").fit(raw, y_cal.to_numpy()),
    }
    return calibrators


def run_walk_forward_backtest(
    raw_df: pd.DataFrame,
    backtest_config: BacktestConfig,
    train_bars: int = 12000,
    calibration_bars: int = 2000,
    test_bars: int = 2000,
    step_bars: int | None = None,
    purge_bars: int | None = None,
    calibration_method: str = "raw",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run leakage-safe walk-forward training, calibration and testing.

    Fold layout:
        model-fit history -> purge -> calibration history -> purge -> future test

    Raw, sigmoid and isotonic probabilities are exported for every OOS test row.
    `calibration_method` only selects which probability stream drives the baseline
    simulated trades; all three streams are retained for later comparison.
    """
    if train_bars < 2000:
        raise ValueError("train_bars must be at least 2000")
    if calibration_bars < 200:
        raise ValueError("calibration_bars must be at least 200")
    if test_bars < 200:
        raise ValueError("test_bars must be at least 200")
    if calibration_method not in {"raw", "sigmoid", "isotonic"}:
        raise ValueError("calibration_method must be raw, sigmoid or isotonic")

    step_bars = int(step_bars or test_bars)
    purge_bars = int(purge_bars or backtest_config.horizon)
    if step_bars <= 0:
        raise ValueError("step_bars must be positive")
    if purge_bars < backtest_config.horizon:
        raise ValueError("purge_bars must be at least as large as horizon")

    featured = build_features(raw_df).reset_index(drop=True)
    labeled = create_tp_sl_labels(
        featured,
        horizon=backtest_config.horizon,
        take_profit_pct=backtest_config.take_profit_pct,
        stop_loss_pct=backtest_config.stop_loss_pct,
    )

    n = len(featured)
    first_test_start = train_bars + purge_bars + calibration_bars + purge_bars
    if first_test_start + test_bars > n:
        raise ValueError("Not enough bars for one calibrated walk-forward fold")

    all_trades: list[dict[str, Any]] = []
    all_predictions: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    balance = backtest_config.initial_balance
    fold_id = 0

    test_start = first_test_start
    while test_start + test_bars <= n:
        fold_id += 1

        calibration_end = test_start - purge_bars
        calibration_start = calibration_end - calibration_bars
        train_end = calibration_start - purge_bars
        train_start = max(0, train_end - train_bars)
        test_end = test_start + test_bars

        if train_end <= train_start or calibration_start < 0:
            test_start += step_bars
            continue

        train_slice = labeled.iloc[train_start:train_end].copy().dropna(
            subset=FEATURE_COLUMNS + ["buy_win", "sell_win"]
        )
        calibration_slice = labeled.iloc[calibration_start:calibration_end].copy().dropna(
            subset=FEATURE_COLUMNS + ["buy_win", "sell_win"]
        )
        test_slice = featured.iloc[test_start:test_end].copy().reset_index(drop=True)

        if len(train_slice) < 1500 or len(calibration_slice) < 100:
            test_start += step_bars
            continue

        y_buy_train = train_slice["buy_win"].astype(int)
        y_sell_train = train_slice["sell_win"].astype(int)
        y_buy_cal = calibration_slice["buy_win"].astype(int)
        y_sell_cal = calibration_slice["sell_win"].astype(int)

        if (
            y_buy_train.nunique() < 2
            or y_sell_train.nunique() < 2
            or y_buy_cal.nunique() < 2
            or y_sell_cal.nunique() < 2
        ):
            test_start += step_bars
            continue

        x_train = train_slice[FEATURE_COLUMNS]
        buy_model = _fit_binary_model(x_train, y_buy_train)
        sell_model = _fit_binary_model(x_train, y_sell_train)

        x_cal = calibration_slice[FEATURE_COLUMNS]
        buy_calibrators = _fit_calibrators(buy_model, x_cal, y_buy_cal)
        sell_calibrators = _fit_calibrators(sell_model, x_cal, y_sell_cal)

        valid_mask = test_slice[FEATURE_COLUMNS].notna().all(axis=1)
        valid_local_indices = np.flatnonzero(valid_mask.to_numpy())
        valid_local_indices = valid_local_indices[
            valid_local_indices < len(test_slice) - backtest_config.horizon - 1
        ]

        test_export = test_slice.copy()
        test_export["fold"] = fold_id
        test_export["local_index"] = np.arange(len(test_export), dtype=int)
        test_export["global_index"] = test_start + test_export["local_index"]
        for side in ("buy", "sell"):
            for method in ("raw", "sigmoid", "isotonic"):
                test_export[f"{side}_probability_{method}"] = np.nan
        test_export["buy_win"] = np.nan
        test_export["sell_win"] = np.nan

        test_labeled = create_tp_sl_labels(
            test_slice,
            horizon=backtest_config.horizon,
            take_profit_pct=backtest_config.take_profit_pct,
            stop_loss_pct=backtest_config.stop_loss_pct,
        )
        if not test_labeled.empty:
            label_count = len(test_labeled)
            test_export.loc[: label_count - 1, "buy_win"] = test_labeled["buy_win"].to_numpy()
            test_export.loc[: label_count - 1, "sell_win"] = test_labeled["sell_win"].to_numpy()

        fold_trades: list[dict[str, Any]] = []
        if len(valid_local_indices):
            x_test = test_slice.iloc[valid_local_indices][FEATURE_COLUMNS]
            buy_raw = buy_model.predict_proba(x_test)[:, 1]
            sell_raw = sell_model.predict_proba(x_test)[:, 1]

            method_probabilities: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for method in ("raw", "sigmoid", "isotonic"):
                buy_p = buy_calibrators[method].transform(buy_raw)
                sell_p = sell_calibrators[method].transform(sell_raw)
                method_probabilities[method] = (buy_p, sell_p)
                test_export.loc[valid_local_indices, f"buy_probability_{method}"] = buy_p
                test_export.loc[valid_local_indices, f"sell_probability_{method}"] = sell_p

            selected_buy, selected_sell = method_probabilities[calibration_method]
            probs = {
                int(local_idx): (float(b), float(s))
                for local_idx, b, s in zip(valid_local_indices, selected_buy, selected_sell)
            }

            local_i = int(valid_local_indices[0])
            last_local = int(valid_local_indices[-1])
            while local_i <= last_local:
                pair = probs.get(local_i)
                if pair is None:
                    local_i += 1
                    continue

                buy_p, sell_p = pair
                side = _decision(buy_p, sell_p, backtest_config)
                if side == "HOLD":
                    local_i += 1
                    continue

                trade, exit_local = _simulate_trade(
                    df=test_slice,
                    signal_index=local_i,
                    side=side,
                    buy_probability=buy_p,
                    sell_probability=sell_p,
                    cfg=backtest_config,
                    balance=balance,
                )
                row = asdict(trade)
                row["fold"] = fold_id
                row["calibration_method"] = calibration_method
                fold_trades.append(row)
                all_trades.append(row)
                balance = trade.balance_after
                local_i = exit_local + 1

        all_predictions.append(test_export)

        fold_df = pd.DataFrame(fold_trades)
        fold_initial = (
            balance
            if fold_df.empty
            else float(fold_df.iloc[0]["balance_after"] - fold_df.iloc[0]["net_pnl"])
        )
        fold_summary = _summary(fold_df, fold_initial)
        fold_rows.append(
            {
                "fold": fold_id,
                "train_start": str(featured.iloc[train_start]["time"]),
                "train_end": str(featured.iloc[train_end - 1]["time"]),
                "calibration_start": str(featured.iloc[calibration_start]["time"]),
                "calibration_end": str(featured.iloc[calibration_end - 1]["time"]),
                "purge_bars": purge_bars,
                "test_start": str(featured.iloc[test_start]["time"]),
                "test_end": str(featured.iloc[test_end - 1]["time"]),
                "train_samples": int(len(train_slice)),
                "calibration_samples": int(len(calibration_slice)),
                "test_bars": int(len(test_slice)),
                "calibration_method": calibration_method,
                "trades": int(len(fold_df)),
                "net_profit": float(fold_df["net_pnl"].sum()) if not fold_df.empty else 0.0,
                "win_rate": float((fold_df["net_pnl"] > 0).mean()) if not fold_df.empty else 0.0,
                "profit_factor": fold_summary.get("profit_factor"),
                "ending_balance": float(balance),
            }
        )

        test_start += step_bars

    trades_df = pd.DataFrame(all_trades)
    folds_df = pd.DataFrame(fold_rows)
    predictions_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()

    if trades_df.empty:
        equity_df = pd.DataFrame(
            [{"time": str(featured.iloc[first_test_start]["time"]), "equity": backtest_config.initial_balance}]
        )
    else:
        equity_df = pd.DataFrame(
            {"time": trades_df["exit_time"], "equity": trades_df["balance_after"]}
        )

    summary = _summary(trades_df, backtest_config.initial_balance)
    summary.update(
        {
            "walk_forward_folds": int(len(folds_df)),
            "train_bars_per_fold": int(train_bars),
            "calibration_bars_per_fold": int(calibration_bars),
            "test_bars_per_fold": int(test_bars),
            "step_bars": int(step_bars),
            "purge_bars": int(purge_bars),
            "calibration_method": calibration_method,
            "out_of_sample_only": True,
        }
    )

    return trades_df, equity_df, folds_df, predictions_df, summary
