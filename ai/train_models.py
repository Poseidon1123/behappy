from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight

from ai.labeling import create_tp_sl_labels
from data.feature_engineering import FEATURE_COLUMNS, build_features


MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")


def _metric_dict(y_true: pd.Series, proba: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    metrics = {
        "samples": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
    else:
        metrics["roc_auc"] = None
    return metrics


def _fit_binary_model(X_train: pd.DataFrame, y_train: pd.Series) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=250,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=42,
    )
    weights = compute_sample_weight(class_weight="balanced", y=y_train)
    model.fit(X_train, y_train, sample_weight=weights)
    return model


def train_buy_sell_models(
    raw_df: pd.DataFrame,
    horizon: int = 8,
    take_profit_pct: float = 0.006,
    stop_loss_pct: float = 0.003,
    validation_fraction: float = 0.20,
) -> dict:
    """Train BUY and SELL models with a chronological train/validation split."""
    if not 0.05 <= validation_fraction <= 0.40:
        raise ValueError("validation_fraction must be between 0.05 and 0.40")

    featured = build_features(raw_df)
    labeled = create_tp_sl_labels(
        featured,
        horizon=horizon,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
    )

    dataset = labeled.dropna(subset=FEATURE_COLUMNS + ["buy_win", "sell_win"]).copy()
    if len(dataset) < 2000:
        raise ValueError(
            f"Only {len(dataset)} usable samples. Collect at least a few thousand M15 bars."
        )

    split_index = int(len(dataset) * (1.0 - validation_fraction))
    train = dataset.iloc[:split_index]
    valid = dataset.iloc[split_index:]

    X_train = train[FEATURE_COLUMNS]
    X_valid = valid[FEATURE_COLUMNS]

    y_buy_train = train["buy_win"].astype(int)
    y_buy_valid = valid["buy_win"].astype(int)
    y_sell_train = train["sell_win"].astype(int)
    y_sell_valid = valid["sell_win"].astype(int)

    if y_buy_train.nunique() < 2 or y_sell_train.nunique() < 2:
        raise ValueError("Training labels contain only one class. Adjust TP/SL/horizon or collect more data.")

    buy_model = _fit_binary_model(X_train, y_buy_train)
    sell_model = _fit_binary_model(X_train, y_sell_train)

    buy_proba = buy_model.predict_proba(X_valid)[:, 1]
    sell_proba = sell_model.predict_proba(X_valid)[:, 1]

    report = {
        "feature_columns": FEATURE_COLUMNS,
        "horizon": horizon,
        "take_profit_pct": take_profit_pct,
        "stop_loss_pct": stop_loss_pct,
        "total_usable_samples": int(len(dataset)),
        "train_samples": int(len(train)),
        "validation_samples": int(len(valid)),
        "train_start": str(train["time"].iloc[0]),
        "train_end": str(train["time"].iloc[-1]),
        "validation_start": str(valid["time"].iloc[0]),
        "validation_end": str(valid["time"].iloc[-1]),
        "buy": _metric_dict(y_buy_valid, buy_proba),
        "sell": _metric_dict(y_sell_valid, sell_proba),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(buy_model, MODEL_DIR / "buy_model.joblib")
    joblib.dump(sell_model, MODEL_DIR / "sell_model.joblib")
    joblib.dump(
        {
            "feature_columns": FEATURE_COLUMNS,
            "horizon": horizon,
            "take_profit_pct": take_profit_pct,
            "stop_loss_pct": stop_loss_pct,
        },
        MODEL_DIR / "model_meta.joblib",
    )

    with (REPORT_DIR / "ai_training_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report
