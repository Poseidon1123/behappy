from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

from ai.features import FEATURE_COLUMNS, add_features
from ai.labeling import LABEL_NAMES, add_labels
from mt5.market_data import MarketData
from mt5.mt5_connector import MT5Connector


def load_config(path: str = "config/config.yaml") -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def main() -> None:
    config = load_config()
    trading = config.get("trading", {})
    ai_cfg = config.get("ai", {})

    symbol = str(trading.get("symbol", "XAUUSD.sc"))
    timeframe = str(trading.get("timeframe", "M15"))
    training_bars = int(ai_cfg.get("training_bars", 50000))
    horizon_bars = int(ai_cfg.get("horizon_bars", 8))
    atr_multiplier = float(ai_cfg.get("label_atr_multiplier", 0.8))
    test_fraction = float(ai_cfg.get("test_fraction", 0.20))
    model_path = Path(ai_cfg.get("model_path", "models/xauusd_m15_model.joblib"))

    print(f"Connecting to MT5 and downloading {training_bars:,} {timeframe} bars for {symbol}...")
    with MT5Connector():
        candles = MarketData().get_bars(
            symbol=symbol,
            timeframe=timeframe,
            count=training_bars,
            include_current_bar=False,
        )

    print(f"Received {len(candles):,} closed candles")
    dataset = add_features(candles)
    dataset = add_labels(
        dataset,
        horizon_bars=horizon_bars,
        atr_multiplier=atr_multiplier,
    )
    dataset = dataset.dropna(subset=FEATURE_COLUMNS + ["label"]).reset_index(drop=True)

    if len(dataset) < 1000:
        raise RuntimeError(
            f"Only {len(dataset)} usable samples. Increase MT5 history before training."
        )

    split = int(len(dataset) * (1.0 - test_fraction))
    if split <= 0 or split >= len(dataset):
        raise ValueError("ai.test_fraction must leave both train and test samples")

    train = dataset.iloc[:split]
    test = dataset.iloc[split:]
    x_train = train[FEATURE_COLUMNS]
    y_train = train["label"].astype(int)
    x_test = test[FEATURE_COLUMNS]
    y_test = test["label"].astype(int)

    print("\nClass distribution:")
    for label, count in dataset["label"].value_counts().sort_index().items():
        print(f"  {LABEL_NAMES[int(label)]:4s}: {count:,} ({count / len(dataset):.1%})")

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    model = HistGradientBoostingClassifier(
        learning_rate=float(ai_cfg.get("learning_rate", 0.06)),
        max_iter=int(ai_cfg.get("max_iter", 300)),
        max_leaf_nodes=int(ai_cfg.get("max_leaf_nodes", 31)),
        l2_regularization=float(ai_cfg.get("l2_regularization", 1.0)),
        random_state=42,
    )

    print(f"\nTraining on {len(train):,} samples; testing on {len(test):,} future samples...")
    model.fit(x_train, y_train, sample_weight=sample_weight)

    predictions = model.predict(x_test)
    accuracy = float(accuracy_score(y_test, predictions))
    labels = sorted(int(x) for x in np.unique(np.concatenate([y_test, predictions])))
    names = [LABEL_NAMES[x] for x in labels]

    print(f"\nTest accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, labels=labels, target_names=names, zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, predictions, labels=labels))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "classes": [int(x) for x in model.classes_],
        "metadata": {
            "symbol": symbol,
            "timeframe": timeframe,
            "training_bars_requested": training_bars,
            "samples": len(dataset),
            "train_samples": len(train),
            "test_samples": len(test),
            "horizon_bars": horizon_bars,
            "label_atr_multiplier": atr_multiplier,
            "test_accuracy": accuracy,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "train_start": str(train.iloc[0]["time"]),
            "train_end": str(train.iloc[-1]["time"]),
            "test_start": str(test.iloc[0]["time"]),
            "test_end": str(test.iloc[-1]["time"]),
        },
    }
    joblib.dump(bundle, model_path)
    print(f"\nSaved real AI model to: {model_path}")
    print("Restart python main.py so the HMI loads the new model.")


if __name__ == "__main__":
    main()
