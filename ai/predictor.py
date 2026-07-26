from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from ai.features import FEATURE_COLUMNS, add_features
from ai.labeling import LABEL_BUY, LABEL_HOLD, LABEL_NAMES, LABEL_SELL


class ModelNotReadyError(RuntimeError):
    pass


@dataclass(frozen=True)
class Prediction:
    buy: float
    hold: float
    sell: float
    decision: str


class AIPredictor:
    def __init__(self, model_path: str = "models/xauusd_m15_model.joblib") -> None:
        self.model_path = Path(model_path)
        self.bundle: dict | None = None
        self.load_if_available()

    @property
    def ready(self) -> bool:
        return self.bundle is not None

    def load_if_available(self) -> bool:
        if not self.model_path.exists():
            self.bundle = None
            return False

        bundle = joblib.load(self.model_path)
        required = {"model", "feature_columns", "classes"}
        if not required.issubset(bundle):
            raise ModelNotReadyError(
                f"Invalid model bundle at {self.model_path}. Missing keys: {required - set(bundle)}"
            )

        self.bundle = bundle
        return True

    def predict(self, candles: pd.DataFrame) -> Prediction:
        if not self.ready or self.bundle is None:
            raise ModelNotReadyError(
                f"AI model not found. Run: python train_ai.py (expected {self.model_path})"
            )

        featured = add_features(candles).dropna(subset=FEATURE_COLUMNS)
        if featured.empty:
            raise ModelNotReadyError("Not enough candles to calculate AI features.")

        columns = list(self.bundle["feature_columns"])
        x = featured.iloc[[-1]][columns]
        model = self.bundle["model"]
        probabilities = model.predict_proba(x)[0]
        classes = [int(value) for value in model.classes_]
        prob_by_class = dict(zip(classes, probabilities))

        buy = float(prob_by_class.get(LABEL_BUY, 0.0))
        hold = float(prob_by_class.get(LABEL_HOLD, 0.0))
        sell = float(prob_by_class.get(LABEL_SELL, 0.0))

        decision_label = int(model.predict(x)[0])
        decision = LABEL_NAMES.get(decision_label, "HOLD")
        return Prediction(buy=buy, hold=hold, sell=sell, decision=decision)

    def metadata(self) -> dict:
        if self.bundle is None:
            return {}
        return dict(self.bundle.get("metadata", {}))
