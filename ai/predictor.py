from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from data.feature_engineering import FEATURE_COLUMNS, build_features


class AIPredictor:
    def __init__(self, model_dir: str = "models") -> None:
        self.model_dir = Path(model_dir)
        self.buy_model = None
        self.sell_model = None
        self.meta = None
        self.reload()

    @property
    def ready(self) -> bool:
        return self.buy_model is not None and self.sell_model is not None

    def reload(self) -> bool:
        buy_path = self.model_dir / "buy_model.joblib"
        sell_path = self.model_dir / "sell_model.joblib"
        meta_path = self.model_dir / "model_meta.joblib"

        if not buy_path.exists() or not sell_path.exists():
            self.buy_model = None
            self.sell_model = None
            self.meta = None
            return False

        self.buy_model = joblib.load(buy_path)
        self.sell_model = joblib.load(sell_path)
        self.meta = joblib.load(meta_path) if meta_path.exists() else None
        return True

    def predict(self, raw_df: pd.DataFrame) -> dict[str, float | str]:
        if not self.ready:
            return {
                "buy_probability": 0.0,
                "sell_probability": 0.0,
                "hold_probability": 1.0,
                "status": "MODEL_NOT_TRAINED",
            }

        featured = build_features(raw_df)
        latest = featured[FEATURE_COLUMNS].dropna().tail(1)
        if latest.empty:
            return {
                "buy_probability": 0.0,
                "sell_probability": 0.0,
                "hold_probability": 1.0,
                "status": "NOT_ENOUGH_DATA",
            }

        # These are independent probabilities: P(BUY reaches TP before SL)
        # and P(SELL reaches TP before SL), based on the separately trained models.
        buy = float(self.buy_model.predict_proba(latest)[:, 1][0])
        sell = float(self.sell_model.predict_proba(latest)[:, 1][0])

        # HOLD is a display/decision aid, not a third trained class.
        edge = max(buy, sell)
        conflict = min(buy, sell)
        hold = float(np.clip((1.0 - edge) + 0.25 * conflict, 0.0, 1.0))

        return {
            "buy_probability": buy,
            "sell_probability": sell,
            "hold_probability": hold,
            "status": "READY",
        }
