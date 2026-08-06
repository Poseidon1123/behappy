from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

CalibrationMethod = Literal["raw", "sigmoid", "isotonic"]


@dataclass
class ProbabilityCalibrator:
    method: CalibrationMethod
    model: object | None = None

    def fit(self, probabilities: np.ndarray, y_true: np.ndarray) -> "ProbabilityCalibrator":
        p = np.asarray(probabilities, dtype=float)
        y = np.asarray(y_true, dtype=int)

        mask = np.isfinite(p)
        p = p[mask]
        y = y[mask]

        if len(p) < 50 or len(np.unique(y)) < 2:
            raise ValueError("Calibration needs at least 50 samples and both classes.")

        if self.method == "raw":
            self.model = None
        elif self.method == "sigmoid":
            model = LogisticRegression(solver="lbfgs")
            model.fit(p.reshape(-1, 1), y)
            self.model = model
        elif self.method == "isotonic":
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(p, y)
            self.model = model
        else:
            raise ValueError(f"Unsupported calibration method: {self.method}")

        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        p = np.asarray(probabilities, dtype=float)

        if self.method == "raw":
            return np.clip(p, 0.0, 1.0)
        if self.model is None:
            raise RuntimeError("Calibrator has not been fitted.")
        if self.method == "sigmoid":
            return self.model.predict_proba(p.reshape(-1, 1))[:, 1]
        if self.method == "isotonic":
            return np.asarray(self.model.predict(p), dtype=float)
        raise ValueError(f"Unsupported calibration method: {self.method}")


def calibration_metrics(
    y_true: pd.Series | np.ndarray,
    probability: pd.Series | np.ndarray,
    bins: int = 10,
) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probability, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask].astype(int)
    p = np.clip(p[mask], 0.0, 1.0)

    if len(y) == 0:
        return {
            "samples": 0,
            "brier_score": float("nan"),
            "ece": float("nan"),
        }

    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_id = np.clip(np.digitize(p, edges, right=True) - 1, 0, bins - 1)

    ece = 0.0
    for idx in range(bins):
        m = bin_id == idx
        if not np.any(m):
            continue
        ece += (np.sum(m) / len(y)) * abs(float(np.mean(p[m])) - float(np.mean(y[m])))

    return {
        "samples": int(len(y)),
        "brier_score": float(brier_score_loss(y, p)),
        "ece": float(ece),
    }
