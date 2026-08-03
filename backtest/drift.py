from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


DRIFT_FEATURES = [
    "ema_diff",
    "atr_ratio",
    "volatility20",
    "momentum3",
    "momentum10",
    "rsi14",
    "spread_relative",
    "h1_ema_diff",
    "h1_adx14",
    "h1_volatility20",
]


@dataclass(frozen=True)
class DriftMap:
    allowed: dict[int, bool]
    scores: dict[int, float]
    cutoff: float
    calibration_scores: tuple[float, ...]


def _psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    expected = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    actual = pd.to_numeric(current, errors="coerce").dropna().to_numpy()
    if len(expected) < 100 or len(actual) < 100:
        return float("nan")

    edges = np.unique(np.quantile(expected, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    expected_pct = np.histogram(expected, bins=edges)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=edges)[0] / len(actual)
    expected_pct = np.clip(expected_pct, 1e-4, None)
    actual_pct = np.clip(actual_pct, 1e-4, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def _drift_score(reference: pd.DataFrame, current: pd.DataFrame) -> float:
    scores = [_psi(reference[col], current[col]) for col in DRIFT_FEATURES]
    finite = [score for score in scores if np.isfinite(score)]
    return float(np.mean(finite)) if finite else float("nan")


def build_causal_drift_map(
    featured: pd.DataFrame,
    *,
    train_start: int,
    train_end: int,
    test_start: int,
    test_end: int,
    recent_window: int = 500,
    score_step: int = 100,
    calibration_step: int = 250,
    cutoff_quantile: float = 0.95,
) -> DriftMap:
    """Build an OOD gate without reading future bars for a decision.

    The cutoff is learned only from rolling windows inside outer training. Test
    scores are refreshed every ``score_step`` bars using data strictly before
    the start of that block; the score is then held constant for the block.
    """
    if not 0.0 < cutoff_quantile < 1.0:
        raise ValueError("cutoff_quantile must be between zero and one")
    if min(recent_window, score_step, calibration_step) <= 0:
        raise ValueError("Drift windows and steps must be positive")
    missing = [col for col in DRIFT_FEATURES if col not in featured.columns]
    if missing:
        raise ValueError(f"Missing drift features: {missing}")

    reference = featured.iloc[train_start:train_end]
    calibration: list[float] = []
    for end in range(recent_window, len(reference) + 1, calibration_step):
        score = _drift_score(reference, reference.iloc[end - recent_window:end])
        if np.isfinite(score):
            calibration.append(score)
    if len(calibration) < 10:
        raise ValueError("Not enough historical windows to calibrate drift cutoff")
    cutoff = float(np.quantile(calibration, cutoff_quantile))

    allowed: dict[int, bool] = {}
    scores: dict[int, float] = {}
    test_length = test_end - test_start
    for block_start in range(0, test_length, score_step):
        absolute = test_start + block_start
        recent_start = absolute - recent_window
        if recent_start < 0:
            score = float("nan")
        else:
            score = _drift_score(reference, featured.iloc[recent_start:absolute])
        block_allowed = bool(np.isfinite(score) and score <= cutoff)
        for local_index in range(block_start, min(block_start + score_step, test_length)):
            allowed[local_index] = block_allowed
            scores[local_index] = score
    return DriftMap(
        allowed=allowed,
        scores=scores,
        cutoff=cutoff,
        calibration_scores=tuple(calibration),
    )
