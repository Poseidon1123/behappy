from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "return_1",
    "return_3",
    "return_5",
    "ema_diff",
    "atr_ratio",
    "volatility_20",
    "momentum_3",
    "momentum_5",
    "momentum_10",
    "rsi_14",
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "hour_sin",
    "hour_cos",
    "london_session",
    "newyork_session",
]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create causal features using only current and past candles."""
    out = df.copy()
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    open_ = out["open"].astype(float)

    out["return_1"] = close.pct_change(1)
    out["return_3"] = close.pct_change(3)
    out["return_5"] = close.pct_change(5)

    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    out["ema_diff"] = (ema_fast - ema_slow) / close.replace(0, np.nan)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(14).mean()
    out["atr_14"] = atr
    out["atr_ratio"] = atr / close.replace(0, np.nan)

    out["volatility_20"] = out["return_1"].rolling(20).std()
    out["momentum_3"] = close / close.shift(3) - 1.0
    out["momentum_5"] = close / close.shift(5) - 1.0
    out["momentum_10"] = close / close.shift(10) - 1.0
    out["rsi_14"] = _rsi(close, 14) / 100.0

    candle_range = (high - low).replace(0, np.nan)
    body_high = pd.concat([open_, close], axis=1).max(axis=1)
    body_low = pd.concat([open_, close], axis=1).min(axis=1)
    out["body_ratio"] = (close - open_).abs() / candle_range
    out["upper_wick_ratio"] = (high - body_high) / candle_range
    out["lower_wick_ratio"] = (body_low - low) / candle_range

    timestamp = pd.to_datetime(out["time"], utc=True)
    hour = timestamp.dt.hour + timestamp.dt.minute / 60.0
    out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)

    # Approximate UTC sessions. They are features, not trading rules.
    out["london_session"] = ((timestamp.dt.hour >= 7) & (timestamp.dt.hour < 16)).astype(int)
    out["newyork_session"] = ((timestamp.dt.hour >= 12) & (timestamp.dt.hour < 21)).astype(int)

    return out


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    featured = add_features(df)
    return featured.dropna(subset=FEATURE_COLUMNS).copy()
