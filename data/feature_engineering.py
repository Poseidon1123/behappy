from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "ema_diff",
    "atr_ratio",
    "volatility20",
    "momentum3",
    "momentum5",
    "momentum10",
    "rsi14",
    "range_pos20",
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
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create causal features using only current/past candles."""
    required = {"time", "open", "high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")

    out = df.copy().sort_values("time").reset_index(drop=True)
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    open_ = out["open"].astype(float)

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    out["ema_diff"] = (ema20 - ema50) / close.replace(0.0, np.nan)

    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = true_range.rolling(14).mean()
    atr50 = true_range.rolling(50).mean()
    out["atr_ratio"] = atr14 / atr50.replace(0.0, np.nan)

    returns = close.pct_change()
    out["volatility20"] = returns.rolling(20).std()
    out["momentum3"] = close.pct_change(3)
    out["momentum5"] = close.pct_change(5)
    out["momentum10"] = close.pct_change(10)
    out["rsi14"] = _rsi(close, 14) / 100.0

    high20 = high.rolling(20).max()
    low20 = low.rolling(20).min()
    range20 = (high20 - low20).replace(0.0, np.nan)
    out["range_pos20"] = (close - low20) / range20

    candle_range = (high - low).replace(0.0, np.nan)
    body_high = pd.concat([open_, close], axis=1).max(axis=1)
    body_low = pd.concat([open_, close], axis=1).min(axis=1)
    out["body_ratio"] = (close - open_).abs() / candle_range
    out["upper_wick_ratio"] = (high - body_high) / candle_range
    out["lower_wick_ratio"] = (body_low - low) / candle_range

    time = pd.to_datetime(out["time"], utc=True)
    hour = time.dt.hour + time.dt.minute / 60.0
    out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    # UTC session approximations; broker/server timezone does not affect these features.
    out["london_session"] = ((hour >= 7.0) & (hour < 16.0)).astype(int)
    out["newyork_session"] = ((hour >= 12.0) & (hour < 21.0)).astype(int)

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    features = build_features(df)
    return features[FEATURE_COLUMNS]
