from __future__ import annotations

import numpy as np
import pandas as pd


M15_COMMON_FEATURES = [
    "ema_diff",
    "atr_ratio",
    "volatility20",
    "momentum3",
    "momentum5",
    "momentum10",
    "rsi14",
    "range_pos20",
    "body_ratio",
    "hour_sin",
    "hour_cos",
    "london_session",
    "newyork_session",
    "spread_atr_ratio",
]

H1_CONTEXT_FEATURES = [
    "h1_ema_diff",
    "h1_ema_slope",
    "h1_rsi14",
    "h1_atr_ratio",
    "h1_adx14",
    "h1_volatility20",
    "h1_range_pos20",
    "regime_trending",
    "regime_ranging",
    "regime_high_vol",
    "regime_low_vol",
]

# BUY and SELL intentionally use different candle/asymmetric context features.
BUY_FEATURE_COLUMNS = M15_COMMON_FEATURES + H1_CONTEXT_FEATURES + [
    "lower_wick_ratio",
    "bullish_body",
    "h1_trend_up",
    "m15_h1_bull_alignment",
    "distance_h1_ema20",
]

SELL_FEATURE_COLUMNS = M15_COMMON_FEATURES + H1_CONTEXT_FEATURES + [
    "upper_wick_ratio",
    "bearish_body",
    "h1_trend_down",
    "m15_h1_bear_alignment",
    "distance_h1_ema20",
]

# Union retained for generic validation/dropna operations.
FEATURE_COLUMNS = list(dict.fromkeys(BUY_FEATURE_COLUMNS + SELL_FEATURE_COLUMNS))


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = _true_range(high, low, close)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr.replace(0.0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / 100.0


def _build_h1_context(m15: pd.DataFrame) -> pd.DataFrame:
    """Aggregate closed M15 candles into completed H1 candles causally.

    H1 bars are labelled at their right edge. merge_asof therefore gives an M15
    row only the latest H1 candle whose closing timestamp is not later than the
    M15 timestamp. This avoids using the still-forming H1 candle.
    """
    base = m15[["time", "open", "high", "low", "close"]].copy()
    base["time"] = pd.to_datetime(base["time"], utc=True)
    base = base.set_index("time")

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    h1 = base.resample("1h", label="right", closed="right").agg(agg).dropna().reset_index()

    close = h1["close"].astype(float)
    high = h1["high"].astype(float)
    low = h1["low"].astype(float)

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    tr = _true_range(high, low, close)
    atr14 = tr.rolling(14).mean()
    atr50 = tr.rolling(50).mean()

    h1["h1_ema_diff"] = (ema20 - ema50) / close.replace(0.0, np.nan)
    h1["h1_ema_slope"] = ema20.pct_change(3)
    h1["h1_rsi14"] = _rsi(close, 14) / 100.0
    h1["h1_atr_ratio"] = atr14 / atr50.replace(0.0, np.nan)
    h1["h1_adx14"] = _adx(high, low, close, 14)
    h1["h1_volatility20"] = close.pct_change().rolling(20).std()

    high20 = high.rolling(20).max()
    low20 = low.rolling(20).min()
    h1["h1_range_pos20"] = (close - low20) / (high20 - low20).replace(0.0, np.nan)
    h1["h1_trend_up"] = ((ema20 > ema50) & (h1["h1_ema_slope"] > 0)).astype(float)
    h1["h1_trend_down"] = ((ema20 < ema50) & (h1["h1_ema_slope"] < 0)).astype(float)
    h1["distance_h1_ema20"] = (close - ema20) / close.replace(0.0, np.nan)

    # Regime flags are based only on completed H1 information.
    h1["regime_trending"] = (h1["h1_adx14"] >= 0.25).astype(float)
    h1["regime_ranging"] = (h1["h1_adx14"] < 0.20).astype(float)
    h1["regime_high_vol"] = (h1["h1_atr_ratio"] >= 1.10).astype(float)
    h1["regime_low_vol"] = (h1["h1_atr_ratio"] <= 0.90).astype(float)

    keep = [
        "time",
        "h1_ema_diff",
        "h1_ema_slope",
        "h1_rsi14",
        "h1_atr_ratio",
        "h1_adx14",
        "h1_volatility20",
        "h1_range_pos20",
        "h1_trend_up",
        "h1_trend_down",
        "distance_h1_ema20",
        "regime_trending",
        "regime_ranging",
        "regime_high_vol",
        "regime_low_vol",
    ]
    return h1[keep]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create causal M15 + completed-H1 + market-regime features."""
    required = {"time", "open", "high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")

    out = df.copy().sort_values("time").reset_index(drop=True)
    out["time"] = pd.to_datetime(out["time"], utc=True)
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    open_ = out["open"].astype(float)

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    out["ema_diff"] = (ema20 - ema50) / close.replace(0.0, np.nan)

    true_range = _true_range(high, low, close)
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
    out["bullish_body"] = (close > open_).astype(float)
    out["bearish_body"] = (close < open_).astype(float)

    time = pd.to_datetime(out["time"], utc=True)
    hour = time.dt.hour + time.dt.minute / 60.0
    out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    out["london_session"] = ((hour >= 7.0) & (hour < 16.0)).astype(int)
    out["newyork_session"] = ((hour >= 12.0) & (hour < 21.0)).astype(int)

    if "spread" in out.columns:
        spread_price = out["spread"].astype(float)
        # MT5 historical spread is expressed in points. Normalizing by ATR gives
        # a causal proxy even before broker point-size metadata is available here.
        out["spread_atr_ratio"] = spread_price / atr14.replace(0.0, np.nan)
    else:
        out["spread_atr_ratio"] = 0.0

    h1 = _build_h1_context(out)
    out = pd.merge_asof(
        out.sort_values("time"),
        h1.sort_values("time"),
        on="time",
        direction="backward",
        allow_exact_matches=True,
    )

    out["m15_h1_bull_alignment"] = (
        (out["ema_diff"] > 0) & (out["h1_trend_up"] > 0)
    ).astype(float)
    out["m15_h1_bear_alignment"] = (
        (out["ema_diff"] < 0) & (out["h1_trend_down"] > 0)
    ).astype(float)

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


def feature_matrix(df: pd.DataFrame, side: str = "BUY") -> pd.DataFrame:
    features = build_features(df)
    columns = BUY_FEATURE_COLUMNS if side.upper() == "BUY" else SELL_FEATURE_COLUMNS
    return features[columns]
