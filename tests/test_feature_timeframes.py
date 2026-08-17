from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from data.feature_engineering import build_features


class TimeframeFeatureTests(unittest.TestCase):
    def _bars(self, periods: int, frequency: str) -> pd.DataFrame:
        index = pd.date_range("2025-01-01", periods=periods, freq=frequency, tz="UTC")
        trend = 2500.0 + np.linspace(0.0, 100.0, periods)
        wave = np.sin(np.arange(periods) / 17.0)
        close = trend + wave
        return pd.DataFrame(
            {
                "time": index,
                "open": close - 0.1,
                "high": close + 0.8,
                "low": close - 0.8,
                "close": close,
                "spread": np.full(periods, 25.0),
            }
        )

    def test_m1_has_ready_higher_timeframe_context(self) -> None:
        featured = build_features(self._bars(6000, "1min"), base_timeframe="M1")
        self.assertTrue(featured.iloc[-1][["h1_ema_diff", "h1_adx14", "h1_volatility20"]].notna().all())

    def test_all_supported_timeframes_build(self) -> None:
        cases = {"M5": "5min", "M15": "15min", "M30": "30min", "H1": "1h", "H4": "4h"}
        for timeframe, frequency in cases.items():
            with self.subTest(timeframe=timeframe):
                featured = build_features(self._bars(1500, frequency), base_timeframe=timeframe)
                self.assertTrue(featured.iloc[-1][["ema_diff", "atr14_abs", "h1_ema_diff"]].notna().all())


if __name__ == "__main__":
    unittest.main()
