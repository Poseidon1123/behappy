from __future__ import annotations

import numpy as np
import pandas as pd

LABEL_SELL = 0
LABEL_HOLD = 1
LABEL_BUY = 2
LABEL_NAMES = {LABEL_SELL: "SELL", LABEL_HOLD: "HOLD", LABEL_BUY: "BUY"}


def add_labels(
    df: pd.DataFrame,
    horizon_bars: int = 8,
    atr_multiplier: float = 0.8,
) -> pd.DataFrame:
    """Create BUY/HOLD/SELL labels from future return scaled by current ATR.

    BUY: future close is above current close by at least atr_multiplier * ATR.
    SELL: future close is below current close by at least the same distance.
    Otherwise HOLD.
    """
    if horizon_bars <= 0:
        raise ValueError("horizon_bars must be greater than zero")
    if atr_multiplier <= 0:
        raise ValueError("atr_multiplier must be greater than zero")

    out = df.copy()
    future_close = out["close"].shift(-horizon_bars)
    future_move = future_close - out["close"]
    threshold = out["atr_14"] * float(atr_multiplier)

    out["future_close"] = future_close
    out["future_move"] = future_move
    out["label"] = np.select(
        [future_move <= -threshold, future_move >= threshold],
        [LABEL_SELL, LABEL_BUY],
        default=LABEL_HOLD,
    ).astype(int)

    # Last horizon rows do not have complete future information.
    return out.iloc[:-horizon_bars].copy()
