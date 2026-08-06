from __future__ import annotations

import numpy as np
import pandas as pd


def create_tp_sl_labels(
    df: pd.DataFrame,
    horizon: int = 8,
    take_profit_pct: float = 0.006,
    stop_loss_pct: float = 0.003,
) -> pd.DataFrame:
    """Create separate BUY/SELL binary labels using first-touch TP/SL logic.

    A BUY label is 1 when the long TP is touched before the long SL within
    `horizon` future candles. SELL is defined symmetrically.

    If TP and SL are both touched inside the same future OHLC candle, the
    intrabar ordering is unknowable. Those samples are therefore marked NaN
    and excluded from training instead of being forced into the loss class.
    """
    if horizon <= 0:
        raise ValueError("horizon must be greater than zero")
    if take_profit_pct <= 0 or stop_loss_pct <= 0:
        raise ValueError("TP and SL percentages must be greater than zero")

    out = df.copy().reset_index(drop=True)
    close = out["close"].astype(float).to_numpy()
    high = out["high"].astype(float).to_numpy()
    low = out["low"].astype(float).to_numpy()

    buy_label = np.full(len(out), np.nan)
    sell_label = np.full(len(out), np.nan)

    last_trainable = len(out) - horizon
    for i in range(max(0, last_trainable)):
        entry = close[i]
        buy_tp = entry * (1.0 + take_profit_pct)
        buy_sl = entry * (1.0 - stop_loss_pct)
        sell_tp = entry * (1.0 - take_profit_pct)
        sell_sl = entry * (1.0 + stop_loss_pct)

        buy_result: float | None = 0.0
        sell_result: float | None = 0.0
        buy_done = False
        sell_done = False

        for j in range(i + 1, i + horizon + 1):
            h = high[j]
            l = low[j]

            if not buy_done:
                hit_tp = h >= buy_tp
                hit_sl = l <= buy_sl
                if hit_tp and hit_sl:
                    buy_result = np.nan
                    buy_done = True
                elif hit_tp:
                    buy_result = 1.0
                    buy_done = True
                elif hit_sl:
                    buy_result = 0.0
                    buy_done = True

            if not sell_done:
                hit_tp = l <= sell_tp
                hit_sl = h >= sell_sl
                if hit_tp and hit_sl:
                    sell_result = np.nan
                    sell_done = True
                elif hit_tp:
                    sell_result = 1.0
                    sell_done = True
                elif hit_sl:
                    sell_result = 0.0
                    sell_done = True

            if buy_done and sell_done:
                break

        buy_label[i] = buy_result
        sell_label[i] = sell_result

    out["buy_win"] = buy_label
    out["sell_win"] = sell_label
    return out
