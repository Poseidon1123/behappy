from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import MetaTrader5 as mt5
import pandas as pd


class MarketDataError(RuntimeError):
    pass


TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}


class MarketData:
    def ensure_symbol(self, symbol: str) -> None:
        info = mt5.symbol_info(symbol)

        if info is None:
            raise MarketDataError(
                f"Symbol '{symbol}' was not found. "
                "Use the exact broker symbol shown in MT5 Market Watch."
            )

        if not info.visible and not mt5.symbol_select(symbol, True):
            raise MarketDataError(
                f"Could not select '{symbol}'. MT5 error: {mt5.last_error()}"
            )

    def get_tick(self, symbol: str) -> dict[str, Any]:
        self.ensure_symbol(symbol)

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise MarketDataError(
                f"Could not read tick for '{symbol}'. "
                f"MT5 error: {mt5.last_error()}"
            )

        data = tick._asdict()
        data["datetime_utc"] = datetime.fromtimestamp(tick.time, tz=timezone.utc)
        data["spread_price"] = float(tick.ask) - float(tick.bid)
        return data

    def get_bars(
        self,
        symbol: str,
        timeframe: str = "M15",
        count: int = 100,
        include_current_bar: bool = False,
    ) -> pd.DataFrame:
        self.ensure_symbol(symbol)

        tf = timeframe.upper()
        if tf not in TIMEFRAMES:
            raise MarketDataError(
                f"Unsupported timeframe '{timeframe}'. "
                f"Supported: {', '.join(TIMEFRAMES)}"
            )

        if count <= 0:
            raise ValueError("count must be greater than zero.")

        # MT5 bar position 0 is the currently forming candle.
        # Default to position 1 so AI will later work on closed candles only.
        start_pos = 0 if include_current_bar else 1

        rates = mt5.copy_rates_from_pos(
            symbol, TIMEFRAMES[tf], start_pos, count
        )

        if rates is None:
            raise MarketDataError(
                f"Could not download bars for '{symbol}'. "
                f"MT5 error: {mt5.last_error()}"
            )

        if len(rates) == 0:
            raise MarketDataError(f"MT5 returned zero bars for '{symbol}'.")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        return df
