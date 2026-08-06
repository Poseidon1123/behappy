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

    def get_bars_chunked(
        self,
        symbol: str,
        timeframe: str = "M15",
        count: int = 100_000,
        include_current_bar: bool = False,
        chunk_size: int = 10_000,
        require_full_count: bool = True,
    ) -> pd.DataFrame:
        """Download a large history using MT5-safe positional chunks.

        Some terminals reject a single large ``copy_rates_from_pos`` request
        with ``(-2, 'Terminal: Invalid params')``. Smaller requests are joined,
        sorted and de-duplicated here. By default a partial history is rejected
        so a supposedly frozen snapshot cannot silently contain fewer bars than
        its filename/configuration claims.
        """
        self.ensure_symbol(symbol)
        tf = timeframe.upper()
        if tf not in TIMEFRAMES:
            raise MarketDataError(
                f"Unsupported timeframe '{timeframe}'. "
                f"Supported: {', '.join(TIMEFRAMES)}"
            )
        if count <= 0:
            raise ValueError("count must be greater than zero.")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        next_position = 0 if include_current_bar else 1
        remaining = count
        chunks: list[pd.DataFrame] = []
        last_error: Any = None

        while remaining > 0:
            requested = min(chunk_size, remaining)
            rates = mt5.copy_rates_from_pos(
                symbol,
                TIMEFRAMES[tf],
                next_position,
                requested,
            )
            if rates is None:
                last_error = mt5.last_error()
                break
            if len(rates) == 0:
                break

            chunk = pd.DataFrame(rates)
            chunk["time"] = pd.to_datetime(chunk["time"], unit="s", utc=True)
            chunks.append(chunk)
            received = len(chunk)
            next_position += received
            remaining -= received
            if received < requested:
                break

        if not chunks:
            raise MarketDataError(
                f"Could not download bars for '{symbol}' in chunks. "
                f"MT5 error: {last_error or mt5.last_error()}"
            )

        result = pd.concat(chunks, ignore_index=True)
        result.sort_values("time", inplace=True)
        result.drop_duplicates(subset=["time"], keep="last", inplace=True)
        result.reset_index(drop=True, inplace=True)

        if require_full_count and len(result) < count:
            raise MarketDataError(
                f"Requested {count:,} closed bars for '{symbol}' {tf}, but MT5 "
                f"provided only {len(result):,}. In MT5 open Tools > Options > "
                "Charts, increase 'Max bars in chart', restart MT5, open the "
                f"{symbol} {tf} chart and scroll/load older history, then retry. "
                f"Last MT5 error: {last_error or mt5.last_error()}"
            )
        if len(result) > count:
            result = result.tail(count).reset_index(drop=True)
        return result
