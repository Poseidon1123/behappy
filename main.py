from __future__ import annotations

from pathlib import Path

import yaml

from mt5.market_data import MarketData, MarketDataError
from mt5.mt5_connector import MT5ConnectionError, MT5Connector
from utils.logger import setup_logger


def load_config(path: str = "config/config.yaml") -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    config = load_config()

    trading = config.get("trading", {})
    app = config.get("application", {})

    symbol = str(trading.get("symbol", "XAUUSD.sc"))
    timeframe = str(trading.get("timeframe", "M15"))
    bars = int(trading.get("bars", 100))

    logger = setup_logger(level=str(app.get("log_level", "INFO")))

    logger.info("Starting AI MT5 Trading Bot - Step 1 (READ ONLY)")
    logger.info("symbol=%s timeframe=%s bars=%s", symbol, timeframe, bars)

    try:
        with MT5Connector() as connector:
            logger.info("MetaTrader 5 connection: OK")

            account = connector.account_snapshot()
            logger.info(
                "Account | login=%s server=%s currency=%s "
                "balance=%.2f equity=%.2f profit=%.2f",
                account.login,
                account.server,
                account.currency,
                account.balance,
                account.equity,
                account.profit,
            )

            market = MarketData()

            tick = market.get_tick(symbol)
            logger.info(
                "%s | bid=%s ask=%s spread=%s",
                symbol,
                tick["bid"],
                tick["ask"],
                tick["spread_price"],
            )

            df = market.get_bars(
                symbol=symbol,
                timeframe=timeframe,
                count=bars,
                include_current_bar=False,
            )

            print("\nLatest 5 CLOSED bars:")
            print(
                df[
                    ["time", "open", "high", "low", "close",
                     "tick_volume", "spread"]
                ].tail(5).to_string(index=False)
            )

            logger.info("Downloaded %d closed bars.", len(df))
            logger.info("Step 1 completed successfully.")

    except (MT5ConnectionError, MarketDataError, FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        logger.warning("Stopped by user.")


if __name__ == "__main__":
    main()
