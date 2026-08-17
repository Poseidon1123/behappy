from __future__ import annotations

import json
from pathlib import Path

import yaml

from ai.train_models import train_buy_sell_models
from mt5.market_data import MarketData
from mt5.mt5_connector import MT5Connector


def load_config(path: str = "config/config.yaml") -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    cfg = load_config()
    trading = cfg.get("trading", {})
    training = cfg.get("ai_training", {})

    symbol = str(trading.get("symbol", "XAUUSD.sc"))
    timeframe = str(trading.get("timeframe", "M15"))
    bars = int(training.get("bars", 50000))
    horizon = int(training.get("horizon", 8))
    tp = float(training.get("take_profit_pct", 0.006))
    sl = float(training.get("stop_loss_pct", 0.003))
    validation_fraction = float(training.get("validation_fraction", 0.20))

    print(f"Collecting {bars} closed bars: {symbol} {timeframe}")

    with MT5Connector():
        market = MarketData()
        df = market.get_bars(
            symbol=symbol,
            timeframe=timeframe,
            count=bars,
            include_current_bar=False,
        )

    print(f"Received {len(df)} bars from MT5")

    data_dir = Path("training_data")
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / f"{symbol.replace('.', '_')}_{timeframe}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved raw training data: {csv_path}")

    report = train_buy_sell_models(
        raw_df=df,
        horizon=horizon,
        take_profit_pct=tp,
        stop_loss_pct=sl,
        validation_fraction=validation_fraction,
    )

    print("\nTraining completed. Validation report:")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("\nModels saved to models/buy_model.joblib and models/sell_model.joblib")
    print("Restart the HMI with: python main.py")


if __name__ == "__main__":
    main()
