# AI MT5 Trading Bot

Python foundation for an AI-assisted MetaTrader 5 trading system.

## Implemented

### Step 1 - MetaTrader 5 connection

- Connect to a running MetaTrader 5 terminal.
- Read account information and live market data.
- Download closed OHLC bars.
- Keep credentials outside source control.

### Step 2 - PySide6 HMI

- Dark desktop dashboard built with PySide6.
- Realtime price chart with pyqtgraph.
- Balance, equity, floating P/L, bid, ask and spread.
- Open-position table and editable risk controls.
- Auto refresh every 2 seconds.

### Step 3 - Real AI model

- Downloads XAUUSD M15 closed candles directly from MT5.
- Causal feature engineering: returns, EMA spread, ATR, volatility, momentum, RSI, candle shape and session/time features.
- Creates SELL / HOLD / BUY labels from the future price move scaled by the current ATR.
- Uses a chronological train/test split to reduce look-ahead leakage.
- Trains a scikit-learn `HistGradientBoostingClassifier` with class-balanced sample weights.
- Saves the trained model locally as `models/xauusd_m15_model.joblib`.
- HMI uses the model's real `predict_proba()` output for BUY / HOLD / SELL probabilities.
- The model file is intentionally ignored by Git because it is generated from your local MT5 history.

> Order execution is still intentionally disabled. The project currently monitors markets and produces AI signals only.

## Requirements

- Windows with MetaTrader 5 desktop installed.
- Python 3.10+ recommended.
- MT5 terminal open and logged in.
- Enough XAUUSD M15 history available in MetaTrader 5.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `config/config.yaml` so `trading.symbol` exactly matches the broker symbol shown in MT5 Market Watch, for example `XAUUSD.sc`.

## Train the AI model

Open MetaTrader 5, make sure XAUUSD M15 history is available, then run:

```bash
python train_ai.py
```

Default training settings are in `config/config.yaml`:

```yaml
ai:
  training_bars: 50000
  horizon_bars: 8
  label_atr_multiplier: 0.8
  test_fraction: 0.20
```

The trainer prints class distribution, test accuracy, classification report and confusion matrix, then saves:

```text
models/xauusd_m15_model.joblib
```

## Run the HMI

```bash
python main.py
```

When the model exists, the HMI displays `REAL AI MODEL LOADED` and uses real probabilities from the trained classifier. If the model is missing, it displays `MODEL NOT TRAINED` and does not generate fake probabilities.

## Safety

The current project does not send, modify or close orders. Validate the model with out-of-sample testing and demo trading before any future live-order module is enabled.
