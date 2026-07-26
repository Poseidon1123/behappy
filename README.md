# AI MT5 Trading Bot

Python foundation for an AI-assisted MetaTrader 5 trading system.

## Implemented

### Step 1 - MetaTrader 5 connection

- Connect to a running MetaTrader 5 terminal.
- Read account information and market ticks.
- Download closed OHLC bars.
- Keep credentials outside source control.

### Step 2 - PySide6 HMI

- Dark desktop dashboard built with PySide6.
- Realtime price chart with pyqtgraph.
- Balance, equity, floating P/L, bid, ask and spread.
- Open positions table.
- Risk settings and BUY/SELL thresholds.
- System log and monitoring controls.

### Step 3 - Real AI pipeline

- Collect XAUUSD M15 closed candles directly from MT5.
- Causal feature engineering: EMA spread, ATR ratio, volatility, momentum, RSI, candle shape, range position and session/time features.
- Separate BUY and SELL binary labels using first-touch TP/SL logic over future candles.
- Chronological train/validation split to reduce look-ahead leakage.
- Separate HistGradientBoostingClassifier models for BUY and SELL.
- Validation report with accuracy, precision, recall, F1 and ROC-AUC.
- Saved models are loaded by the HMI and used for real BUY/SELL probability display.

> BUY probability means the model-estimated probability that a long trade reaches the configured TP before SL within the training horizon. SELL is defined symmetrically. HOLD is a derived no-edge indicator, not a third trained class.

## Requirements

- Windows with MetaTrader 5 desktop installed.
- Python 3.10+ recommended.
- MT5 terminal open and logged in.
- Enough historical M15 bars available in MT5 for training.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `config/config.yaml` so `trading.symbol` exactly matches the broker symbol in MT5 Market Watch, for example `XAUUSD.sc`.

## Train the AI

Open MetaTrader 5, then run:

```bash
python train_ai.py
```

Default training configuration:

```yaml
ai_training:
  bars: 50000
  horizon: 8
  take_profit_pct: 0.006
  stop_loss_pct: 0.003
  validation_fraction: 0.20
```

The training script will:

1. download closed M15 bars from MT5;
2. save a local raw CSV under `training_data/`;
3. generate features and BUY/SELL labels;
4. train the two models;
5. save models under `models/`;
6. save validation metrics under `reports/ai_training_report.json`.

Generated model/data/report files are intentionally ignored by Git because they are machine-specific artifacts and can be large.

## Run the HMI

```bash
python main.py
```

After training, the header should show `AI MODEL READY`. If the HMI was already open, click `RELOAD AI MODEL`.

## Important validation note

The first model is a baseline, not proof of profitability. Evaluate validation ROC-AUC, precision/recall at your actual thresholds, then perform walk-forward backtesting including spread, commissions, slippage and position-management rules before enabling any live order execution.

## Safety

The project is still read-only. The monitoring button does not send, modify or close orders. Live order execution should only be added after the AI, backtest and risk layers have been validated on demo data.
