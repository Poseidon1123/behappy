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
- Purged chronological train/validation split to reduce look-ahead leakage.
- Ambiguous same-bar TP/SL samples are excluded from training.
- Separate HistGradientBoostingClassifier models for BUY and SELL.
- Validation report with accuracy, precision, recall, F1 and ROC-AUC.
- Saved models are loaded by the HMI and used for real BUY/SELL probability display.

> BUY probability means the model-estimated probability that a long trade reaches the configured TP before SL within the training horizon. SELL is defined symmetrically. HOLD is a derived no-edge indicator, not a third trained class.

### Step 4 - Backtester v1

- Uses the trained BUY/SELL models on historical closed candles.
- Signal is generated from a completed candle and entry occurs at the next candle open.
- Only one simulated position is open at a time.
- First-touch TP/SL logic with a time-based exit at the configured horizon.
- Same-bar TP+SL during backtest is handled conservatively as an SL.
- Deducts observed MT5 spread, configured round-trip slippage and commission per lot.
- Uses MT5 `point` and `trade_contract_size` for P/L and transaction-cost calculations.
- Produces trade log, equity curve and summary statistics.

### Step 5 - Walk-Forward validation and probability calibration

- Retrains a fresh BUY/SELL model pair in every fold.
- Each test fold is strictly future/out-of-sample relative to its fitting and calibration windows.
- Uses two purge gaps: one between model fitting and calibration, and another between calibration and test.
- Fits Sigmoid and Isotonic calibration only on historical calibration bars.
- Exports raw, sigmoid and isotonic BUY/SELL probabilities for the identical future test rows.
- Runs separate threshold ranges for raw vs calibrated probabilities.
- Reports BUY/SELL performance separately.
- Reports Brier Score and Expected Calibration Error (ECE).
- Adds cost-adjusted break-even probability and calibrated edge-margin analysis.

### Step 6 - Final Untouched Holdout Test

The final holdout test is deliberately isolated from development tuning.

Locked candidates are hard-coded in `backtest/holdout.py`:

```text
raw 0.55
raw 0.60
raw 0.80
sigmoid 0.20
```

No threshold sweep or optimizer is allowed inside the final holdout runner.

Because the most recent 30,000-50,000 bars were already used during development, the final holdout runner does **not** reuse them. By default it downloads 75,000 bars, excludes the newest 50,000 completely, then constructs one older virgin chronological block:

```text
12,000 model-fit bars
-> 8 purge bars
-> 2,000 calibration bars
-> 8 purge bars
-> 5,000 FINAL HOLDOUT bars
-> newest 50,000 development bars excluded entirely
```

Run it once when you are ready to consume that holdout:

```bash
python run_final_holdout.py
```

Outputs:

```text
reports/final_holdout/manifest.json
reports/final_holdout/predictions.csv
reports/final_holdout/candidate_results.csv
reports/final_holdout/candidate_side_results.csv
reports/final_holdout/trades_raw_055.csv
reports/final_holdout/trades_raw_060.csv
reports/final_holdout/trades_raw_080.csv
reports/final_holdout/trades_sigmoid_020.csv
```

`manifest.json` stores the exact train/calibration/holdout timestamps, frozen candidate definitions, trading-cost assumptions and a SHA-256 lock hash. Once this holdout has been run and inspected, treat it as consumed: do not retune the model or thresholds against it and still call it final/untouched.

## Requirements

- Windows with MetaTrader 5 desktop installed.
- Python 3.10+ recommended.
- MT5 terminal open and logged in.
- Enough historical M15 bars available in MT5 for training/backtesting.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `config/config.yaml` so `trading.symbol` exactly matches the broker symbol in MT5 Market Watch, for example `XAUUSD.sc`.

## Train the AI

```bash
python train_ai.py
```

## Run Backtester v1

```bash
python run_backtest.py
```

## Run calibrated Walk-Forward Backtest

```bash
python run_walk_forward.py
```

The walk-forward reports are saved under `reports/walk_forward_*.csv/json` and include raw/calibrated probabilities, threshold sweeps, calibration quality and cost-adjusted edge analysis.

## Run the HMI

```bash
python main.py
```

After training, the header should show `AI MODEL READY`. If the HMI was already open, click `RELOAD AI MODEL`.

## Important validation note

Backtester v1 can be partly in-sample if its data overlaps model training. Walk-forward validation is the main development test. The final untouched holdout is a one-time confirmation test for pre-registered candidates; it should not become another optimization dataset after results are seen.

## Safety

The project is still read-only. The monitoring button does not send, modify or close orders. Live order execution should only be added after validation and dedicated risk controls have been verified on demo data.
