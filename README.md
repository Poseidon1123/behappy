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
- Fits Sigmoid (Platt-style) and Isotonic calibration only on historical calibration bars.
- Exports raw, sigmoid and isotonic BUY/SELL probabilities for the identical future test rows.
- Runs the same threshold sweep for all three probability streams.
- Reports BUY/SELL performance separately.
- Reports Brier Score and Expected Calibration Error (ECE); lower is better.
- Builds calibration bins comparing predicted probability with realized TP-before-SL frequency.

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

Open MetaTrader 5, then run:

```bash
python train_ai.py
```

The trainer downloads closed M15 bars, builds features/labels, applies the purged chronological split, trains the two models and saves local model/report artifacts.

## Run Backtester v1

Train the models first, then run:

```bash
python run_backtest.py
```

Default settings are under `backtest:` in `config/config.yaml`. Set `commission_per_lot_round_turn` to your broker's actual round-trip commission per 1.00 lot. Historical spread is taken from MT5; symbol point size and contract size are read from the broker specification.

Backtest output:

```text
reports/backtest_trades.csv
reports/backtest_equity.csv
reports/backtest_summary.json
```

## Run calibrated Walk-Forward Backtest

Run:

```bash
python run_walk_forward.py
```

Default settings:

```yaml
walk_forward:
  bars: 30000
  train_bars: 12000
  calibration_bars: 2000
  test_bars: 2000
  step_bars: 2000
  purge_bars: 8
  baseline_calibration_method: "raw"
  threshold_sweep:
    - 0.50
    - 0.55
    - 0.60
    - 0.65
    - 0.70
    - 0.72
    - 0.75
    - 0.80
    - 0.85
    - 0.90
  calibration_bins: 10
```

With M15 data, each fold follows this leakage-safe order:

```text
12,000 model-fit bars
-> 8 purge bars
-> 2,000 calibration bars
-> 8 purge bars
-> 2,000 future test bars
```

The rolling window then advances by `step_bars` and repeats. The calibration samples are earlier than the test samples and are never reused as future test data for the same fold.

Outputs:

```text
reports/walk_forward_trades.csv
reports/walk_forward_equity.csv
reports/walk_forward_folds.csv
reports/walk_forward_predictions.csv
reports/walk_forward_side_summary.csv
reports/walk_forward_threshold_sweep.csv
reports/walk_forward_calibration.csv
reports/walk_forward_calibration_metrics.csv
reports/walk_forward_calibration_best.csv
reports/walk_forward_summary.json
reports/walk_forward_analysis.json
```

`walk_forward_predictions.csv` contains raw, sigmoid and isotonic probabilities for the same OOS test rows.

`walk_forward_threshold_sweep.csv` compares trade count, win rate, profit factor, net P/L, expectancy, max drawdown, BUY P/L and SELL P/L for every calibration method at every configured threshold.

`walk_forward_calibration_metrics.csv` compares Brier Score and ECE for raw, sigmoid and isotonic probability streams separately for BUY and SELL. Lower values mean the reported probabilities better match realized frequencies.

`walk_forward_calibration.csv` provides probability-bin diagnostics for all three methods.

`walk_forward_calibration_best.csv` reports the best tested threshold by Profit Factor and by Net Profit for each method. These are diagnostic/model-selection results only; any selected method/threshold must be confirmed on a later untouched period before deployment.

## Run the HMI

```bash
python main.py
```

After training, the header should show `AI MODEL READY`. If the HMI was already open, click `RELOAD AI MODEL`.

## Important validation note

Backtester v1 can be partly in-sample if its data overlaps model training. The calibrated walk-forward test is preferred because model fitting, probability calibration and final testing occur in chronological order with purge gaps. A profitable result still does not prove future profitability; inspect fold-to-fold stability, costs, drawdown, trade count, calibration quality and demo performance before enabling order execution.

## Safety

The project is still read-only. The monitoring button does not send, modify or close orders. Live order execution should only be added after walk-forward validation, probability calibration and risk controls have been validated on demo data.
