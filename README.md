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

### Step 3 - Real AI pipeline v2

- M15 remains the signal/execution timeframe.
- Completed H1 candles are aggregated causally from closed M15 history.
- H1 context includes EMA spread/slope, RSI, ATR ratio, ADX, volatility and range position.
- Market-regime features describe trending/ranging and high/low-volatility conditions.
- BUY and SELL models use separate feature sets and asymmetric candle information.
- Separate BUY and SELL binary labels use first-touch TP/SL logic over future candles.
- Purged chronological train/validation split reduces look-ahead leakage.
- Ambiguous same-bar TP/SL samples are excluded from training.
- HistGradientBoostingClassifier models are trained independently for BUY and SELL.

> BUY probability means the model-estimated probability that a long trade reaches the configured TP before SL within the training horizon. SELL is defined symmetrically. HOLD is a derived no-edge state, not a third trained class.

### Step 4 - Backtester v1

- Signal is generated from a completed candle and entry occurs at the next candle open.
- Only one simulated position is open at a time.
- First-touch TP/SL logic with a time-based exit at the configured horizon.
- Same-bar TP+SL during backtest is handled conservatively as an SL.
- Deducts observed MT5 spread, configured round-trip slippage and commission per lot.
- Uses MT5 `point` and `trade_contract_size` for P/L and transaction-cost calculations.

### Step 5 - Walk-Forward validation and probability calibration

- Retrains a fresh BUY/SELL model pair in every fold.
- Each test fold is strictly future/out-of-sample relative to fitting/calibration windows.
- Uses purge gaps between model fitting, calibration and test.
- Fits Sigmoid and Isotonic calibration only on historical calibration bars.
- Reports raw/calibrated probabilities, BUY/SELL performance, Brier Score and ECE.
- Adds cost-adjusted break-even probability and calibrated edge-margin analysis.

### Step 6 - Final Untouched Holdout Test

The previously run 2024 final holdout is consumed and must not be reused to choose v2 features, thresholds or hyperparameters. The holdout runner remains only for reproducibility of that historical experiment.

### Step 7 - Feature Ablation Walk-Forward

Run five feature groups on identical walk-forward windows and trading-cost assumptions:

```text
A_m15_baseline
  M15 common features + side-specific candle shape

B_m15_raw_h1
  A + continuous/raw H1 indicators

C_m15_h1_regime_full
  full v2 H1 + regime + directional/alignment flags

D_m15_h1_regime_no_alignment
  H1 + regime, but removes hand-built H1 direction/alignment flags

E_m15_regime_only
  M15 + regime flags, without raw H1 indicators
```

Run:

```bash
python run_feature_ablation.py
```

Outputs:

```text
reports/feature_ablation/summary_fixed_threshold.csv
reports/feature_ablation/side_summary_fixed_threshold.csv
reports/feature_ablation/threshold_sweep.csv
reports/feature_ablation/best_thresholds.csv
reports/feature_ablation/fold_summary.csv
reports/feature_ablation/feature_groups.json
reports/feature_ablation/trades_<group>.csv
```

The fixed-threshold summary is the cleanest apples-to-apples comparison. The threshold sweep is development/model-selection information only and must later be confirmed on a genuinely new untouched period.

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

Reports are saved under `reports/walk_forward_*.csv/json`.

## Run Feature Ablation

```bash
python run_feature_ablation.py
```

## Run the HMI

```bash
python main.py
```

After retraining, reload the AI model from the HMI. The predictor status for the new architecture is `READY_MTF`.

## Important validation note

Backtester v1 can be partly in-sample if its data overlaps model training. Walk-forward validation is the primary development test. Feature ablation is also development/model-selection analysis. A final holdout should be consumed only once for a frozen candidate and must not be recycled for feature selection.

## Safety

The project remains read-only. Monitoring does not send, modify or close orders. Live execution should only be added after the new architecture passes walk-forward validation, a new untouched confirmation period, and dedicated demo-tested risk controls.
