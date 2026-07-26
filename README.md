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

### Step 3 - Multi-Timeframe AI pipeline v2

The current AI architecture is `multi_timeframe_m15_h1_regime_side_specific_v2`.

- M15 remains the execution/signal timeframe.
- Completed H1 candles are aggregated causally from closed M15 bars.
- MT5 candle timestamps are treated as bar-open times; H1 context is aligned using M15 close time (`time + 15 minutes`) to avoid boundary leakage.
- H1 context includes EMA20/EMA50 spread, EMA slope, RSI14, ATR ratio, ADX14, volatility and range position.
- Market-regime flags include trending, ranging, high-volatility and low-volatility states.
- BUY and SELL use separate feature lists instead of being forced to share an identical feature set.
- BUY-specific context includes lower wick, bullish body, H1 uptrend and M15/H1 bullish alignment.
- SELL-specific context includes upper wick, bearish body, H1 downtrend and M15/H1 bearish alignment.
- Separate BUY and SELL binary labels still use first-touch TP/SL logic.
- Purged chronological validation and ambiguous same-bar TP/SL exclusion remain enabled.
- Saved model metadata records the architecture and each side's exact feature list.

> BUY probability estimates whether a long trade reaches TP before SL within the configured horizon. SELL is defined symmetrically. HOLD is a derived no-edge state, not a third trained class.

### Step 4 - Backtester v1

- Uses the trained side-specific BUY/SELL models.
- Signal is generated from a completed candle and entry occurs at the next candle open.
- Only one simulated position is open at a time.
- First-touch TP/SL logic with a time-based exit at the configured horizon.
- Same-bar TP+SL during backtest is handled conservatively as an SL.
- Deducts observed MT5 spread, configured round-trip slippage and commission per lot.
- Uses MT5 `point` and `trade_contract_size` for P/L and transaction-cost calculations.

### Step 5 - Walk-Forward validation and probability calibration

- Retrains a fresh side-specific BUY/SELL model pair in every fold.
- Each test fold is future/out-of-sample relative to fitting and calibration windows.
- Uses purge gaps between fitting, calibration and test.
- Fits Sigmoid and Isotonic calibration only on historical calibration bars.
- Exports raw, sigmoid and isotonic probabilities for identical future test rows.
- Reports threshold sweeps, BUY/SELL performance, Brier Score, ECE and cost-adjusted break-even analysis.

Run development validation with:

```bash
python run_walk_forward.py
```

### Step 6 - Final Untouched Holdout Test (v1 consumed)

The previously registered v1 holdout was already run and inspected. Its historical period must now be treated as **consumed**.

The old locked candidates were:

```text
raw 0.55
raw 0.60
raw 0.80
sigmoid 0.20
```

Do **not** rerun that same 2024 holdout to choose features, thresholds or hyperparameters for the new M15+H1 v2 architecture. Doing so would turn the final holdout into development data.

For v2:

1. Develop and compare the new architecture using walk-forward data only.
2. Freeze the selected v2 architecture and candidate rules.
3. Validate later on a genuinely new untouched period that was not used to design v2.

The existing `run_final_holdout.py` remains for historical reproducibility, not for v2 model selection.

## Requirements

- Windows with MetaTrader 5 desktop installed.
- Python 3.10+ recommended.
- MT5 terminal open and logged in.
- Enough historical M15 bars available for H1 warm-up, training and backtesting.

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

Training now creates side-specific M15+H1 models and stores their exact feature lists in `models/model_meta.joblib`.

## Run Backtester v1

```bash
python run_backtest.py
```

## Run calibrated Walk-Forward Backtest

```bash
python run_walk_forward.py
```

Reports are saved under `reports/walk_forward_*.csv/json`.

## Run the HMI

```bash
python main.py
```

After retraining, reload the AI model from the HMI. The predictor status for the new architecture is `READY_MTF`.

## Important validation note

Backtester v1 can be partly in-sample if its data overlaps model training. Walk-forward validation is the primary development test. A final holdout should be consumed only once for a frozen candidate and must not be recycled for feature selection.

## Safety

The project remains read-only. Monitoring does not send, modify or close orders. Live execution should only be added after the new architecture passes walk-forward validation, a new untouched confirmation period, and dedicated demo-tested risk controls.
