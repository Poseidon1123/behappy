# AI MT5 Trading Bot

Python foundation for an AI-assisted MetaTrader 5 trading system.

## Implemented

### Step 1 - MetaTrader 5 connection
- Connect to a running MetaTrader 5 terminal.
- Read account information and market ticks.
- Download closed OHLC bars.

### Step 2 - PySide6 HMI
- Realtime price chart, account state, open positions, risk controls and AI probabilities.

### Step 3 - AI research pipeline
- Separate BUY/SELL binary models.
- Purged chronological validation.
- Ambiguous same-bar TP/SL training samples are excluded.
- M15, H1 context and market-regime feature experiments.

### Step 4 - Backtester
- Next-bar-open entries.
- TP/SL/time exits.
- Spread, slippage and commission modelling.

### Step 5 - Walk-Forward + calibration
- Raw, Sigmoid and Isotonic probability comparison.
- Cost-adjusted break-even probability analysis.

### Step 6 - Consumed Final Holdout
The previously used 2024 holdout is consumed and must not be reused for current architecture or threshold selection.

### Step 7 - Feature Ablation
Run:
```bash
python run_feature_ablation.py
```

### Step 8 - Nested Walk-Forward v3
BUY is fixed to `A_m15_baseline`; SELL is chosen inside inner validation from A/B/E and BUY/SELL thresholds are selected independently.

Run:
```bash
python run_nested_walk_forward.py
```

### Step 9 - Nested v3.1 Robust Selector
v3.1 reduces dependence on one inner validation window. For every outer fold it evaluates each threshold/SELL architecture across multiple historical inner sub-folds, then rewards average performance and repeated positive expectancy while penalizing fold-to-fold instability.

Default robust structure:
```text
12,000 outer-training bars
  -> 3 historical inner validation sub-folds (1,500 bars each)
  -> purge gaps of at least horizon bars
  -> robust BUY threshold selection
  -> robust SELL A/B/E architecture + threshold selection
  -> refit selected models on the full outer-training window
  -> 2,000 future outer-test bars
```

BUY architecture remains fixed:
```text
A_m15_baseline
```

SELL candidates remain predeclared:
```text
A_m15_baseline
B_m15_raw_h1
E_m15_regime_only
```

Threshold grid remains predeclared:
```text
0.50 0.55 0.60 0.65 0.70 0.72 0.75 0.80
```

Run:
```bash
python run_nested_robust.py
```

Outputs:
```text
reports/nested_v3_1/summary.json
reports/nested_v3_1/outer_trades.csv
reports/nested_v3_1/outer_folds.csv
reports/nested_v3_1/outer_side_summary.csv
reports/nested_v3_1/inner_selections.csv
reports/nested_v3_1/robust_candidate_scores.csv
reports/nested_v3_1/inner_split_diagnostics.csv
reports/nested_v3_1/sell_architecture_selection_counts.csv
```

`robust_candidate_scores.csv` is especially useful for checking why a threshold/architecture was selected: it reports mean and standard deviation of PF/expectancy, positive inner-split ratio and the final robust score.

If all candidates are weak, v3.1 still selects the least-bad candidate using inner history only. It never inspects an outer test before deciding to keep or discard that fold.

## Install
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

## Main commands
```bash
python train_ai.py
python run_backtest.py
python run_walk_forward.py
python run_feature_ablation.py
python run_nested_walk_forward.py
python run_nested_robust.py
python main.py
```

## Validation policy
Ordinary walk-forward and feature ablation are development/model-selection analyses. Nested v3 and robust v3.1 provide stronger outer-test estimates because selections occur only inside historical training data, but they are still not substitutes for a genuinely new untouched confirmation period after the final rule set is frozen.

## Safety
The project remains read-only. Monitoring does not send, modify or close orders. Live execution should only be added after a stable nested result, a new untouched confirmation period and dedicated demo-tested risk controls.
