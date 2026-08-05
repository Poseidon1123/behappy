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

### Step 10 - Frozen v3.1 Forward Confirmation
The v3.1 rule set is frozen in `config/frozen_v3_1.json`. The development cutoff is fixed at `2026-06-25 04:30:00 UTC`. The post-cutoff confirmation window has now been inspected and is therefore consumed.

Run for reproducibility only:
```bash
python run_forward_confirmation.py
```

### Step 11 - v4 Meta-Labeling / Trade Gate
v4 keeps the primary BUY/SELL opportunity models, but adds a second AI layer that decides whether a candidate should actually be traded or skipped.

The first v4 baseline uses TP/SL outcome labels and a fixed gate threshold of `0.55`. Meta models are trained only from out-of-fold primary probabilities generated inside each outer-training history.

Run:
```bash
python run_meta_labeling.py
```

### Step 12 - v4.1 Economic Meta Gate
v4.1 changes the gate objective from a TP/SL class label to an economic label:

```text
meta_target = 1 when realized net PnL > 0 after spread + slippage + commission
meta_target = 0 otherwise
```

Key changes:
- BUY and SELL use different meta-feature sets.
- Meta training candidates are harvested more broadly at a fixed OOF probability floor of `0.45`, even when the final primary threshold is higher.
- Primary probabilities used for meta training remain strictly out-of-fold.
- Large wins/losses receive more sample weight than near-flat outcomes.
- The gate threshold stays fixed at `0.55`; there is no gate-threshold optimizer in this baseline test.
- Outer tests are still excluded from both primary and meta training.

Run:
```bash
python run_meta_labeling_v41.py
```

Outputs:
```text
reports/meta_labeling_v4_1/summary.json
reports/meta_labeling_v4_1/baseline_trades.csv
reports/meta_labeling_v4_1/gated_trades.csv
reports/meta_labeling_v4_1/fold_comparison.csv
reports/meta_labeling_v4_1/meta_training_availability.csv
```

The key test is whether v4.1 can improve Profit Factor and expectancy while preserving enough trades across multiple outer folds. Previously inspected holdouts and the June-July 2026 forward-confirmation window remain consumed and are not valid clean confirmations for v4.1.

### Step 13 - v4.2 Hybrid Side-Specific Gate
v4.2 keeps BUY signals from the primary model unfiltered and applies the v4.1
economic meta gate only to SELL candidates. It independently replays three
strategies on every outer fold so skipped SELL trades can expose later signals:

```text
v3.1 baseline     -> no gate
v4.1 comparison   -> BUY and SELL gated
v4.2 hybrid       -> BUY direct, SELL gated
```

The default simulated initial balance is now `1000.0`. Fixed lot remains
`0.01`, so absolute trade PnL is unchanged while balance and drawdown
percentages reflect the smaller account.

Run:
```bash
python run_meta_labeling_v42.py
```

Outputs:
```text
reports/meta_labeling_v4_2/summary.json
reports/meta_labeling_v4_2/baseline_trades.csv
reports/meta_labeling_v4_2/v41_all_gated_trades.csv
reports/meta_labeling_v4_2/v42_hybrid_trades.csv
reports/meta_labeling_v4_2/fold_comparison.csv
reports/meta_labeling_v4_2/meta_training_availability.csv
```

### Step 14 - Frozen Dataset + Reproducible Selection
Live MT5 downloads move forward over time, so two runs using the "latest
30,000 bars" are not directly comparable. Freeze one closed-bar dataset first:

```bash
python export_mt5_snapshot.py
```

This creates an untracked CSV plus a JSON manifest containing the exact period,
row count, broker contract metadata and SHA-256 checksum. Do not edit the CSV.

Run all current candidates offline on that immutable snapshot:

```bash
python run_reproducible_experiment.py --data snapshots/xauusd_sc_m15_30000.csv
```

The experiment verifies the checksum before training and independently replays
the baseline, v4.1 all-gated and v4.2 SELL-only strategies. Selection criteria
are declared in `config/config.yaml` before the test. If no candidate passes
every locked requirement, `decision.json` returns `NO_DEPLOY` instead of
forcing the least-bad model into live trading.

Large histories are downloaded in chunks to avoid MT5's single-request size
limit. For example:

```bash
python export_mt5_snapshot.py --bars 100000 --output snapshots/xauusd_sc_m15_100000.csv
```

If the broker terminal has fewer bars cached, the exporter stops instead of
silently creating a short snapshot and explains how to increase MT5's
`Max bars in chart` setting.

### Step 15 - v5 Causal Drift/OOD Gate
v5 adds a distribution-drift gate to the v4.2 hybrid architecture and also
replays a BUY-only control strategy. Drift calibration uses only rolling
windows inside each outer-training history. Test scores are computed from bars
strictly before each decision block and never read future outer-test bars.

Fixed v5 drift policy:

```text
recent window       500 bars
score refresh       100 bars
calibration step    250 bars
training percentile 95%
```

Run on a frozen snapshot:

```bash
python run_v5_experiment.py --data snapshots/xauusd_sc_m15_90000.csv
```

The runner independently compares baseline, v4.1 all-gated, v4.2 hybrid,
v4.3 BUY-only and v5 causal drift. It retains the locked `NO_DEPLOY` acceptance
policy and writes drift diagnostics alongside trade and fold reports.

### Step 16 - v5.1 Inner-Selected ATR Exits
v5.1 keeps v5 signal, meta and drift logic fixed, then selects one exit policy
inside historical inner validation for each outer fold. The outer test never
participates in choosing TP, SL or holding horizon.

Predeclared candidates:

```text
fixed TP 0.6%, SL 0.3%, horizon 8
ATR TP 3.0, SL 1.5, horizon 12
ATR TP 3.0, SL 2.0, horizon 16
ATR TP 4.0, SL 2.0, horizon 24
```

ATR is calculated on the closed signal bar; entry remains next-bar open.
Same-bar TP/SL ambiguity remains a conservative SL.

Run:

```bash
python run_v51_experiment.py --data snapshots/xauusd_sc_m15_90000.csv
```

Reports include the policy selected per outer fold and every inner candidate's
robust score. The locked deployment criteria remain unchanged.

### Step 17 - Frozen v5.1 Shadow/Paper Bot

Freeze one candidate from the immutable snapshot. The bundle records the data
SHA-256, training cutoff, selected primary architectures and thresholds, SELL
economic meta gate, causal drift reference/cutoff and inner-selected exit policy.

```bash
python freeze_v51_candidate.py --data snapshots/xauusd_sc_m15_90000.csv
```

Run one pass over newly closed MT5 bars:

```bash
python run_v51_shadow.py
```

Or keep polling for new closed bars:

```bash
python run_v51_shadow.py --watch --poll-seconds 30
```

The runner starts with the configured virtual balance (`1000.00`), persists its
state after each bar, and writes every decision, simulated entry and exit under
`shadow_logs/`. Signals enter at the next bar open and use the same conservative
cost and same-bar TP/SL assumptions as the backtest. It is deliberately marked
`SHADOW_ONLY`, rejects non-shadow bundles and contains no broker order operation.

### Step 18 - Guarded MT5 Demo Execution

The demo runner reuses the same frozen bundle, fixed `0.01` lot and v5.1
decision pipeline. It hard-refuses every account whose MT5 `trade_mode` is not
`ACCOUNT_TRADE_MODE_DEMO`, limits the account to one open position, applies
spread/daily-loss/drawdown gates and calls `order_check` before `order_send`.

First run it without order permission:

```bash
python run_v51_demo.py --once
```

After confirming the printed account mode is `DEMO`, enable demo orders:

```bash
python run_v51_demo.py --enable-demo-orders
```

Keep the MT5 Algo Trading button enabled. The `--enable-demo-orders` flag cannot
override the hard demo-account check. Real-money execution remains unavailable.
In continuous mode the runner does not poll every 30 seconds. It waits for the
next opening boundary of the bundle timeframe, adds a short data-settlement
delay (default 2 seconds), and evaluates the newly closed candle exactly once.
Starting the bot midway through a candle therefore waits for the next candle.
`--once` remains an explicit diagnostic command and evaluates immediately.
Persistent events are written to
`demo_logs/v51_demo_events.jsonl`.

Optional fixed-percentage TP/SL overrides are DEMO experiments and must always
be provided as a pair. For example:

```bash
python run_v51_demo.py --enable-demo-orders --take-profit-percent 0.60 --stop-loss-percent 0.30
```

Demo lot can also be overridden. The runner checks the requested value against
the symbol's broker-provided minimum, maximum and volume step before any order:

```bash
python run_v51_demo.py --enable-demo-orders --fixed-lot 0.02
```

### Step 19 - v5.1 Demo Control HMI

Open the dedicated desktop dashboard:

```bash
python run_v51_demo_hmi.py
```

The HMI can start/stop the guarded demo runner; enable or disable demo orders;
adjust BUY, SELL and SELL-meta thresholds; adjust fixed lot; optionally override TP and SL as a
percentage of entry; tune spread, daily-loss, drawdown and the short candle-open
data delay; inspect market charts/candles across M1–H4; and monitor open
positions, the bot's 30-day deal history and persistent AI/execution events.

The bot evaluates entries only once near the beginning of each selected model
candle. MT5/broker TP and SL remain active continuously between decisions.

The bot-processing selector supports M1, M5, M15, M30, H1 and H4. Each selection
requires its own frozen bundle; the HMI never reuses the M15 model on another
timeframe. Select a timeframe and click `BUILD SELECTED MODEL` to download its
closed MT5 bars, freeze a dedicated bundle and enable Start. Chart timeframe is
independently selectable. New timeframe models are DEVELOPMENT/DEMO candidates
and do not inherit the M15 validation result.

The equivalent CLI workflow is:

```bash
python prepare_v51_timeframe.py --timeframe M5 --bars 30000
```

Changing model thresholds marks the session `EXPERIMENTAL` and requires a new
confirmation before starting. Lot remains fixed at `0.01`, and the backend still
hard-refuses real accounts regardless of HMI settings.

Stop any command-line `run_v51_demo.py` process with `Ctrl+C` before starting the
HMI. An operating-system lock prevents two runners from sharing one state and
potentially processing the same closed candle concurrently.

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
python run_forward_confirmation.py
python run_meta_labeling.py
python run_meta_labeling_v41.py
python run_meta_labeling_v42.py
python export_mt5_snapshot.py
python run_reproducible_experiment.py --data snapshots/xauusd_sc_m15_30000.csv
python run_v5_experiment.py --data snapshots/xauusd_sc_m15_90000.csv
python run_v51_experiment.py --data snapshots/xauusd_sc_m15_90000.csv
python freeze_v51_candidate.py --data snapshots/xauusd_sc_m15_90000.csv
python run_v51_shadow.py
python run_v51_demo.py --once
python run_v51_demo_hmi.py
python main.py
```

## Validation policy
Ordinary walk-forward and feature ablation are development/model-selection analyses. Nested v3/v3.1 and v4/v4.1/v4.2 meta-labeling use stricter outer-test separation, but a genuinely new untouched period is still required after a final rule set is frozen. Previously inspected holdouts remain consumed.

## Safety
The project remains read-only. Monitoring does not send, modify or close orders. Live execution should only be added after a stable nested result, a new untouched confirmation period and dedicated demo-tested risk controls.
