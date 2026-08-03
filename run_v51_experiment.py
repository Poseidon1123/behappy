from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from backtest.engine import BacktestConfig
from backtest.meta_labeling_v51 import run_meta_labeling_walk_forward_v51
from data.snapshot import load_snapshot


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inner-selected ATR exits on v5")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    return parser.parse_args()


def _fold_ratio(folds: pd.DataFrame, prefix: str) -> float:
    column = f"{prefix}_net_profit"
    if folds.empty or column not in folds:
        return 0.0
    return float((folds[column].astype(float) > 0.0).mean())


def _assessment(
    summary: dict[str, Any],
    folds: pd.DataFrame,
    criteria: dict[str, Any],
) -> dict[str, Any]:
    variants = {
        "baseline": (summary["baseline"], "baseline"),
        "v4.1_all_gated": (summary["v41_all_gated"], "v41"),
        "v4.2_hybrid": (summary["v42_hybrid"], "v42"),
        "v4.3_buy_only": (summary["buy_only"], "buy_only"),
        "v5_causal_drift": (summary["v5_causal_drift"], "v5"),
        "v5.1_atr_adaptive": (summary["v51_atr_adaptive"], "v51"),
    }
    rows: list[dict[str, Any]] = []
    for name, (metrics, prefix) in variants.items():
        pf = metrics.get("profit_factor")
        fold_ratio = _fold_ratio(folds, prefix)
        checks = {
            "profit_factor": pf is not None and float(pf) >= float(criteria["min_profit_factor"]),
            "expectancy": float(metrics.get("expectancy", 0.0)) >= float(criteria["min_expectancy"]),
            "max_drawdown_pct": float(metrics.get("max_drawdown_pct", float("inf"))) <= float(criteria["max_drawdown_pct"]),
            "profitable_fold_ratio": fold_ratio >= float(criteria["min_profitable_fold_ratio"]),
            "trades": int(metrics.get("trades", 0)) >= int(criteria["min_trades"]),
        }
        rows.append(
            {
                "variant": name,
                "passed": all(checks.values()),
                "checks": checks,
                "profitable_fold_ratio": fold_ratio,
                "metrics": metrics,
            }
        )
    eligible = [row for row in rows if row["passed"]]
    eligible.sort(
        key=lambda row: (
            float(row["metrics"].get("profit_factor") or 0.0),
            float(row["metrics"].get("expectancy", 0.0)),
            -float(row["metrics"].get("max_drawdown_pct", float("inf"))),
        ),
        reverse=True,
    )
    selected = eligible[0]["variant"] if eligible else "NO_DEPLOY"
    return {
        "criteria_locked_before_test": criteria,
        "selected_candidate": selected,
        "deployment_allowed": selected != "NO_DEPLOY",
        "variants": rows,
    }


def main() -> None:
    args = _arguments()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    backtest = config.get("backtest", {})
    nested = config.get("nested_robust", {})
    meta = config.get("meta_labeling_v42", config.get("meta_labeling_v41", {}))
    drift = config.get("drift_v5", {})
    criteria_config = config.get("research_acceptance", {})
    data, manifest = load_snapshot(args.data, args.manifest)

    criteria = {
        "min_profit_factor": float(criteria_config.get("min_profit_factor", 1.10)),
        "min_expectancy": float(criteria_config.get("min_expectancy", 0.0)),
        "max_drawdown_pct": float(criteria_config.get("max_drawdown_pct", 5.0)),
        "min_profitable_fold_ratio": float(criteria_config.get("min_profitable_fold_ratio", 0.625)),
        "min_trades": int(criteria_config.get("min_trades", 200)),
    }
    cfg = BacktestConfig(
        initial_balance=float(backtest.get("initial_balance", 1000.0)),
        fixed_lot=float(backtest.get("fixed_lot", 0.01)),
        buy_threshold=float(backtest.get("buy_threshold", 0.72)),
        sell_threshold=float(backtest.get("sell_threshold", 0.72)),
        min_probability_edge=float(backtest.get("min_probability_edge", 0.08)),
        horizon=int(backtest.get("horizon", 8)),
        take_profit_pct=float(backtest.get("take_profit_pct", 0.006)),
        stop_loss_pct=float(backtest.get("stop_loss_pct", 0.003)),
        slippage_points=float(backtest.get("slippage_points", 2.0)),
        commission_per_lot_round_turn=float(backtest.get("commission_per_lot_round_turn", 0.0)),
        point=float(manifest["point"]),
        contract_size=float(manifest["contract_size"]),
    )
    thresholds = [float(x) for x in nested.get(
        "thresholds", [0.50, 0.55, 0.60, 0.65, 0.70, 0.72, 0.75, 0.80]
    )]
    print("V5.1 INNER-SELECTED ATR EXIT EXPERIMENT")
    print(f"SHA-256: {manifest['sha256']}")
    print(f"Period: {manifest['first_bar_utc']} -> {manifest['last_bar_utc']}")
    print("Drift cutoff is calibrated from OUTER TRAINING only")

    baseline, v41, v42, buy_only, v5, v51, folds, meta_training, drift_diagnostics, exit_selections, summary = run_meta_labeling_walk_forward_v51(
        data,
        cfg,
        thresholds=thresholds,
        outer_train_bars=int(nested.get("outer_train_bars", 12000)),
        inner_validation_bars=int(nested.get("inner_validation_bars", 1500)),
        inner_splits=int(nested.get("inner_splits", 3)),
        outer_test_bars=int(nested.get("outer_test_bars", 2000)),
        step_bars=int(nested.get("step_bars", 2000)),
        purge_bars=int(nested.get("purge_bars", cfg.horizon)),
        min_total_inner_trades=int(nested.get("min_total_inner_trades", 45)),
        meta_gate_threshold=float(meta.get("gate_threshold", 0.55)),
        buy_harvest_threshold=float(meta.get("buy_harvest_threshold", 0.45)),
        sell_harvest_threshold=float(meta.get("sell_harvest_threshold", 0.45)),
        min_meta_samples=int(meta.get("min_meta_samples", 250)),
        drift_recent_window=int(drift.get("recent_window", 500)),
        drift_score_step=int(drift.get("score_step", 100)),
        drift_calibration_step=int(drift.get("calibration_step", 250)),
        drift_cutoff_quantile=float(drift.get("cutoff_quantile", 0.95)),
    )
    assessment = _assessment(summary, folds, criteria)
    summary["dataset_manifest"] = manifest
    summary["assessment"] = assessment

    out = Path("reports") / f"v51_{manifest['sha256'][:12]}"
    out.mkdir(parents=True, exist_ok=True)
    for filename, frame in {
        "baseline_trades.csv": baseline,
        "v41_all_gated_trades.csv": v41,
        "v42_hybrid_trades.csv": v42,
        "v43_buy_only_trades.csv": buy_only,
        "v5_causal_drift_trades.csv": v5,
        "v51_atr_adaptive_trades.csv": v51,
        "fold_comparison.csv": folds,
        "meta_training_availability.csv": meta_training,
        "drift_diagnostics.csv": drift_diagnostics,
        "exit_policy_selections.csv": exit_selections,
    }.items():
        frame.to_csv(out / filename, index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "decision.json").write_text(json.dumps(assessment, indent=2), encoding="utf-8")

    print("\nRESULTS")
    print(json.dumps(
        {
            "baseline": summary["baseline"],
            "v4.1_all_gated": summary["v41_all_gated"],
            "v4.2_hybrid": summary["v42_hybrid"],
            "v4.3_buy_only": summary["buy_only"],
            "v5_causal_drift": summary["v5_causal_drift"],
            "v5.1_atr_adaptive": summary["v51_atr_adaptive"],
        },
        indent=2,
    ))
    print("\nLOCKED DECISION")
    print(json.dumps(assessment, indent=2))
    print(f"\nReports saved under {out}/")


if __name__ == "__main__":
    main()
