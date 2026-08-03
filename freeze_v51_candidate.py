from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from backtest.engine import BacktestConfig
from data.snapshot import load_snapshot
from shadow.bundle import freeze_v51_bundle, save_bundle


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the v5.1 SHADOW-ONLY candidate")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--output", type=Path, default=Path("models/v51_shadow_bundle.joblib"))
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    bt = config.get("backtest", {})
    nested = config.get("nested_robust", {})
    meta = config.get("meta_labeling_v42", config.get("meta_labeling_v41", {}))
    drift = config.get("drift_v5", {})
    data, snapshot_manifest = load_snapshot(args.data, args.manifest)
    cfg = BacktestConfig(
        initial_balance=float(bt.get("initial_balance", 1000.0)),
        fixed_lot=float(bt.get("fixed_lot", 0.01)),
        buy_threshold=float(bt.get("buy_threshold", 0.72)),
        sell_threshold=float(bt.get("sell_threshold", 0.72)),
        min_probability_edge=float(bt.get("min_probability_edge", 0.08)),
        horizon=int(bt.get("horizon", 8)),
        take_profit_pct=float(bt.get("take_profit_pct", 0.006)),
        stop_loss_pct=float(bt.get("stop_loss_pct", 0.003)),
        slippage_points=float(bt.get("slippage_points", 2.0)),
        commission_per_lot_round_turn=float(bt.get("commission_per_lot_round_turn", 0.0)),
        point=float(snapshot_manifest["point"]),
        contract_size=float(snapshot_manifest["contract_size"]),
    )
    bundle, manifest = freeze_v51_bundle(
        data,
        snapshot_manifest,
        cfg,
        thresholds=[float(x) for x in nested.get("thresholds", [0.50, 0.55, 0.60, 0.65, 0.70, 0.72, 0.75, 0.80])],
        outer_train_bars=int(nested.get("outer_train_bars", 12000)),
        inner_validation_bars=int(nested.get("inner_validation_bars", 1500)),
        inner_splits=int(nested.get("inner_splits", 3)),
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
    output, manifest_path, final_manifest = save_bundle(bundle, manifest, args.output)
    print("V5.1 CANDIDATE FROZEN — SHADOW ONLY")
    print(json.dumps(final_manifest, indent=2))
    print(f"Bundle: {output}")
    print(f"Manifest: {manifest_path}")
    print("SAFETY: deployment_allowed=false; this bundle must not send broker orders.")


if __name__ == "__main__":
    main()
