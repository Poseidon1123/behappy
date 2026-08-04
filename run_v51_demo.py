from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd

from backtest.drift import drift_score
from backtest.meta_labeling_v41 import _meta_vector
from backtest.nested_robust import _outer_decision
from data.feature_engineering import build_features
from demo.executor import (
    append_demo_event,
    bot_positions,
    close_demo_position,
    load_demo_state,
    open_demo_position,
    save_demo_state,
)
from demo.lock import SingleInstanceError, single_instance_lock
from demo.safety import DemoSafetyError, SafetyLimits, refresh_equity_limits, require_demo_account, require_spread
from shadow.engine import load_verified_bundle


TIMEFRAME_SECONDS = {
    "M1": 60,
    "M5": 5 * 60,
    "M15": 15 * 60,
    "M30": 30 * 60,
    "H1": 60 * 60,
    "H4": 4 * 60 * 60,
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen v5.1 DEMO-ACCOUNT execution")
    parser.add_argument("--bundle", type=Path, default=Path("models/v51_shadow_bundle.joblib"))
    parser.add_argument("--state", type=Path, default=Path("demo_state/v51_demo_state.json"))
    parser.add_argument("--events", type=Path, default=Path("demo_logs/v51_demo_events.jsonl"))
    parser.add_argument("--bars", type=int, default=10000)
    parser.add_argument(
        "--bar-open-delay-seconds",
        type=float,
        default=2.0,
        help="Wait this many seconds after a new timeframe candle opens before evaluating the closed candle",
    )
    parser.add_argument("--deviation-points", type=int, default=20)
    parser.add_argument("--max-spread-points", type=float, default=100.0)
    parser.add_argument("--max-daily-loss-pct", type=float, default=2.0)
    parser.add_argument("--max-drawdown-pct", type=float, default=5.0)
    parser.add_argument("--buy-threshold", type=float, default=None)
    parser.add_argument("--sell-threshold", type=float, default=None)
    parser.add_argument("--meta-threshold", type=float, default=None)
    parser.add_argument(
        "--take-profit-percent",
        type=float,
        default=None,
        help="Experimental fixed TP distance in percent of entry; overrides the bundle exit policy",
    )
    parser.add_argument(
        "--stop-loss-percent",
        type=float,
        default=None,
        help="Experimental fixed SL distance in percent of entry; overrides the bundle exit policy",
    )
    parser.add_argument("--enable-demo-orders", action="store_true")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def _signal(
    bundle: dict,
    frame: pd.DataFrame,
    state: dict,
    *,
    buy_threshold: float | None = None,
    sell_threshold: float | None = None,
    meta_threshold: float | None = None,
) -> dict:
    raw = pd.concat([bundle["raw_context_tail"], frame], ignore_index=True)
    raw["time"] = pd.to_datetime(raw["time"], utc=True)
    raw.sort_values("time", inplace=True)
    raw.drop_duplicates("time", keep="last", inplace=True)
    timeframe = str(bundle["snapshot_manifest"].get("timeframe", "M15"))
    featured = build_features(raw, base_timeframe=timeframe).reset_index(drop=True)
    row = featured.iloc[-1]
    bar_time = row["time"].isoformat()
    previous = pd.Timestamp(state["last_processed_bar_utc"]) if state.get("last_processed_bar_utc") else None
    new_bars = int((featured["time"] > previous).sum()) if previous is not None else 1
    if state.get("last_processed_bar_utc") == bar_time:
        return {"side": "HOLD", "reason": "BAR_ALREADY_PROCESSED", "bar_time": bar_time, "new_bars": 0}
    required = bundle["buy_features"] + bundle["sell_features"] + ["atr14_abs"]
    if not row[required].notna().all():
        return {"side": "HOLD", "reason": "FEATURES_NOT_READY", "bar_time": bar_time, "new_bars": new_bars}
    recent = featured.iloc[max(0, len(featured) - 1 - bundle["drift_recent_window"]):-1]
    score = drift_score(bundle["drift_reference"], recent)
    if not math.isfinite(score) or score > bundle["drift_cutoff"]:
        return {"side": "HOLD", "reason": "DRIFT_BLOCK", "bar_time": bar_time, "new_bars": new_bars, "drift_score": score}
    buy_p = float(bundle["buy_model"].predict_proba(pd.DataFrame([row[bundle["buy_features"]]]))[:, 1][0])
    sell_p = float(bundle["sell_model"].predict_proba(pd.DataFrame([row[bundle["sell_features"]]]))[:, 1][0])
    buy_limit = float(bundle["buy_threshold"] if buy_threshold is None else buy_threshold)
    sell_limit = float(bundle["sell_threshold"] if sell_threshold is None else sell_threshold)
    meta_limit = float(bundle["meta_gate_threshold"] if meta_threshold is None else meta_threshold)
    side = _outer_decision(buy_p, sell_p, buy_threshold=buy_limit, sell_threshold=sell_limit, min_probability_edge=bundle["min_probability_edge"])
    meta_p = None
    reason = "PRIMARY_HOLD"
    if side == "SELL":
        vector = _meta_vector(row, side="SELL", primary=sell_p, opposing=buy_p, primary_threshold=sell_limit)
        meta_p = float(bundle["sell_meta_model"].predict_proba(vector)[:, 1][0])
        if meta_p < meta_limit:
            side, reason = "HOLD", "SELL_META_BLOCK"
        else:
            reason = "SELL_META_PASS"
    elif side == "BUY":
        reason = "BUY_PRIMARY"
    return {"side": side, "reason": reason, "bar_time": bar_time, "new_bars": new_bars, "buy_probability": buy_p, "sell_probability": sell_p, "meta_probability": meta_p, "drift_score": score, "atr14_abs": float(row["atr14_abs"])}


def _cycle(args: argparse.Namespace) -> dict:
    import MetaTrader5 as mt5
    from mt5.market_data import MarketData
    from mt5.mt5_connector import MT5Connector

    bundle, manifest = load_verified_bundle(args.bundle)
    state = load_demo_state(args.state, manifest["bundle_sha256"])
    snapshot, cfg, policy = bundle["snapshot_manifest"], bundle["backtest_config"], bundle["exit_policy"]
    limits = SafetyLimits(args.max_daily_loss_pct, args.max_drawdown_pct, args.max_spread_points)
    with MT5Connector() as connector:
        account, terminal = mt5.account_info(), mt5.terminal_info()
        require_demo_account(mt5, account, terminal)
        equity_block = refresh_equity_limits(state, equity=float(account.equity), limits=limits)
        bars = MarketData().get_bars_chunked(snapshot["symbol"], snapshot["timeframe"], count=args.bars, chunk_size=min(5000, args.bars), require_full_count=False)
        signal = _signal(
            bundle,
            bars,
            state,
            buy_threshold=args.buy_threshold,
            sell_threshold=args.sell_threshold,
            meta_threshold=args.meta_threshold,
        )
        positions = bot_positions(mt5, snapshot["symbol"])
        all_positions = mt5.positions_get()
        if all_positions is None:
            raise RuntimeError(f"positions_get failed: {mt5.last_error()}")

        if positions:
            if len(positions) != 1:
                raise DemoSafetyError("REFUSED: more than one v5.1 demo position exists")
            state["position_bars_elapsed"] = int(state.get("position_bars_elapsed", 0)) + int(signal["new_bars"])
            if state["position_bars_elapsed"] >= int(policy.horizon) and args.enable_demo_orders:
                result = close_demo_position(mt5, positions[0], args.deviation_points)
                action = {"action": "TIME_EXIT", "retcode": int(result.retcode), "ticket": int(positions[0].ticket)}
                state["position_bars_elapsed"] = 0
                state["position_entry_bar_utc"] = None
            else:
                action = {"action": "POSITION_MANAGED", "ticket": int(positions[0].ticket), "bars_elapsed": state["position_bars_elapsed"]}
        elif signal["side"] in {"BUY", "SELL"}:
            if equity_block:
                raise DemoSafetyError(f"REFUSED: {equity_block}; new entries are blocked")
            if len(all_positions) > 0:
                raise DemoSafetyError("REFUSED: another MT5 position is open; max_positions=1")
            tick, info = mt5.symbol_info_tick(snapshot["symbol"]), mt5.symbol_info(snapshot["symbol"])
            if tick is None or info is None:
                raise RuntimeError(f"Could not read {snapshot['symbol']} tick/info")
            spread = require_spread(tick, float(info.point), args.max_spread_points)
            entry = float(tick.ask if signal["side"] == "BUY" else tick.bid)
            risk_override = args.take_profit_percent is not None
            if risk_override:
                tp_distance = entry * float(args.take_profit_percent) / 100.0
                sl_distance = entry * float(args.stop_loss_percent) / 100.0
            elif policy.mode == "atr":
                tp_distance = signal["atr14_abs"] * policy.take_profit_atr
                sl_distance = signal["atr14_abs"] * policy.stop_loss_atr
            else:
                tp_distance, sl_distance = entry * cfg.take_profit_pct, entry * cfg.stop_loss_pct
            tp = entry + tp_distance if signal["side"] == "BUY" else entry - tp_distance
            sl = entry - sl_distance if signal["side"] == "BUY" else entry + sl_distance
            if args.enable_demo_orders:
                result = open_demo_position(mt5, symbol=snapshot["symbol"], side=signal["side"], volume=float(cfg.fixed_lot), stop_loss=sl, take_profit=tp, deviation_points=args.deviation_points)
                action = {"action": "DEMO_ORDER_SENT", "retcode": int(result.retcode), "order": int(result.order), "deal": int(result.deal), "spread_points": spread}
                state["last_order_signal_utc"] = signal["bar_time"]
                state["position_entry_bar_utc"] = signal["bar_time"]
                state["position_bars_elapsed"] = 0
            else:
                action = {"action": "DRY_RUN_SIGNAL", "message": "Add --enable-demo-orders to permit DEMO orders", "spread_points": spread, "sl": sl, "tp": tp}
        else:
            action = {"action": "NO_ORDER"}
        state["last_processed_bar_utc"] = signal["bar_time"]
        save_demo_state(args.state, state)
        active_thresholds = {
            "buy": float(bundle["buy_threshold"] if args.buy_threshold is None else args.buy_threshold),
            "sell": float(bundle["sell_threshold"] if args.sell_threshold is None else args.sell_threshold),
            "sell_meta": float(bundle["meta_gate_threshold"] if args.meta_threshold is None else args.meta_threshold),
        }
        active_thresholds["frozen_candidate_unchanged"] = (
            abs(active_thresholds["buy"] - float(bundle["buy_threshold"])) < 1e-12
            and abs(active_thresholds["sell"] - float(bundle["sell_threshold"])) < 1e-12
            and abs(active_thresholds["sell_meta"] - float(bundle["meta_gate_threshold"])) < 1e-12
        )
        active_risk = {
            "mode": "HMI_FIXED_PERCENT" if args.take_profit_percent is not None else f"BUNDLE_{str(policy.mode).upper()}",
            "take_profit_percent": args.take_profit_percent,
            "stop_loss_percent": args.stop_loss_percent,
            "bundle_exit_policy": getattr(policy, "policy_id", None),
        }
        outcome = {"account_login": int(account.login), "account_trade_mode": "DEMO", "orders_enabled": args.enable_demo_orders, "signal": signal, "execution": action, "thresholds": active_thresholds, "risk": active_risk, "safety": {"daily_loss_pct": state["daily_loss_pct"], "drawdown_pct": state["drawdown_pct"], "entry_block": equity_block, "max_positions": 1, "fixed_lot": cfg.fixed_lot}}
        append_demo_event(args.events, outcome)
        return outcome


def main() -> None:
    args = _arguments()
    if args.bars < 1000:
        raise ValueError("--bars must be >=1000")
    if not 0.0 <= args.bar_open_delay_seconds <= 30.0:
        raise ValueError("--bar-open-delay-seconds must be between 0 and 30")
    if (args.take_profit_percent is None) != (args.stop_loss_percent is None):
        raise ValueError("TP and SL overrides must be supplied together")
    for name, value in (("take-profit", args.take_profit_percent), ("stop-loss", args.stop_loss_percent)):
        if value is not None and not 0.01 <= value <= 20.0:
            raise ValueError(f"--{name}-percent must be between 0.01 and 20.0")
    for name, value in (("buy", args.buy_threshold), ("sell", args.sell_threshold), ("meta", args.meta_threshold)):
        if value is not None and not 0.50 <= value <= 0.99:
            raise ValueError(f"--{name}-threshold must be between 0.50 and 0.99")
    print("V5.1 MT5 DEMO ONLY — REAL ACCOUNTS ARE HARD-BLOCKED")
    bundle, _manifest = load_verified_bundle(args.bundle)
    timeframe = str(bundle["snapshot_manifest"].get("timeframe", "M15")).upper()
    if timeframe not in TIMEFRAME_SECONDS:
        raise ValueError(f"Unsupported execution timeframe: {timeframe}")
    period_seconds = TIMEFRAME_SECONDS[timeframe]
    lock_path = args.state.with_suffix(args.state.suffix + ".lock")
    try:
        with single_instance_lock(lock_path):
            while True:
                if not args.once:
                    now = time.time()
                    next_open = (math.floor(now / period_seconds) + 1) * period_seconds
                    wait_seconds = max(0.0, next_open + args.bar_open_delay_seconds - now)
                    print(
                        f"Waiting {wait_seconds:.1f}s for next {timeframe} candle open "
                        f"(+{args.bar_open_delay_seconds:g}s data delay)",
                        flush=True,
                    )
                    time.sleep(wait_seconds)
                try:
                    print(json.dumps(_cycle(args), indent=2, default=str))
                except DemoSafetyError as exc:
                    print(json.dumps({"status": "SAFETY_BLOCK", "reason": str(exc)}, indent=2))
                    raise SystemExit(2) from exc
                if args.once:
                    break
    except SingleInstanceError as exc:
        print(json.dumps({"status": "INSTANCE_BLOCK", "reason": str(exc)}, indent=2))
        raise SystemExit(3) from exc


if __name__ == "__main__":
    main()
