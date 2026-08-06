from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from backtest.drift import drift_score
from backtest.meta_labeling_v41 import _meta_vector
from backtest.nested_robust import _outer_decision
from data.feature_engineering import build_features


EVENT_FIELDS = [
    "event_time_utc", "bar_time_utc", "event_type", "side", "reason",
    "buy_probability", "sell_probability", "meta_probability", "drift_score",
    "entry_price", "exit_price", "take_profit", "stop_loss", "net_pnl",
    "balance", "bundle_sha256",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_verified_bundle(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle_path = Path(path)
    manifest_path = bundle_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = _sha256(bundle_path)
    if actual != manifest.get("bundle_sha256"):
        raise ValueError(f"Bundle SHA-256 mismatch: expected {manifest.get('bundle_sha256')}, got {actual}")
    bundle = joblib.load(bundle_path)
    if bundle.get("execution_mode") != "SHADOW_ONLY" or bundle.get("deployment_allowed") is not False:
        raise ValueError("Refusing bundle unless execution_mode=SHADOW_ONLY and deployment_allowed=false")
    return bundle, manifest


def _state_default(bundle: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    cfg = bundle["backtest_config"]
    return {
        "state_version": 1,
        "execution_mode": "SHADOW_ONLY",
        "bundle_sha256": manifest["bundle_sha256"],
        "balance": float(cfg.initial_balance),
        "last_processed_bar_utc": bundle["training_cutoff_utc"],
        "pending_signal": None,
        "open_position": None,
        "trades": 0,
        "wins": 0,
        "losses": 0,
        "net_profit": 0.0,
        "drift_score": None,
        "drift_bars_remaining": 0,
    }


def load_state(path: str | Path, bundle: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    state_path = Path(path)
    if not state_path.exists():
        return _state_default(bundle, manifest)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("execution_mode") != "SHADOW_ONLY":
        raise ValueError("State is not SHADOW_ONLY")
    if state.get("bundle_sha256") != manifest["bundle_sha256"]:
        raise ValueError("State belongs to another frozen bundle; use a new state path")
    return state


def save_state(path: str | Path, state: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(temporary, target)


def _event(path: str | Path, manifest: dict[str, Any], **values: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    row = {key: values.get(key) for key in EVENT_FIELDS}
    row["event_time_utc"] = datetime.now(timezone.utc).isoformat()
    row["bundle_sha256"] = manifest["bundle_sha256"]
    exists = target.exists() and target.stat().st_size > 0
    with target.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _open_pending(state: dict[str, Any], row: pd.Series, bundle: dict[str, Any]) -> None:
    signal = state["pending_signal"]
    if signal is None:
        return
    cfg, policy = bundle["backtest_config"], bundle["exit_policy"]
    entry = float(row["open"])
    if policy.mode == "atr":
        tp_distance = float(signal["atr14_abs"]) * float(policy.take_profit_atr)
        sl_distance = float(signal["atr14_abs"]) * float(policy.stop_loss_atr)
    else:
        tp_distance = entry * float(cfg.take_profit_pct)
        sl_distance = entry * float(cfg.stop_loss_pct)
    side = signal["side"]
    state["open_position"] = {
        **signal,
        "entry_time": row["time"].isoformat(),
        "entry_price": entry,
        "take_profit": entry + tp_distance if side == "BUY" else entry - tp_distance,
        "stop_loss": entry - sl_distance if side == "BUY" else entry + sl_distance,
        "spread_points": float(row.get("spread", 0.0)),
        "bars_held": 0,
        "horizon": int(policy.horizon),
    }
    state["pending_signal"] = None


def _position_exit(position: dict[str, Any], row: pd.Series) -> tuple[str, float] | None:
    high, low = float(row["high"]), float(row["low"])
    tp, sl, side = position["take_profit"], position["stop_loss"], position["side"]
    hit_tp = high >= tp if side == "BUY" else low <= tp
    hit_sl = low <= sl if side == "BUY" else high >= sl
    if hit_tp and hit_sl:
        return "SL_AMBIGUOUS", sl
    if hit_tp:
        return "TP", tp
    if hit_sl:
        return "SL", sl
    if position["bars_held"] >= position["horizon"]:
        return "TIME", float(row["close"])
    return None


def _close_position(state: dict[str, Any], exit_price: float, cfg: Any) -> float:
    pos = state["open_position"]
    direction = 1.0 if pos["side"] == "BUY" else -1.0
    gross = (exit_price - pos["entry_price"]) * direction * cfg.contract_size * cfg.fixed_lot
    spread = pos["spread_points"] * cfg.point * cfg.contract_size * cfg.fixed_lot
    slippage = 2.0 * cfg.slippage_points * cfg.point * cfg.contract_size * cfg.fixed_lot
    commission = cfg.commission_per_lot_round_turn * cfg.fixed_lot
    net = gross - spread - slippage - commission
    state["balance"] += net
    state["net_profit"] += net
    state["trades"] += 1
    state["wins" if net > 0 else "losses"] += 1
    return float(net)


def _probabilities(bundle: dict[str, Any], row: pd.Series) -> tuple[float, float]:
    buy = float(bundle["buy_model"].predict_proba(pd.DataFrame([row[bundle["buy_features"]]]))[:, 1][0])
    sell = float(bundle["sell_model"].predict_proba(pd.DataFrame([row[bundle["sell_features"]]]))[:, 1][0])
    return buy, sell


def process_closed_bars(
    new_raw_bars: pd.DataFrame,
    bundle: dict[str, Any],
    manifest: dict[str, Any],
    state: dict[str, Any],
    events_path: str | Path,
    state_path: str | Path,
) -> dict[str, Any]:
    raw = pd.concat([bundle["raw_context_tail"], new_raw_bars], ignore_index=True)
    fetched_times = pd.to_datetime(new_raw_bars["time"], utc=True)
    last = pd.Timestamp(state["last_processed_bar_utc"])
    if fetched_times.empty:
        raise ValueError("MT5 returned no closed bars")
    if fetched_times.min() > last:
        raise ValueError(
            "Fetched MT5 history does not overlap the last processed bar. "
            "Increase --bars before continuing; shadow state was left unchanged."
        )
    raw["time"] = pd.to_datetime(raw["time"], utc=True)
    raw.sort_values("time", inplace=True)
    raw.drop_duplicates("time", keep="last", inplace=True)
    timeframe = str(bundle["snapshot_manifest"].get("timeframe", "M15"))
    featured = build_features(raw, base_timeframe=timeframe).reset_index(drop=True)
    indices = featured.index[featured["time"] > last].tolist()
    processed = 0
    for index in indices:
        row = featured.iloc[index]
        bar_time = row["time"].isoformat()
        occupied_at_bar_open = state["pending_signal"] is not None or state["open_position"] is not None
        if state["pending_signal"] is not None and state["open_position"] is None:
            _open_pending(state, row, bundle)
            pos = state["open_position"]
            _event(events_path, manifest, bar_time_utc=bar_time, event_type="ENTRY", side=pos["side"], reason="NEXT_BAR_OPEN", entry_price=pos["entry_price"], take_profit=pos["take_profit"], stop_loss=pos["stop_loss"], balance=state["balance"])

        if state["open_position"] is not None:
            state["open_position"]["bars_held"] += 1
            outcome = _position_exit(state["open_position"], row)
            if outcome is not None:
                reason, exit_price = outcome
                pos = dict(state["open_position"])
                net = _close_position(state, exit_price, bundle["backtest_config"])
                _event(events_path, manifest, bar_time_utc=bar_time, event_type="EXIT", side=pos["side"], reason=reason, buy_probability=pos["buy_probability"], sell_probability=pos["sell_probability"], meta_probability=pos.get("meta_probability"), drift_score=pos.get("drift_score"), entry_price=pos["entry_price"], exit_price=exit_price, take_profit=pos["take_profit"], stop_loss=pos["stop_loss"], net_pnl=net, balance=state["balance"])
                state["open_position"] = None

        if state["drift_bars_remaining"] <= 0:
            start = max(0, index - int(bundle["drift_recent_window"]))
            score = drift_score(bundle["drift_reference"], featured.iloc[start:index])
            state["drift_score"] = score if math.isfinite(score) else None
            state["drift_bars_remaining"] = int(bundle["drift_score_step"])
        state["drift_bars_remaining"] -= 1

        reason, side, buy_p, sell_p, meta_p = "POSITION_ACTIVE_OR_EXITED", "HOLD", None, None, None
        required = bundle["buy_features"] + bundle["sell_features"] + ["atr14_abs"]
        if not occupied_at_bar_open and state["open_position"] is None and state["pending_signal"] is None and row[required].notna().all():
            score = state["drift_score"]
            if score is None or score > bundle["drift_cutoff"]:
                reason = "DRIFT_BLOCK"
            else:
                buy_p, sell_p = _probabilities(bundle, row)
                side = _outer_decision(buy_p, sell_p, buy_threshold=bundle["buy_threshold"], sell_threshold=bundle["sell_threshold"], min_probability_edge=bundle["min_probability_edge"])
                reason = "PRIMARY_HOLD" if side == "HOLD" else "PRIMARY_PASS"
                if side == "SELL":
                    vector = _meta_vector(row, side="SELL", primary=sell_p, opposing=buy_p, primary_threshold=bundle["sell_threshold"])
                    meta_p = float(bundle["sell_meta_model"].predict_proba(vector)[:, 1][0])
                    if meta_p < bundle["meta_gate_threshold"]:
                        side, reason = "HOLD", "SELL_META_BLOCK"
                if side in {"BUY", "SELL"}:
                    state["pending_signal"] = {"signal_time": bar_time, "side": side, "buy_probability": buy_p, "sell_probability": sell_p, "meta_probability": meta_p, "drift_score": score, "atr14_abs": float(row["atr14_abs"])}
                    reason = "BUY_PRIMARY" if side == "BUY" else "SELL_META_PASS"
        _event(events_path, manifest, bar_time_utc=bar_time, event_type="DECISION", side=side, reason=reason, buy_probability=buy_p, sell_probability=sell_p, meta_probability=meta_p, drift_score=state["drift_score"], balance=state["balance"])
        state["last_processed_bar_utc"] = bar_time
        save_state(state_path, state)
        processed += 1
    return {"processed_bars": processed, "balance": state["balance"], "trades": state["trades"], "wins": state["wins"], "losses": state["losses"], "net_profit": state["net_profit"], "pending_signal": state["pending_signal"] is not None, "open_position": state["open_position"] is not None}
