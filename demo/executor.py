from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MAGIC = 26042026
COMMENT = "v51_demo_only"


def load_demo_state(path: str | Path, bundle_sha256: str) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {
            "state_version": 1,
            "execution_mode": "DEMO_ONLY",
            "bundle_sha256": bundle_sha256,
            "last_processed_bar_utc": None,
            "last_order_signal_utc": None,
            "position_entry_bar_utc": None,
            "position_bars_elapsed": 0,
            "position_bars_elapsed_by_ticket": {},
        }
    state = json.loads(target.read_text(encoding="utf-8"))
    if state.get("execution_mode") != "DEMO_ONLY":
        raise ValueError("Demo state has an unexpected execution mode")
    if state.get("bundle_sha256") != bundle_sha256:
        raise ValueError("Demo state belongs to another frozen bundle; use a new state path")
    state.setdefault("position_bars_elapsed_by_ticket", {})
    return state


def save_demo_state(path: str | Path, state: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(temporary, target)


def append_demo_event(path: str | Path, event: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {"event_time_utc": datetime.now(timezone.utc).isoformat(), **event}
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def bot_positions(mt5: Any, symbol: str) -> list[Any]:
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        raise RuntimeError(f"positions_get failed: {mt5.last_error()}")
    return [position for position in positions if int(position.magic) == MAGIC]


def _filling_type(mt5: Any, symbol_info: Any) -> int:
    if int(symbol_info.trade_exemode) == int(getattr(mt5, "SYMBOL_TRADE_EXECUTION_MARKET", -1)):
        flags = int(getattr(symbol_info, "filling_mode", 0))
        if flags & int(getattr(mt5, "SYMBOL_FILLING_IOC", 2)):
            return int(mt5.ORDER_FILLING_IOC)
        return int(mt5.ORDER_FILLING_FOK)
    return int(mt5.ORDER_FILLING_RETURN)


def _send_checked(mt5: Any, request: dict[str, Any]) -> Any:
    checked = mt5.order_check(request)
    if checked is None:
        raise RuntimeError(f"order_check returned None: {mt5.last_error()}")
    success_check = {0, int(getattr(mt5, "TRADE_RETCODE_DONE", 10009))}
    if int(checked.retcode) not in success_check:
        raise RuntimeError(f"order_check refused request: retcode={checked.retcode}, comment={checked.comment}")
    result = mt5.order_send(request)
    if result is None:
        raise RuntimeError(f"order_send returned None: {mt5.last_error()}")
    accepted = {
        int(getattr(mt5, "TRADE_RETCODE_DONE", 10009)),
        int(getattr(mt5, "TRADE_RETCODE_PLACED", 10008)),
        int(getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010)),
    }
    if int(result.retcode) not in accepted:
        raise RuntimeError(f"order_send failed: retcode={result.retcode}, comment={result.comment}")
    return result


def open_demo_position(
    mt5: Any,
    *,
    symbol: str,
    side: str,
    volume: float,
    stop_loss: float,
    take_profit: float,
    deviation_points: int,
) -> Any:
    info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if info is None or tick is None:
        raise RuntimeError(f"Could not read symbol/tick for {symbol}: {mt5.last_error()}")
    digits = int(info.digits)
    price = float(tick.ask if side == "BUY" else tick.bid)
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL,
        "price": round(price, digits),
        "sl": round(float(stop_loss), digits),
        "tp": round(float(take_profit), digits),
        "deviation": int(deviation_points),
        "magic": MAGIC,
        "comment": COMMENT,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": _filling_type(mt5, info),
    }
    return _send_checked(mt5, request)


def close_demo_position(mt5: Any, position: Any, deviation_points: int) -> Any:
    tick = mt5.symbol_info_tick(position.symbol)
    info = mt5.symbol_info(position.symbol)
    if tick is None or info is None:
        raise RuntimeError(f"Could not read symbol/tick for {position.symbol}: {mt5.last_error()}")
    closing_buy = int(position.type) == int(mt5.POSITION_TYPE_SELL)
    price = float(tick.ask if closing_buy else tick.bid)
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "position": int(position.ticket),
        "volume": float(position.volume),
        "type": mt5.ORDER_TYPE_BUY if closing_buy else mt5.ORDER_TYPE_SELL,
        "price": round(price, int(info.digits)),
        "deviation": int(deviation_points),
        "magic": MAGIC,
        "comment": COMMENT + "_time_exit",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": _filling_type(mt5, info),
    }
    return _send_checked(mt5, request)
