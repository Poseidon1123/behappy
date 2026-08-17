from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


class DemoSafetyError(RuntimeError):
    pass


@dataclass(frozen=True)
class SafetyLimits:
    max_daily_loss_pct: float = 2.0
    max_drawdown_pct: float = 5.0
    max_spread_points: float = 100.0


def require_demo_account(mt5: Any, account: Any, terminal: Any) -> None:
    demo_mode = getattr(mt5, "ACCOUNT_TRADE_MODE_DEMO", None)
    if demo_mode is None or int(account.trade_mode) != int(demo_mode):
        raise DemoSafetyError("REFUSED: MT5 account is not ACCOUNT_TRADE_MODE_DEMO")
    if not bool(getattr(account, "trade_allowed", False)):
        raise DemoSafetyError("REFUSED: trading is disabled for this demo account")
    if not bool(getattr(terminal, "trade_allowed", False)):
        raise DemoSafetyError("REFUSED: Algo Trading is disabled in the MT5 terminal")


def require_position_mode(mt5: Any, account: Any, max_positions: int) -> None:
    if int(max_positions) <= 1:
        return
    hedging_mode = getattr(mt5, "ACCOUNT_MARGIN_MODE_RETAIL_HEDGING", None)
    if hedging_mode is None or int(account.margin_mode) != int(hedging_mode):
        raise DemoSafetyError(
            "REFUSED: max_positions > 1 requires an MT5 hedging account; netting accounts merge positions"
        )


def refresh_equity_limits(
    state: dict[str, Any],
    *,
    equity: float,
    limits: SafetyLimits,
    now: datetime | None = None,
) -> str | None:
    current = now or datetime.now(timezone.utc)
    day = current.date().isoformat()
    if state.get("equity_day_utc") != day:
        state["equity_day_utc"] = day
        state["day_start_equity"] = float(equity)
    state["peak_equity"] = max(float(state.get("peak_equity", equity)), float(equity))
    daily_loss = 100.0 * (float(state["day_start_equity"]) - equity) / max(float(state["day_start_equity"]), 1e-12)
    drawdown = 100.0 * (float(state["peak_equity"]) - equity) / max(float(state["peak_equity"]), 1e-12)
    state["daily_loss_pct"] = daily_loss
    state["drawdown_pct"] = drawdown
    if daily_loss >= limits.max_daily_loss_pct:
        return f"daily equity loss {daily_loss:.3f}% reached limit"
    if drawdown >= limits.max_drawdown_pct:
        return f"equity drawdown {drawdown:.3f}% reached limit"
    return None


def require_spread(tick: Any, point: float, limit_points: float) -> float:
    spread = (float(tick.ask) - float(tick.bid)) / float(point)
    if spread < 0.0 or spread > limit_points:
        raise DemoSafetyError(f"REFUSED: spread {spread:.1f} points exceeds {limit_points:.1f}")
    return spread


def require_volume(volume: float, symbol_info: Any) -> float:
    requested = float(volume)
    minimum = float(symbol_info.volume_min)
    maximum = float(symbol_info.volume_max)
    step = float(symbol_info.volume_step)
    if step <= 0.0:
        raise DemoSafetyError("REFUSED: broker returned an invalid volume step")
    if requested < minimum - 1e-12 or requested > maximum + 1e-12:
        raise DemoSafetyError(
            f"REFUSED: lot {requested:g} is outside broker range {minimum:g}..{maximum:g}"
        )
    steps = round((requested - minimum) / step)
    normalized = minimum + steps * step
    if abs(normalized - requested) > max(1e-9, step * 1e-6):
        raise DemoSafetyError(
            f"REFUSED: lot {requested:g} does not match broker step {step:g}"
        )
    return round(normalized, 8)
