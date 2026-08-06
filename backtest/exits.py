from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtest.engine import BacktestConfig, TradeRecord, _simulate_trade


@dataclass(frozen=True)
class ExitPolicy:
    policy_id: str
    mode: str
    horizon: int
    take_profit_atr: float | None = None
    stop_loss_atr: float | None = None


FIXED_EXIT = ExitPolicy(policy_id="fixed_pct_8", mode="fixed_pct", horizon=8)
EXIT_POLICIES = (
    FIXED_EXIT,
    ExitPolicy("atr_tp3_sl1_5_h12", "atr", 12, take_profit_atr=3.0, stop_loss_atr=1.5),
    ExitPolicy("atr_tp3_sl2_h16", "atr", 16, take_profit_atr=3.0, stop_loss_atr=2.0),
    ExitPolicy("atr_tp4_sl2_h24", "atr", 24, take_profit_atr=4.0, stop_loss_atr=2.0),
)


def simulate_trade_with_exit_policy(
    df: pd.DataFrame,
    signal_index: int,
    side: str,
    buy_probability: float,
    sell_probability: float,
    cfg: BacktestConfig,
    balance: float,
    policy: ExitPolicy,
) -> tuple[TradeRecord, int]:
    if policy.mode == "fixed_pct":
        return _simulate_trade(
            df=df,
            signal_index=signal_index,
            side=side,
            buy_probability=buy_probability,
            sell_probability=sell_probability,
            cfg=cfg,
            balance=balance,
        )
    if policy.mode != "atr":
        raise ValueError(f"Unknown exit mode: {policy.mode}")
    if policy.take_profit_atr is None or policy.stop_loss_atr is None:
        raise ValueError("ATR exit policy requires TP and SL multipliers")
    if "atr14_abs" not in df.columns:
        raise ValueError("ATR exit policy requires atr14_abs")

    entry_index = signal_index + 1
    entry_row = df.iloc[entry_index]
    entry_price = float(entry_row["open"])
    atr = float(df.iloc[signal_index]["atr14_abs"])
    if not pd.notna(atr) or atr <= 0.0:
        raise ValueError("ATR must be positive at the signal bar")

    tp_distance = atr * policy.take_profit_atr
    sl_distance = atr * policy.stop_loss_atr
    if side == "BUY":
        tp_price = entry_price + tp_distance
        sl_price = entry_price - sl_distance
    else:
        tp_price = entry_price - tp_distance
        sl_price = entry_price + sl_distance

    exit_index = min(entry_index + policy.horizon - 1, len(df) - 1)
    exit_price = float(df.iloc[exit_index]["close"])
    exit_reason = "TIME"
    for j in range(entry_index, min(entry_index + policy.horizon, len(df))):
        row = df.iloc[j]
        high = float(row["high"])
        low = float(row["low"])
        if side == "BUY":
            hit_tp = high >= tp_price
            hit_sl = low <= sl_price
        else:
            hit_tp = low <= tp_price
            hit_sl = high >= sl_price
        if hit_tp and hit_sl:
            exit_index, exit_price, exit_reason = j, sl_price, "SL_AMBIGUOUS"
            break
        if hit_tp:
            exit_index, exit_price, exit_reason = j, tp_price, "TP"
            break
        if hit_sl:
            exit_index, exit_price, exit_reason = j, sl_price, "SL"
            break

    direction = 1.0 if side == "BUY" else -1.0
    gross_pnl = (exit_price - entry_price) * direction * cfg.contract_size * cfg.fixed_lot
    spread_points = float(entry_row.get("spread", 0.0))
    spread_cost = spread_points * cfg.point * cfg.contract_size * cfg.fixed_lot
    slippage_cost = 2.0 * cfg.slippage_points * cfg.point * cfg.contract_size * cfg.fixed_lot
    commission_cost = cfg.commission_per_lot_round_turn * cfg.fixed_lot
    net_pnl = gross_pnl - spread_cost - slippage_cost - commission_cost
    balance_after = balance + net_pnl
    return TradeRecord(
        signal_time=str(df.iloc[signal_index]["time"]),
        entry_time=str(entry_row["time"]),
        exit_time=str(df.iloc[exit_index]["time"]),
        side=side,
        buy_probability=buy_probability,
        sell_probability=sell_probability,
        entry_price=entry_price,
        exit_price=exit_price,
        exit_reason=exit_reason,
        bars_held=exit_index - entry_index + 1,
        gross_pnl=gross_pnl,
        spread_cost=spread_cost,
        slippage_cost=slippage_cost,
        commission_cost=commission_cost,
        net_pnl=net_pnl,
        balance_after=balance_after,
    ), exit_index
