from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd

from data.feature_engineering import FEATURE_COLUMNS, build_features


@dataclass(frozen=True)
class BacktestConfig:
    initial_balance: float = 10_000.0
    fixed_lot: float = 0.01
    buy_threshold: float = 0.72
    sell_threshold: float = 0.72
    min_probability_edge: float = 0.08
    horizon: int = 8
    take_profit_pct: float = 0.006
    stop_loss_pct: float = 0.003
    slippage_points: float = 2.0
    commission_per_lot_round_turn: float = 0.0
    point: float = 0.01
    contract_size: float = 100.0


@dataclass(frozen=True)
class TradeRecord:
    signal_time: str
    entry_time: str
    exit_time: str
    side: str
    buy_probability: float
    sell_probability: float
    entry_price: float
    exit_price: float
    exit_reason: str
    bars_held: int
    gross_pnl: float
    spread_cost: float
    slippage_cost: float
    commission_cost: float
    net_pnl: float
    balance_after: float


def _validate_config(cfg: BacktestConfig) -> None:
    if cfg.initial_balance <= 0:
        raise ValueError("initial_balance must be positive")
    if cfg.fixed_lot <= 0:
        raise ValueError("fixed_lot must be positive")
    if cfg.horizon <= 0:
        raise ValueError("horizon must be positive")
    if cfg.take_profit_pct <= 0 or cfg.stop_loss_pct <= 0:
        raise ValueError("TP and SL must be positive")
    if cfg.point <= 0 or cfg.contract_size <= 0:
        raise ValueError("point and contract_size must be positive")


def _decision(buy: float, sell: float, cfg: BacktestConfig) -> str:
    buy_ok = buy >= cfg.buy_threshold
    sell_ok = sell >= cfg.sell_threshold

    if buy_ok and not sell_ok:
        return "BUY"
    if sell_ok and not buy_ok:
        return "SELL"
    if buy_ok and sell_ok:
        if buy - sell >= cfg.min_probability_edge:
            return "BUY"
        if sell - buy >= cfg.min_probability_edge:
            return "SELL"
    return "HOLD"


def _simulate_trade(
    df: pd.DataFrame,
    signal_index: int,
    side: str,
    buy_probability: float,
    sell_probability: float,
    cfg: BacktestConfig,
    balance: float,
) -> tuple[TradeRecord, int]:
    """Enter at the next bar open and scan forward for first TP/SL touch.

    MT5 OHLC bars are commonly bid-based. Backtester v1 therefore treats the
    observed spread as an explicit round-trip transaction cost rather than
    pretending bid/ask OHLC are both available. If TP and SL are touched in the
    same bar, the result is conservatively treated as SL.
    """
    entry_index = signal_index + 1
    entry_row = df.iloc[entry_index]
    entry_price = float(entry_row["open"])

    if side == "BUY":
        tp_price = entry_price * (1.0 + cfg.take_profit_pct)
        sl_price = entry_price * (1.0 - cfg.stop_loss_pct)
    else:
        tp_price = entry_price * (1.0 - cfg.take_profit_pct)
        sl_price = entry_price * (1.0 + cfg.stop_loss_pct)

    exit_index = min(entry_index + cfg.horizon - 1, len(df) - 1)
    exit_price = float(df.iloc[exit_index]["close"])
    exit_reason = "TIME"

    for j in range(entry_index, min(entry_index + cfg.horizon, len(df))):
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
            # OHLC cannot reveal intrabar ordering. Be conservative.
            exit_index = j
            exit_price = sl_price
            exit_reason = "SL_AMBIGUOUS"
            break
        if hit_tp:
            exit_index = j
            exit_price = tp_price
            exit_reason = "TP"
            break
        if hit_sl:
            exit_index = j
            exit_price = sl_price
            exit_reason = "SL"
            break

    direction = 1.0 if side == "BUY" else -1.0
    price_move = (exit_price - entry_price) * direction
    gross_pnl = price_move * cfg.contract_size * cfg.fixed_lot

    spread_points = float(entry_row.get("spread", 0.0))
    spread_cost = spread_points * cfg.point * cfg.contract_size * cfg.fixed_lot
    slippage_cost = (
        2.0 * cfg.slippage_points * cfg.point * cfg.contract_size * cfg.fixed_lot
    )
    commission_cost = cfg.commission_per_lot_round_turn * cfg.fixed_lot
    net_pnl = gross_pnl - spread_cost - slippage_cost - commission_cost
    balance_after = balance + net_pnl

    trade = TradeRecord(
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
    )
    return trade, exit_index


def _summary(trades: pd.DataFrame, initial_balance: float) -> dict[str, Any]:
    if trades.empty:
        return {
            "initial_balance": initial_balance,
            "final_balance": initial_balance,
            "net_profit": 0.0,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "profit_factor": None,
            "expectancy": 0.0,
            "max_drawdown_pct": 0.0,
            "total_costs": 0.0,
        }

    pnl = trades["net_pnl"].astype(float)
    wins = int((pnl > 0).sum())
    losses = int((pnl <= 0).sum())
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None

    equity = pd.concat(
        [pd.Series([initial_balance]), trades["balance_after"].astype(float)],
        ignore_index=True,
    )
    peaks = equity.cummax()
    drawdown = (equity - peaks) / peaks.replace(0, np.nan)
    max_drawdown_pct = float(abs(drawdown.min()) * 100.0)

    total_costs = float(
        trades[["spread_cost", "slippage_cost", "commission_cost"]]
        .sum(axis=1)
        .sum()
    )

    return {
        "initial_balance": initial_balance,
        "final_balance": float(trades.iloc[-1]["balance_after"]),
        "net_profit": float(pnl.sum()),
        "trades": int(len(trades)),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(trades),
        "profit_factor": profit_factor,
        "expectancy": float(pnl.mean()),
        "max_drawdown_pct": max_drawdown_pct,
        "total_costs": total_costs,
        "average_bars_held": float(trades["bars_held"].mean()),
        "buy_trades": int((trades["side"] == "BUY").sum()),
        "sell_trades": int((trades["side"] == "SELL").sum()),
    }


def run_backtest(
    raw_df: pd.DataFrame,
    buy_model_path: str = "models/buy_model.joblib",
    sell_model_path: str = "models/sell_model.joblib",
    config: BacktestConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cfg = config or BacktestConfig()
    _validate_config(cfg)

    if len(raw_df) < 200:
        raise ValueError("Backtest needs at least 200 bars")

    buy_model = joblib.load(buy_model_path)
    sell_model = joblib.load(sell_model_path)

    df = build_features(raw_df).reset_index(drop=True)
    valid_mask = df[FEATURE_COLUMNS].notna().all(axis=1)
    valid_indices = np.flatnonzero(valid_mask.to_numpy())
    valid_indices = valid_indices[valid_indices < len(df) - cfg.horizon - 1]

    if len(valid_indices) == 0:
        raise ValueError("No usable feature rows for backtest")

    x = df.loc[valid_indices, FEATURE_COLUMNS]
    buy_prob = buy_model.predict_proba(x)[:, 1]
    sell_prob = sell_model.predict_proba(x)[:, 1]
    probability_by_index = {
        int(idx): (float(b), float(s))
        for idx, b, s in zip(valid_indices, buy_prob, sell_prob)
    }

    trades: list[TradeRecord] = []
    balance = cfg.initial_balance
    i = int(valid_indices[0])
    last_signal_index = int(valid_indices[-1])

    while i <= last_signal_index:
        probabilities = probability_by_index.get(i)
        if probabilities is None:
            i += 1
            continue

        buy, sell = probabilities
        side = _decision(buy, sell, cfg)
        if side == "HOLD":
            i += 1
            continue

        trade, exit_index = _simulate_trade(
            df=df,
            signal_index=i,
            side=side,
            buy_probability=buy,
            sell_probability=sell,
            cfg=cfg,
            balance=balance,
        )
        trades.append(trade)
        balance = trade.balance_after

        # One open position at a time. The next signal may occur only after exit.
        i = exit_index + 1

    trades_df = pd.DataFrame([asdict(t) for t in trades])

    if trades_df.empty:
        equity_df = pd.DataFrame(
            [{"time": str(df.iloc[valid_indices[0]]["time"]), "equity": cfg.initial_balance}]
        )
    else:
        equity_df = pd.DataFrame(
            {
                "time": trades_df["exit_time"],
                "equity": trades_df["balance_after"],
            }
        )

    return trades_df, equity_df, _summary(trades_df, cfg.initial_balance)
