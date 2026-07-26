from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import MetaTrader5 as mt5
from dotenv import load_dotenv


class MT5ConnectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class AccountSnapshot:
    login: int
    server: str
    currency: str
    balance: float
    equity: float
    margin: float
    margin_free: float
    profit: float


class MT5Connector:
    """Owns the lifecycle of the MetaTrader 5 Python connection."""

    def __init__(self) -> None:
        load_dotenv()
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        path = os.getenv("MT5_PATH", "").strip()
        login_raw = os.getenv("MT5_LOGIN", "").strip()
        password = os.getenv("MT5_PASSWORD", "").strip()
        server = os.getenv("MT5_SERVER", "").strip()

        kwargs: dict[str, Any] = {}

        if login_raw:
            try:
                kwargs["login"] = int(login_raw)
            except ValueError as exc:
                raise MT5ConnectionError(
                    "MT5_LOGIN must be an integer account number."
                ) from exc

        if password:
            kwargs["password"] = password
        if server:
            kwargs["server"] = server

        ok = mt5.initialize(path, **kwargs) if path else mt5.initialize(**kwargs)

        if not ok:
            raise MT5ConnectionError(
                f"Could not initialize MetaTrader 5. MT5 error: {mt5.last_error()}"
            )

        self._connected = True

        if mt5.account_info() is None:
            error = mt5.last_error()
            self.disconnect()
            raise MT5ConnectionError(
                "MT5 is connected but no account is logged in. "
                f"MT5 error: {error}"
            )

    def disconnect(self) -> None:
        if self._connected:
            mt5.shutdown()
            self._connected = False

    def terminal_info(self) -> dict[str, Any]:
        info = mt5.terminal_info()
        if info is None:
            raise MT5ConnectionError(
                f"Could not read terminal info. MT5 error: {mt5.last_error()}"
            )
        return info._asdict()

    def account_snapshot(self) -> AccountSnapshot:
        info = mt5.account_info()
        if info is None:
            raise MT5ConnectionError(
                f"Could not read account info. MT5 error: {mt5.last_error()}"
            )

        return AccountSnapshot(
            login=int(info.login),
            server=str(info.server),
            currency=str(info.currency),
            balance=float(info.balance),
            equity=float(info.equity),
            margin=float(info.margin),
            margin_free=float(info.margin_free),
            profit=float(info.profit),
        )

    def __enter__(self) -> "MT5Connector":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()
