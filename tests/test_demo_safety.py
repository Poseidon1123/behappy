from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from demo.executor import _send_checked
from demo.lock import SingleInstanceError, single_instance_lock
from demo.safety import DemoSafetyError, SafetyLimits, refresh_equity_limits, require_demo_account, require_spread


class FakeMT5:
    ACCOUNT_TRADE_MODE_DEMO = 0
    TRADE_RETCODE_DONE = 10009


class DemoSafetyTests(unittest.TestCase):
    def test_real_account_is_hard_blocked(self) -> None:
        with self.assertRaises(DemoSafetyError):
            require_demo_account(
                FakeMT5,
                SimpleNamespace(trade_mode=2, trade_allowed=True),
                SimpleNamespace(trade_allowed=True),
            )

    def test_demo_account_is_allowed(self) -> None:
        require_demo_account(
            FakeMT5,
            SimpleNamespace(trade_mode=0, trade_allowed=True),
            SimpleNamespace(trade_allowed=True),
        )

    def test_equity_limit_blocks_entries_without_raising(self) -> None:
        state = {"equity_day_utc": "2026-08-03", "day_start_equity": 1000.0, "peak_equity": 1000.0}
        reason = refresh_equity_limits(
            state,
            equity=979.0,
            limits=SafetyLimits(max_daily_loss_pct=2.0),
            now=datetime(2026, 8, 3, tzinfo=timezone.utc),
        )
        self.assertIn("daily equity loss", reason or "")

    def test_wide_spread_is_blocked(self) -> None:
        with self.assertRaises(DemoSafetyError):
            require_spread(SimpleNamespace(ask=2501.5, bid=2500.0), 0.01, 100.0)

    def test_failed_order_check_never_calls_order_send(self) -> None:
        class RejectedMT5(FakeMT5):
            send_calls = 0

            @staticmethod
            def order_check(request):
                return SimpleNamespace(retcode=10016, comment="Invalid stops")

            @classmethod
            def order_send(cls, request):
                cls.send_calls += 1

            @staticmethod
            def last_error():
                return (1, "Success")

        with self.assertRaises(RuntimeError):
            _send_checked(RejectedMT5, {"symbol": "XAUUSD.sc"})
        self.assertEqual(RejectedMT5.send_calls, 0)

    def test_second_runner_is_blocked(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            lock_path = f"{directory}/demo.lock"
            with single_instance_lock(lock_path):
                with self.assertRaises(SingleInstanceError):
                    with single_instance_lock(lock_path):
                        pass


if __name__ == "__main__":
    unittest.main()
