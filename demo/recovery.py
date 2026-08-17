from __future__ import annotations


class TransientMT5Error(RuntimeError):
    """An MT5/terminal failure that may clear after connectivity returns."""


class OrderStatusUnknownError(TransientMT5Error):
    """The terminal did not confirm whether an order request was processed."""
