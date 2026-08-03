from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = ["time", "open", "high", "low", "close", "spread"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_bars(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Snapshot is missing required columns: {missing}")

    result = frame.copy()
    result["time"] = pd.to_datetime(result["time"], utc=True, errors="raise")
    result.sort_values("time", inplace=True)
    result.reset_index(drop=True, inplace=True)
    if result.empty:
        raise ValueError("Snapshot contains no bars")
    if result["time"].duplicated().any():
        raise ValueError("Snapshot contains duplicate candle timestamps")
    if not result["time"].is_monotonic_increasing:
        raise ValueError("Snapshot timestamps are not strictly increasing")
    numeric = ["open", "high", "low", "close", "spread"]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="raise")
    if result[numeric].isna().any(axis=None):
        raise ValueError("Snapshot contains missing OHLC/spread values")
    if (result[["open", "high", "low", "close"]] <= 0).any(axis=None):
        raise ValueError("Snapshot contains non-positive prices")
    return result


def save_snapshot(
    frame: pd.DataFrame,
    csv_path: str | Path,
    *,
    symbol: str,
    timeframe: str,
    point: float,
    contract_size: float,
) -> tuple[Path, Path, dict[str, Any]]:
    data = _validate_bars(frame)
    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file = csv_file.with_suffix(".manifest.json")

    export = data.copy()
    export["time"] = export["time"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    export.to_csv(csv_file, index=False, float_format="%.10g", lineterminator="\n")
    digest = _sha256(csv_file)
    manifest = {
        "schema_version": 1,
        "symbol": symbol,
        "timeframe": timeframe.upper(),
        "closed_bars_only": True,
        "rows": len(export),
        "first_bar_utc": data.iloc[0]["time"].isoformat(),
        "last_bar_utc": data.iloc[-1]["time"].isoformat(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": digest,
        "columns": list(export.columns),
        "point": float(point),
        "contract_size": float(contract_size),
    }
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return csv_file, manifest_file, manifest


def load_snapshot(
    csv_path: str | Path,
    manifest_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    csv_file = Path(csv_path)
    manifest_file = Path(manifest_path) if manifest_path else csv_file.with_suffix(".manifest.json")
    if not csv_file.is_file():
        raise FileNotFoundError(f"Snapshot CSV not found: {csv_file}")
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Snapshot manifest not found: {manifest_file}")

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    actual_hash = _sha256(csv_file)
    if actual_hash != manifest.get("sha256"):
        raise ValueError(
            "Snapshot SHA-256 mismatch. The CSV changed after it was frozen: "
            f"expected {manifest.get('sha256')}, got {actual_hash}"
        )
    data = _validate_bars(pd.read_csv(csv_file))
    if len(data) != int(manifest.get("rows", -1)):
        raise ValueError("Snapshot row count does not match its manifest")
    if data.iloc[0]["time"].isoformat() != manifest.get("first_bar_utc"):
        raise ValueError("Snapshot first timestamp does not match its manifest")
    if data.iloc[-1]["time"].isoformat() != manifest.get("last_bar_utc"):
        raise ValueError("Snapshot last timestamp does not match its manifest")
    return data, manifest
