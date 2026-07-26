from __future__ import annotations

import sys
from pathlib import Path

import yaml
from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow


def load_config(path: str = "config/config.yaml") -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    config = load_config()

    app = QApplication(sys.argv)
    app.setApplicationName("AI MT5 Trading Bot")

    window = MainWindow(config)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
