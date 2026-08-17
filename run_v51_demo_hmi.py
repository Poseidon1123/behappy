from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the v5.1 MT5 Demo control HMI")
    parser.add_argument("--bundle", type=Path, default=Path("models/v51_shadow_bundle.joblib"))
    args = parser.parse_args()
    from PySide6.QtWidgets import QApplication, QMessageBox

    app = QApplication(sys.argv)
    app.setApplicationName("v5.1 Demo Trading HMI")
    try:
        from gui.demo_hmi import DemoHMI

        window = DemoHMI(args.bundle)
        window.show()
        raise SystemExit(app.exec())
    except Exception as exc:
        QMessageBox.critical(None, "HMI startup error", str(exc))
        raise


if __name__ == "__main__":
    main()
