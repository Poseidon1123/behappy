from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QProcess, QSettings, QTimer, Qt
from PySide6.QtGui import QColor, QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from demo.executor import MAGIC
from gui.main_window import APP_STYLE
from mt5.market_data import MarketData
from mt5.mt5_connector import MT5Connector
from shadow.engine import load_verified_bundle


class DemoHMI(QMainWindow):
    def __init__(self, bundle_path: str | Path = "models/v51_shadow_bundle.joblib") -> None:
        super().__init__()
        self.root_dir = Path.cwd()
        self.bundle_path = Path(bundle_path)
        self.bundle, self.manifest = load_verified_bundle(self.bundle_path)
        snapshot = self.bundle["snapshot_manifest"]
        self.symbol = str(snapshot["symbol"])
        self.execution_timeframe = str(snapshot["timeframe"])
        self.bundle_paths = {self.execution_timeframe.upper(): self.bundle_path}
        self.events_path = self.root_dir / "demo_logs" / "v51_demo_events.jsonl"
        self.state_path = self.root_dir / "demo_state" / "v51_demo_state.json"
        self.connector = MT5Connector()
        self.market = MarketData()
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_process_output)
        self.process.finished.connect(self._process_finished)
        self.process_mode: str | None = None
        self.building_timeframe: str | None = None
        self.last_event_line = ""
        self.ui_settings = QSettings("Behappy", "v51-demo-hmi")

        self.setWindowTitle("v5.1 XAUUSD Demo Bot — Control HMI")
        self.setMinimumSize(900, 600)
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(1400, 850)
        else:
            available = screen.availableGeometry()
            self.resize(min(1540, available.width()), min(940, available.height()))
        self.setStyleSheet(APP_STYLE)
        pg.setConfigOptions(antialias=True)
        self._build_ui()
        self._connect_mt5()

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_dashboard)
        self.refresh_timer.start(2500)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(6)

        header = QHBoxLayout()
        title = QLabel("v5.1 DEMO TRADING HMI")
        title.setStyleSheet("font-size:24px; font-weight:800;")
        self.account_status = QLabel("● MT5 DISCONNECTED")
        self.bot_status = QLabel("● BOT STOPPED")
        self.mode_status = QLabel("DEMO ONLY")
        self.mode_status.setStyleSheet("color:#38bdf8; font-weight:800;")
        header.addWidget(title)
        header.addStretch()
        self.symbol_model_label = QLabel(f"{self.symbol} • MODEL {self.execution_timeframe}")
        header.addWidget(self.symbol_model_label)
        header.addWidget(self.mode_status)
        header.addWidget(self.account_status)
        header.addWidget(self.bot_status)
        root.addLayout(header)

        cards = QGridLayout()
        self.balance = self._card("Balance")
        self.equity = self._card("Equity")
        self.floating = self._card("Floating P/L")
        self.bid = self._card("Bid")
        self.ask = self._card("Ask")
        self.spread = self._card("Spread points")
        for column, card in enumerate((self.balance, self.equity, self.floating, self.bid, self.ask, self.spread)):
            cards.addWidget(card[0], 0, column)
        root.addLayout(cards)

        upper = QSplitter(Qt.Horizontal)
        upper.addWidget(self._market_panel())
        upper.addWidget(self._control_panel())
        upper.setSizes([1000, 480])

        tabs = QTabWidget()
        tabs.addTab(self._positions_table(), "Open positions")
        tabs.addTab(self._deals_table(), "Bot deal history")
        tabs.addTab(self._events_panel(), "AI / execution events")

        self.main_splitter = QSplitter(Qt.Vertical)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(8)
        self.main_splitter.setStyleSheet(
            "QSplitter::handle{background:#334155;border-radius:3px;margin:1px;}"
            "QSplitter::handle:hover{background:#38bdf8;}"
        )
        self.main_splitter.addWidget(upper)
        self.main_splitter.addWidget(tabs)
        self.main_splitter.handle(1).setToolTip(
            "Drag up/down to resize market controls and trade/event tables"
        )
        self.main_splitter.setSizes([570, 300])
        saved_splitter = self.ui_settings.value("main_splitter_state")
        if saved_splitter is not None:
            self.main_splitter.restoreState(saved_splitter)
        root.addWidget(self.main_splitter, 1)

    def _card(self, caption: str) -> tuple[QFrame, QLabel]:
        frame = QFrame()
        frame.setStyleSheet("QFrame{background:#111827;border:1px solid #263247;border-radius:10px;}")
        layout = QVBoxLayout(frame)
        title = QLabel(caption)
        title.setStyleSheet("color:#94a3b8;font-size:12px;border:0;")
        value = QLabel("-")
        value.setStyleSheet("font-size:19px;font-weight:700;border:0;")
        layout.addWidget(title)
        layout.addWidget(value)
        return frame, value

    def _market_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Chart timeframe"))
        self.chart_timeframe = QComboBox()
        self.chart_timeframe.addItems(["M1", "M5", "M15", "M30", "H1", "H4"])
        self.chart_timeframe.setCurrentText(self.execution_timeframe)
        self.chart_timeframe.currentTextChanged.connect(lambda _text: self.refresh_dashboard())
        toolbar.addWidget(self.chart_timeframe)
        toolbar.addWidget(QLabel("Bot processing timeframe"))
        self.bot_timeframe = QComboBox()
        self.bot_timeframe.addItems(["M1", "M5", "M15", "M30", "H1", "H4"])
        self.bot_timeframe.setCurrentText(self.execution_timeframe)
        self.bot_timeframe.currentTextChanged.connect(self._change_model_timeframe)
        toolbar.addWidget(self.bot_timeframe)
        self.model_readiness = QLabel(f"{self.execution_timeframe} MODEL READY")
        self.model_readiness.setStyleSheet("color:#22c55e;font-weight:800;")
        toolbar.addWidget(self.model_readiness)
        toolbar.addWidget(QLabel("Training bars"))
        self.training_bars = QSpinBox()
        self.training_bars.setRange(15000, 100000)
        self.training_bars.setSingleStep(5000)
        self.training_bars.setValue(30000)
        toolbar.addWidget(self.training_bars)
        self.build_model_button = QPushButton("BUILD SELECTED MODEL")
        self.build_model_button.clicked.connect(self.build_selected_model)
        toolbar.addWidget(self.build_model_button)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        self.chart = pg.PlotWidget()
        self.chart.showGrid(x=True, y=True, alpha=0.18)
        self.chart.setLabel("left", "Price")
        self.price_curve = self.chart.plot([], [], pen=pg.mkPen("#38bdf8", width=2))
        layout.addWidget(self.chart, 3)

        self.candles = QTableWidget(0, 7)
        self.candles.setHorizontalHeaderLabels(["UTC time", "Open", "High", "Low", "Close", "Volume", "Spread"])
        self.candles.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.candles.setAlternatingRowColors(True)
        layout.addWidget(self.candles, 2)
        return panel

    def _control_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        signal = QGroupBox("Latest frozen-model signal")
        grid = QGridLayout(signal)
        self.signal_side = QLabel("HOLD")
        self.signal_side.setAlignment(Qt.AlignCenter)
        self.signal_side.setStyleSheet("font-size:28px;font-weight:900;padding:10px;")
        self.buy_probability = QLabel("BUY: -")
        self.sell_probability = QLabel("SELL: -")
        self.meta_probability = QLabel("SELL meta: -")
        self.drift_score = QLabel("Drift: -")
        self.signal_reason = QLabel("Reason: waiting")
        grid.addWidget(self.signal_side, 0, 0, 1, 2)
        grid.addWidget(self.buy_probability, 1, 0)
        grid.addWidget(self.sell_probability, 1, 1)
        grid.addWidget(self.meta_probability, 2, 0)
        grid.addWidget(self.drift_score, 2, 1)
        grid.addWidget(self.signal_reason, 3, 0, 1, 2)
        layout.addWidget(signal)

        thresholds = QGroupBox("Decision thresholds")
        form = QFormLayout(thresholds)
        self.buy_threshold = self._spin(0.50, 0.99, self.bundle["buy_threshold"], 0.01, 2)
        self.sell_threshold = self._spin(0.50, 0.99, self.bundle["sell_threshold"], 0.01, 2)
        self.meta_threshold = self._spin(0.50, 0.99, self.bundle["meta_gate_threshold"], 0.01, 2)
        for spin in (self.buy_threshold, self.sell_threshold, self.meta_threshold):
            spin.valueChanged.connect(lambda _value: self._threshold_status())
        self.threshold_note = QLabel("Frozen v5.1 thresholds")
        self.threshold_note.setWordWrap(True)
        form.addRow("BUY probability", self.buy_threshold)
        form.addRow("SELL probability", self.sell_threshold)
        form.addRow("SELL meta gate", self.meta_threshold)
        form.addRow(self.threshold_note)
        layout.addWidget(thresholds)

        safety = QGroupBox("Demo safety")
        risk_form = QFormLayout(safety)
        self.max_spread = self._spin(1, 1000, 100, 5, 0)
        self.daily_loss = self._spin(0.1, 20, 2, 0.1, 1)
        self.max_drawdown = self._spin(0.5, 50, 5, 0.5, 1)
        self.override_tp_sl = QCheckBox("Override model TP / SL (DEMO experimental)")
        default_cfg = self.bundle["backtest_config"]
        self.take_profit_percent = self._spin(0.01, 20.0, float(default_cfg.take_profit_pct) * 100.0, 0.05, 2)
        self.stop_loss_percent = self._spin(0.01, 20.0, float(default_cfg.stop_loss_pct) * 100.0, 0.05, 2)
        self.bar_open_delay = self._spin(0.0, 30.0, 2.0, 0.5, 1)
        self.override_tp_sl.toggled.connect(self._update_tp_sl_controls)
        self.demo_orders = QCheckBox("ENABLE DEMO ORDERS")
        self.demo_orders.setStyleSheet("color:#fbbf24;font-weight:800;")
        risk_form.addRow("Max spread (points)", self.max_spread)
        risk_form.addRow("Max daily loss (%)", self.daily_loss)
        risk_form.addRow("Max drawdown (%)", self.max_drawdown)
        risk_form.addRow(self.override_tp_sl)
        risk_form.addRow("Take profit (%)", self.take_profit_percent)
        risk_form.addRow("Stop loss (%)", self.stop_loss_percent)
        risk_form.addRow("Decision delay after candle open (s)", self.bar_open_delay)
        self.candle_decision_note = QLabel(
            "The AI evaluates once at the start of each selected model candle. Broker TP/SL remain active continuously."
        )
        self.candle_decision_note.setWordWrap(True)
        risk_form.addRow(self.candle_decision_note)
        self.fixed_lot = self._spin(0.01, 100.0, float(self.bundle["backtest_config"].fixed_lot), 0.01, 2)
        risk_form.addRow("Fixed lot", self.fixed_lot)
        risk_form.addRow(self.demo_orders)
        self._update_tp_sl_controls(False)
        layout.addWidget(safety)

        control = QGroupBox("Bot control")
        buttons = QGridLayout(control)
        self.start_button = QPushButton("START BOT")
        self.stop_button = QPushButton("STOP BOT")
        self.refresh_button = QPushButton("REFRESH")
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_bot)
        self.stop_button.clicked.connect(self.stop_bot)
        self.refresh_button.clicked.connect(self.refresh_dashboard)
        buttons.addWidget(self.start_button, 0, 0, 1, 2)
        buttons.addWidget(self.stop_button, 1, 0)
        buttons.addWidget(self.refresh_button, 1, 1)
        layout.addWidget(control)
        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(panel)
        scroll.setMinimumSize(360, 0)
        return scroll

    def _spin(self, minimum: float, maximum: float, value: float, step: float, decimals: int) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(float(value))
        return spin

    def _positions_table(self) -> QWidget:
        self.positions = QTableWidget(0, 10)
        self.positions.setHorizontalHeaderLabels(["Ticket", "Magic", "Symbol", "Side", "Lot", "Open", "Current", "SL", "TP", "Profit"])
        self.positions.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        return self.positions

    def _deals_table(self) -> QWidget:
        self.deals = QTableWidget(0, 9)
        self.deals.setHorizontalHeaderLabels(["Time", "Deal", "Order", "Symbol", "Type", "Entry", "Lot", "Price", "Profit"])
        self.deals.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        return self.deals

    def _events_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.event_table = QTableWidget(0, 8)
        self.event_table.setHorizontalHeaderLabels(["Time", "Bar", "Signal", "Reason", "BUY p", "SELL p", "Action", "Threshold mode"])
        self.event_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(130)
        layout.addWidget(self.event_table)
        layout.addWidget(self.log)
        return panel

    def _connect_mt5(self) -> None:
        try:
            self.connector.connect()
            account = mt5.account_info()
            demo = account is not None and account.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO
            if demo:
                self.account_status.setText(f"● DEMO {account.login}")
                self.account_status.setStyleSheet("color:#22c55e;font-weight:800;")
            else:
                self.account_status.setText("● REAL ACCOUNT — EXECUTION BLOCKED")
                self.account_status.setStyleSheet("color:#ef4444;font-weight:900;")
                self.demo_orders.setChecked(False)
                self.demo_orders.setEnabled(False)
            self._write_log("Connected to MetaTrader 5")
            self.refresh_dashboard()
        except Exception as exc:
            self.account_status.setText("● MT5 CONNECTION ERROR")
            self.account_status.setStyleSheet("color:#ef4444;font-weight:800;")
            self._write_log(f"MT5 error: {exc}")

    def _bundle_candidate(self, timeframe: str) -> Path:
        if timeframe in self.bundle_paths:
            return self.bundle_paths[timeframe]
        if timeframe == "M15":
            generated = self.root_dir / "models" / "v51_shadow_m15.joblib"
            if generated.exists():
                return generated
            return self.root_dir / "models" / "v51_shadow_bundle.joblib"
        return self.root_dir / "models" / f"v51_shadow_{timeframe.lower()}.joblib"

    def _change_model_timeframe(self, timeframe: str) -> None:
        candidate = self._bundle_candidate(timeframe)
        if not candidate.exists():
            self.model_readiness.setText(f"{timeframe} MODEL NOT TRAINED")
            self.model_readiness.setStyleSheet("color:#ef4444;font-weight:900;")
            self.start_button.setEnabled(False)
            self._write_log(
                f"Missing {candidate}. Export a {timeframe} snapshot and freeze its own bundle first."
            )
            return
        try:
            bundle, manifest = load_verified_bundle(candidate)
            actual = str(bundle["snapshot_manifest"].get("timeframe", "")).upper()
            if actual != timeframe:
                raise ValueError(f"Bundle timeframe is {actual}, not {timeframe}")
            self.bundle_path = candidate
            self.bundle_paths[timeframe] = candidate
            self.bundle, self.manifest = bundle, manifest
            self.symbol = str(bundle["snapshot_manifest"]["symbol"])
            self.execution_timeframe = timeframe
            suffix = "" if timeframe == "M15" else f"_{timeframe.lower()}"
            self.events_path = self.root_dir / "demo_logs" / f"v51_demo{suffix}_events.jsonl"
            self.state_path = self.root_dir / "demo_state" / f"v51_demo{suffix}_state.json"
            self.buy_threshold.setValue(float(bundle["buy_threshold"]))
            self.sell_threshold.setValue(float(bundle["sell_threshold"]))
            self.meta_threshold.setValue(float(bundle["meta_gate_threshold"]))
            self.fixed_lot.setValue(float(bundle["backtest_config"].fixed_lot))
            self.take_profit_percent.setValue(float(bundle["backtest_config"].take_profit_pct) * 100.0)
            self.stop_loss_percent.setValue(float(bundle["backtest_config"].stop_loss_pct) * 100.0)
            self.symbol_model_label.setText(f"{self.symbol} • MODEL {timeframe}")
            self.model_readiness.setText(f"{timeframe} MODEL READY")
            self.model_readiness.setStyleSheet("color:#22c55e;font-weight:800;")
            self.start_button.setEnabled(True)
            self.chart_timeframe.setCurrentText(timeframe)
            self._threshold_status()
            self._write_log(f"Loaded {timeframe} bundle: {candidate}")
            self.refresh_dashboard()
        except Exception as exc:
            self.model_readiness.setText(f"{timeframe} BUNDLE ERROR")
            self.model_readiness.setStyleSheet("color:#ef4444;font-weight:900;")
            self.start_button.setEnabled(False)
            self._write_log(f"Could not load {timeframe} bundle: {exc}")

    def start_bot(self) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            return
        account = mt5.account_info()
        if account is None or account.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO:
            QMessageBox.critical(self, "Safety block", "Only an MT5 DEMO account can run this bot.")
            return
        experimental = self._experimental_thresholds()
        lot_changed = abs(self.fixed_lot.value() - float(self.bundle["backtest_config"].fixed_lot)) > 1e-9
        if experimental or self.override_tp_sl.isChecked() or lot_changed:
            changed = "Thresholds, TP/SL and/or lot"
            answer = QMessageBox.warning(
                self,
                "Experimental settings",
                f"{changed} differ from the frozen v5.1 candidate. These settings have not been validated. Continue on DEMO?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        if self.demo_orders.isChecked():
            answer = QMessageBox.question(
                self,
                "Enable DEMO orders",
                "The bot will send orders to the connected DEMO account. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        args = [
            "-u", str(self.root_dir / "run_v51_demo.py"),
            "--bundle", str(self.bundle_path),
            "--state", str(self.state_path),
            "--events", str(self.events_path),
            "--bar-open-delay-seconds", str(self.bar_open_delay.value()),
            "--max-spread-points", str(self.max_spread.value()),
            "--max-daily-loss-pct", str(self.daily_loss.value()),
            "--max-drawdown-pct", str(self.max_drawdown.value()),
            "--buy-threshold", str(self.buy_threshold.value()),
            "--sell-threshold", str(self.sell_threshold.value()),
            "--meta-threshold", str(self.meta_threshold.value()),
        ]
        if abs(self.fixed_lot.value() - float(self.bundle["backtest_config"].fixed_lot)) > 1e-9:
            args.extend(["--fixed-lot", str(self.fixed_lot.value())])
        if self.override_tp_sl.isChecked():
            args.extend([
                "--take-profit-percent", str(self.take_profit_percent.value()),
                "--stop-loss-percent", str(self.stop_loss_percent.value()),
            ])
        if self.demo_orders.isChecked():
            args.append("--enable-demo-orders")
        self.process.setWorkingDirectory(str(self.root_dir))
        self.process_mode = "bot"
        self.process.start(sys.executable, args)
        if not self.process.waitForStarted(5000):
            QMessageBox.critical(self, "Start error", self.process.errorString())
            return
        self.bot_status.setText("● BOT RUNNING")
        self.bot_status.setStyleSheet("color:#22c55e;font-weight:800;")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._set_controls_enabled(False)
        self._write_log("Demo bot started")

    def build_selected_model(self) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.warning(self, "Process busy", "Stop the current bot/process first.")
            return
        timeframe = self.bot_timeframe.currentText()
        answer = QMessageBox.question(
            self,
            f"Build {timeframe} model",
            f"Download {self.training_bars.value():,} closed {timeframe} bars and train a new DEVELOPMENT bundle? This may take several minutes.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        args = [
            "-u", str(self.root_dir / "prepare_v51_timeframe.py"),
            "--timeframe", timeframe,
            "--bars", str(self.training_bars.value()),
            "--chunk-size", "5000",
        ]
        self.process.setWorkingDirectory(str(self.root_dir))
        self.process_mode = "build"
        self.building_timeframe = timeframe
        self.process.start(sys.executable, args)
        if not self.process.waitForStarted(5000):
            QMessageBox.critical(self, "Build error", self.process.errorString())
            self.process_mode = None
            self.building_timeframe = None
            return
        self.bot_status.setText(f"● BUILDING {timeframe} MODEL")
        self.bot_status.setStyleSheet("color:#f59e0b;font-weight:800;")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._set_controls_enabled(False)
        self._write_log(f"Building {timeframe} model from {self.training_bars.value():,} bars")

    def stop_bot(self) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.terminate()
            if not self.process.waitForFinished(3000):
                self.process.kill()
                self.process.waitForFinished(1000)
            return
        self._process_finished()

    def _process_finished(self, exit_code: int = 0, *_args) -> None:
        completed_mode = self.process_mode
        completed_timeframe = self.building_timeframe
        self.process_mode = None
        self.building_timeframe = None
        self.bot_status.setText("● BOT STOPPED")
        self.bot_status.setStyleSheet("color:#94a3b8;font-weight:800;")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._set_controls_enabled(True)
        if completed_mode == "build":
            if exit_code != 0:
                self.bot_status.setText(f"● {completed_timeframe or ''} BUILD FAILED")
                self.bot_status.setStyleSheet("color:#ef4444;font-weight:900;")
                self._write_log(f"Model build failed with exit code {exit_code}")
                self._change_model_timeframe(self.bot_timeframe.currentText())
                return
            if completed_timeframe:
                self.bundle_paths.pop(completed_timeframe, None)
            self._change_model_timeframe(completed_timeframe or self.bot_timeframe.currentText())

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in (self.bot_timeframe, self.training_bars, self.build_model_button, self.buy_threshold, self.sell_threshold, self.meta_threshold, self.max_spread, self.daily_loss, self.max_drawdown, self.override_tp_sl, self.fixed_lot, self.bar_open_delay, self.demo_orders):
            widget.setEnabled(enabled)
        self.take_profit_percent.setEnabled(enabled and self.override_tp_sl.isChecked())
        self.stop_loss_percent.setEnabled(enabled and self.override_tp_sl.isChecked())

    def _update_tp_sl_controls(self, checked: bool) -> None:
        process_idle = self.process.state() == QProcess.ProcessState.NotRunning
        self.take_profit_percent.setEnabled(bool(checked) and process_idle)
        self.stop_loss_percent.setEnabled(bool(checked) and process_idle)

    def _experimental_thresholds(self) -> bool:
        return any((
            abs(self.buy_threshold.value() - float(self.bundle["buy_threshold"])) > 1e-9,
            abs(self.sell_threshold.value() - float(self.bundle["sell_threshold"])) > 1e-9,
            abs(self.meta_threshold.value() - float(self.bundle["meta_gate_threshold"])) > 1e-9,
        ))

    def _threshold_status(self) -> None:
        if self._experimental_thresholds():
            self.threshold_note.setText("EXPERIMENTAL — differs from frozen v5.1; DEMO only")
            self.threshold_note.setStyleSheet("color:#f59e0b;font-weight:800;")
        else:
            self.threshold_note.setText("Frozen v5.1 thresholds")
            self.threshold_note.setStyleSheet("color:#22c55e;font-weight:700;")

    def refresh_dashboard(self) -> None:
        if not self.connector.connected:
            return
        try:
            account = mt5.account_info()
            tick = mt5.symbol_info_tick(self.symbol)
            info = mt5.symbol_info(self.symbol)
            if account is not None:
                self.balance[1].setText(f"{account.balance:.2f} {account.currency}")
                self.equity[1].setText(f"{account.equity:.2f} {account.currency}")
                self.floating[1].setText(f"{account.profit:.2f} {account.currency}")
            if tick is not None and info is not None:
                self.bid[1].setText(str(tick.bid))
                self.ask[1].setText(str(tick.ask))
                self.spread[1].setText(f"{(tick.ask - tick.bid) / info.point:.1f}")
            bars = self.market.get_bars(self.symbol, self.chart_timeframe.currentText(), 160, False)
            self._update_market(bars)
            self._update_positions()
            self._update_deals()
            self._update_events()
        except Exception as exc:
            self._write_log(f"Refresh error: {exc}")

    def _update_market(self, bars) -> None:
        recent = bars.tail(100)
        self.price_curve.setData(np.arange(len(recent)), recent["close"].astype(float).to_numpy())
        self.chart.setTitle(f"{self.symbol} {self.chart_timeframe.currentText()} • last close {recent.iloc[-1]['close']}")
        rows = bars.tail(14).iloc[::-1]
        self.candles.setRowCount(len(rows))
        for r, (_, row) in enumerate(rows.iterrows()):
            values = [row["time"], row["open"], row["high"], row["low"], row["close"], row.get("tick_volume", ""), row.get("spread", "")]
            for c, value in enumerate(values):
                self.candles.setItem(r, c, QTableWidgetItem(str(value)))

    def _update_positions(self) -> None:
        positions = list(mt5.positions_get() or [])
        self.positions.setRowCount(len(positions))
        for r, pos in enumerate(positions):
            values = [pos.ticket, pos.magic, pos.symbol, "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL", pos.volume, pos.price_open, pos.price_current, pos.sl, pos.tp, pos.profit]
            for c, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if c == 9:
                    item.setForeground(QColor("#22c55e" if float(pos.profit) >= 0 else "#ef4444"))
                self.positions.setItem(r, c, item)

    def _update_deals(self) -> None:
        end = datetime.now(timezone.utc)
        deals = [deal for deal in (mt5.history_deals_get(end - timedelta(days=30), end) or []) if int(deal.magic) == MAGIC]
        deals = deals[-100:][::-1]
        self.deals.setRowCount(len(deals))
        for r, deal in enumerate(deals):
            side = "BUY" if deal.type == mt5.DEAL_TYPE_BUY else "SELL" if deal.type == mt5.DEAL_TYPE_SELL else str(deal.type)
            entry = "IN" if deal.entry == mt5.DEAL_ENTRY_IN else "OUT" if deal.entry == mt5.DEAL_ENTRY_OUT else str(deal.entry)
            values = [datetime.fromtimestamp(deal.time, timezone.utc).isoformat(), deal.ticket, deal.order, deal.symbol, side, entry, deal.volume, deal.price, deal.profit]
            for c, value in enumerate(values):
                self.deals.setItem(r, c, QTableWidgetItem(str(value)))

    def _update_events(self) -> None:
        if not self.events_path.exists():
            return
        lines = [line for line in self.events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        records = []
        for line in lines[-100:]:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        records.reverse()
        self.event_table.setRowCount(len(records))
        for r, event in enumerate(records):
            signal, execution, thresholds = event.get("signal", {}), event.get("execution", {}), event.get("thresholds", {})
            mode = "UNKNOWN" if not thresholds else "FROZEN" if thresholds.get("frozen_candidate_unchanged") else "EXPERIMENTAL"
            values = [event.get("event_time_utc"), signal.get("bar_time"), signal.get("side"), signal.get("reason"), self._prob(signal.get("buy_probability")), self._prob(signal.get("sell_probability")), execution.get("action"), mode]
            for c, value in enumerate(values):
                self.event_table.setItem(r, c, QTableWidgetItem(str(value)))
        if records:
            self._show_signal(records[0])

    def _show_signal(self, event: dict) -> None:
        signal = event.get("signal", {})
        side = str(signal.get("side", "HOLD"))
        self.signal_side.setText(side)
        color = "#22c55e" if side == "BUY" else "#ef4444" if side == "SELL" else "#94a3b8"
        self.signal_side.setStyleSheet(f"font-size:28px;font-weight:900;padding:10px;color:{color};")
        self.buy_probability.setText(f"BUY: {self._prob(signal.get('buy_probability'))}")
        self.sell_probability.setText(f"SELL: {self._prob(signal.get('sell_probability'))}")
        self.meta_probability.setText(f"SELL meta: {self._prob(signal.get('meta_probability'))}")
        self.drift_score.setText(f"Drift: {self._number(signal.get('drift_score'))}")
        self.signal_reason.setText(f"Reason: {signal.get('reason', '-')}")

    def _prob(self, value) -> str:
        return "-" if value is None else f"{float(value):.4f}"

    def _number(self, value) -> str:
        return "-" if value is None else f"{float(value):.4f}"

    def _read_process_output(self) -> None:
        text = bytes(self.process.readAllStandardOutput()).decode(errors="replace").strip()
        if text:
            self._write_log(text[-1500:])

    def _write_log(self, message: str) -> None:
        self.log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

    def closeEvent(self, event) -> None:  # noqa: N802
        self.ui_settings.setValue("main_splitter_state", self.main_splitter.saveState())
        self.stop_bot()
        self.connector.disconnect()
        event.accept()
