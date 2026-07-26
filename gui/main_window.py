from __future__ import annotations

from datetime import datetime

import MetaTrader5 as mt5
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from mt5.market_data import MarketData
from mt5.mt5_connector import MT5Connector


APP_STYLE = """
QMainWindow, QWidget { background: #0b1220; color: #e5e7eb; font-family: Segoe UI; }
QGroupBox { border: 1px solid #263247; border-radius: 10px; margin-top: 10px; padding: 10px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; color: #94a3b8; }
QPushButton { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 8px 14px; font-weight: 600; }
QPushButton:hover { background: #334155; }
QPushButton:disabled { color: #64748b; }
QTableWidget { background: #111827; alternate-background-color: #0f172a; border: 1px solid #263247; gridline-color: #263247; }
QHeaderView::section { background: #1e293b; color: #cbd5e1; padding: 7px; border: 0; }
QTextEdit, QDoubleSpinBox, QSpinBox { background: #111827; border: 1px solid #334155; border-radius: 6px; padding: 5px; }
QProgressBar { border: 1px solid #334155; border-radius: 6px; text-align: center; background: #111827; }
QProgressBar::chunk { background: #2563eb; border-radius: 5px; }
"""


class MainWindow(QMainWindow):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        trading = config.get("trading", {})
        risk = config.get("risk", {})

        self.symbol = str(trading.get("symbol", "XAUUSD.sc"))
        self.timeframe = str(trading.get("timeframe", "M15"))
        self.bars = int(trading.get("bars", 100))
        self.risk_cfg = risk

        self.connector = MT5Connector()
        self.market = MarketData()
        self.bot_running = False

        self.setWindowTitle("AI MT5 Trading Bot - HMI")
        self.resize(1440, 900)
        self.setStyleSheet(APP_STYLE)
        pg.setConfigOptions(antialias=True)

        self._build_ui()
        self._connect_mt5()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_data)
        self.timer.start(2000)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("AI MT5 TRADING BOT")
        title.setStyleSheet("font-size: 24px; font-weight: 800;")
        self.connection_label = QLabel("● MT5 DISCONNECTED")
        self.bot_status_label = QLabel("BOT STOPPED")
        self.symbol_label = QLabel(f"{self.symbol}  •  {self.timeframe}")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.symbol_label)
        header.addWidget(self.connection_label)
        header.addWidget(self.bot_status_label)
        root.addLayout(header)

        metrics = QGridLayout()
        self.balance_value = self._metric_card("Balance")
        self.equity_value = self._metric_card("Equity")
        self.profit_value = self._metric_card("Floating P/L")
        self.bid_value = self._metric_card("Bid")
        self.ask_value = self._metric_card("Ask")
        self.spread_value = self._metric_card("Spread")
        cards = [self.balance_value, self.equity_value, self.profit_value,
                 self.bid_value, self.ask_value, self.spread_value]
        for i, (card, _) in enumerate(cards):
            metrics.addWidget(card, 0, i)
        root.addLayout(metrics)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_market_panel())
        splitter.addWidget(self._build_control_panel())
        splitter.setSizes([950, 430])
        root.addWidget(splitter, stretch=1)

        bottom = QSplitter(Qt.Horizontal)
        bottom.addWidget(self._build_positions_panel())
        bottom.addWidget(self._build_log_panel())
        bottom.setSizes([900, 500])
        root.addWidget(bottom, stretch=0)

    def _metric_card(self, title: str) -> tuple[QFrame, QLabel]:
        frame = QFrame()
        frame.setStyleSheet("QFrame { background:#111827; border:1px solid #263247; border-radius:10px; }")
        layout = QVBoxLayout(frame)
        caption = QLabel(title)
        caption.setStyleSheet("color:#94a3b8; font-size:12px; border:0;")
        value = QLabel("-")
        value.setStyleSheet("font-size:20px; font-weight:700; border:0;")
        layout.addWidget(caption)
        layout.addWidget(value)
        return frame, value

    def _build_market_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        chart_group = QGroupBox("Realtime price chart")
        chart_layout = QVBoxLayout(chart_group)
        self.chart = pg.PlotWidget()
        self.chart.showGrid(x=True, y=True, alpha=0.18)
        self.chart.setLabel("left", "Price")
        self.chart.setLabel("bottom", "Recent closed candles")
        self.price_curve = self.chart.plot([], [], pen=pg.mkPen(width=2))
        chart_layout.addWidget(self.chart)
        layout.addWidget(chart_group, stretch=3)

        candle_group = QGroupBox("Recent closed candles")
        candle_layout = QVBoxLayout(candle_group)
        self.candle_table = QTableWidget(0, 7)
        self.candle_table.setAlternatingRowColors(True)
        self.candle_table.setHorizontalHeaderLabels(
            ["Time", "Open", "High", "Low", "Close", "Volume", "Spread"]
        )
        self.candle_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        candle_layout.addWidget(self.candle_table)
        layout.addWidget(candle_group, stretch=2)
        return panel

    def _build_control_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        ai_group = QGroupBox("AI signal preview")
        ai_layout = QVBoxLayout(ai_group)
        note = QLabel("Preview heuristic only — real AI model not connected yet")
        note.setWordWrap(True)
        note.setStyleSheet("color:#f59e0b;")
        ai_layout.addWidget(note)
        self.buy_bar = self._probability_row(ai_layout, "BUY")
        self.sell_bar = self._probability_row(ai_layout, "SELL")
        self.hold_bar = self._probability_row(ai_layout, "HOLD")
        self.ai_decision = QLabel("Decision: HOLD")
        self.ai_decision.setAlignment(Qt.AlignCenter)
        self.ai_decision.setStyleSheet("font-size:20px; font-weight:800; padding:10px;")
        ai_layout.addWidget(self.ai_decision)
        layout.addWidget(ai_group)

        risk_group = QGroupBox("Risk settings")
        form = QFormLayout(risk_group)
        self.risk_per_trade = self._double_spin(0.05, 5.0, self.risk_cfg.get("risk_per_trade_pct", 0.5), "%")
        self.max_positions = QSpinBox(); self.max_positions.setRange(1, 20); self.max_positions.setValue(int(self.risk_cfg.get("max_positions", 2)))
        self.daily_loss = self._double_spin(0.1, 20.0, self.risk_cfg.get("max_daily_loss_pct", 2.0), "%")
        self.max_drawdown = self._double_spin(1.0, 50.0, self.risk_cfg.get("max_drawdown_pct", 10.0), "%")
        self.sl_pct = self._double_spin(0.01, 10.0, self.risk_cfg.get("stop_loss_pct", 0.3), "%")
        self.tp_pct = self._double_spin(0.01, 20.0, self.risk_cfg.get("take_profit_pct", 0.6), "%")
        self.buy_threshold = self._double_spin(0.50, 0.99, self.risk_cfg.get("buy_threshold", 0.72), "")
        self.sell_threshold = self._double_spin(0.50, 0.99, self.risk_cfg.get("sell_threshold", 0.72), "")
        form.addRow("Risk / trade", self.risk_per_trade)
        form.addRow("Max positions", self.max_positions)
        form.addRow("Max daily loss", self.daily_loss)
        form.addRow("Max drawdown", self.max_drawdown)
        form.addRow("Stop Loss", self.sl_pct)
        form.addRow("Take Profit", self.tp_pct)
        form.addRow("BUY threshold", self.buy_threshold)
        form.addRow("SELL threshold", self.sell_threshold)
        layout.addWidget(risk_group)

        controls = QGroupBox("Bot control")
        control_layout = QGridLayout(controls)
        self.start_button = QPushButton("START MONITOR")
        self.stop_button = QPushButton("STOP")
        self.refresh_button = QPushButton("REFRESH NOW")
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_bot)
        self.stop_button.clicked.connect(self.stop_bot)
        self.refresh_button.clicked.connect(self.refresh_data)
        control_layout.addWidget(self.start_button, 0, 0, 1, 2)
        control_layout.addWidget(self.stop_button, 1, 0)
        control_layout.addWidget(self.refresh_button, 1, 1)
        layout.addWidget(controls)
        layout.addStretch()
        return panel

    def _probability_row(self, layout: QVBoxLayout, name: str) -> QProgressBar:
        label = QLabel(name)
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setFormat("%p%")
        layout.addWidget(label)
        layout.addWidget(bar)
        return bar

    def _double_spin(self, minimum: float, maximum: float, value: float, suffix: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(2)
        spin.setSingleStep(0.05)
        spin.setValue(float(value))
        spin.setSuffix(suffix)
        return spin

    def _build_positions_panel(self) -> QGroupBox:
        group = QGroupBox("Open positions")
        layout = QVBoxLayout(group)
        self.positions_table = QTableWidget(0, 9)
        self.positions_table.setAlternatingRowColors(True)
        self.positions_table.setHorizontalHeaderLabels(
            ["Ticket", "Symbol", "Type", "Volume", "Open", "Current", "SL", "TP", "Profit"]
        )
        self.positions_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.positions_table)
        return group

    def _build_log_panel(self) -> QGroupBox:
        group = QGroupBox("System log")
        layout = QVBoxLayout(group)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(190)
        layout.addWidget(self.log_box)
        return group

    def _connect_mt5(self) -> None:
        try:
            self.connector.connect()
            self.connection_label.setText("● MT5 CONNECTED")
            self.connection_label.setStyleSheet("color:#22c55e; font-weight:700;")
            self._log("Connected to MetaTrader 5")
            self.refresh_data()
        except Exception as exc:
            self.connection_label.setText("● MT5 ERROR")
            self.connection_label.setStyleSheet("color:#ef4444; font-weight:700;")
            self._log(f"MT5 connection error: {exc}")

    def refresh_data(self) -> None:
        if not self.connector.connected:
            return
        try:
            account = self.connector.account_snapshot()
            self.balance_value[1].setText(f"{account.balance:.2f} {account.currency}")
            self.equity_value[1].setText(f"{account.equity:.2f} {account.currency}")
            self.profit_value[1].setText(f"{account.profit:.2f} {account.currency}")

            tick = self.market.get_tick(self.symbol)
            self.bid_value[1].setText(str(tick["bid"]))
            self.ask_value[1].setText(str(tick["ask"]))
            self.spread_value[1].setText(str(round(tick["spread_price"], 5)))

            df = self.market.get_bars(
                symbol=self.symbol,
                timeframe=self.timeframe,
                count=max(60, min(self.bars, 300)),
                include_current_bar=False,
            )
            self._update_chart(df)
            self._update_candle_table(df.tail(12))
            self._update_positions()
            self._update_ai_preview(df)
        except Exception as exc:
            self._log(f"Refresh error: {exc}")

    def _update_chart(self, df) -> None:
        recent = df.tail(60)
        x = np.arange(len(recent))
        y = recent["close"].astype(float).to_numpy()
        self.price_curve.setData(x, y)
        if len(y):
            self.chart.setTitle(f"{self.symbol} {self.timeframe}  |  Last close: {y[-1]:.5f}")

    def _update_candle_table(self, df) -> None:
        self.candle_table.setRowCount(len(df))
        for r, (_, row) in enumerate(df.iterrows()):
            values = [
                str(row["time"]), row["open"], row["high"], row["low"],
                row["close"], row["tick_volume"], row["spread"],
            ]
            for c, value in enumerate(values):
                self.candle_table.setItem(r, c, QTableWidgetItem(str(value)))

    def _update_positions(self) -> None:
        positions = mt5.positions_get()
        if positions is None:
            positions = []
        self.positions_table.setRowCount(len(positions))
        for r, pos in enumerate(positions):
            side = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
            values = [pos.ticket, pos.symbol, side, pos.volume, pos.price_open,
                      pos.price_current, pos.sl, pos.tp, pos.profit]
            for c, value in enumerate(values):
                self.positions_table.setItem(r, c, QTableWidgetItem(str(value)))

    def _update_ai_preview(self, df) -> None:
        closes = df["close"].astype(float)
        if len(closes) < 20:
            return
        fast = closes.tail(5).mean()
        slow = closes.tail(20).mean()
        volatility = max(closes.tail(20).std(), 1e-9)
        score = np.clip((fast - slow) / volatility, -2.0, 2.0)
        buy = 1.0 / (1.0 + np.exp(-2.0 * score))
        sell = 1.0 - buy
        confidence = abs(buy - sell)
        hold = max(0.0, 0.55 - confidence)
        total = buy + sell + hold
        buy, sell, hold = buy / total, sell / total, hold / total

        self.buy_bar.setValue(round(buy * 100))
        self.sell_bar.setValue(round(sell * 100))
        self.hold_bar.setValue(round(hold * 100))

        decision = "HOLD"
        if buy >= self.buy_threshold.value():
            decision = "BUY"
        elif sell >= self.sell_threshold.value():
            decision = "SELL"
        self.ai_decision.setText(f"Decision: {decision}")

    def start_bot(self) -> None:
        self.bot_running = True
        self.bot_status_label.setText("MONITOR RUNNING")
        self.bot_status_label.setStyleSheet("color:#22c55e; font-weight:700;")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._log("Monitoring started. Order execution remains disabled.")

    def stop_bot(self) -> None:
        self.bot_running = False
        self.bot_status_label.setText("BOT STOPPED")
        self.bot_status_label.setStyleSheet("")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._log("Monitoring stopped")

    def _log(self, message: str) -> None:
        self.log_box.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

    def closeEvent(self, event) -> None:  # noqa: N802
        self.connector.disconnect()
        event.accept()
