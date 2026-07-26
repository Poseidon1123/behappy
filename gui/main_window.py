from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from mt5.market_data import MarketData
from mt5.mt5_connector import MT5Connector


class MainWindow(QMainWindow):
    def __init__(self, config: dict) -> None:
        super().__init__()

        trading = config.get("trading", {})
        self.symbol = str(trading.get("symbol", "XAUUSD.sc"))
        self.timeframe = str(trading.get("timeframe", "M15"))
        self.bars = int(trading.get("bars", 100))

        self.connector = MT5Connector()
        self.market = MarketData()
        self.bot_running = False

        self.setWindowTitle("AI MT5 Trading Bot")
        self.resize(1180, 760)

        self._build_ui()
        self._connect_mt5()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_data)
        self.timer.start(2000)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        title = QLabel("AI MT5 TRADING BOT")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: 700; padding: 8px;")
        root.addWidget(title)

        status_row = QHBoxLayout()
        self.connection_label = QLabel("MT5: DISCONNECTED")
        self.bot_status_label = QLabel("BOT: STOPPED")
        self.symbol_label = QLabel(f"SYMBOL: {self.symbol} | {self.timeframe}")
        status_row.addWidget(self.connection_label)
        status_row.addWidget(self.bot_status_label)
        status_row.addWidget(self.symbol_label)
        status_row.addStretch()
        root.addLayout(status_row)

        top_grid = QGridLayout()
        self.balance_value = self._metric_box("Balance", "-")
        self.equity_value = self._metric_box("Equity", "-")
        self.profit_value = self._metric_box("Profit", "-")
        self.bid_value = self._metric_box("Bid", "-")
        self.ask_value = self._metric_box("Ask", "-")
        self.spread_value = self._metric_box("Spread", "-")

        for i, widget in enumerate([
            self.balance_value[0], self.equity_value[0], self.profit_value[0],
            self.bid_value[0], self.ask_value[0], self.spread_value[0],
        ]):
            top_grid.addWidget(widget, i // 3, i % 3)
        root.addLayout(top_grid)

        controls = QHBoxLayout()
        self.start_button = QPushButton("START BOT")
        self.stop_button = QPushButton("STOP BOT")
        self.refresh_button = QPushButton("REFRESH NOW")
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_bot)
        self.stop_button.clicked.connect(self.stop_bot)
        self.refresh_button.clicked.connect(self.refresh_data)
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        controls.addWidget(self.refresh_button)
        controls.addStretch()
        root.addLayout(controls)

        self.candle_table = QTableWidget(0, 7)
        self.candle_table.setHorizontalHeaderLabels([
            "Time", "Open", "High", "Low", "Close", "Volume", "Spread"
        ])
        root.addWidget(self.candle_table, stretch=2)

        log_group = QGroupBox("System Log")
        log_layout = QVBoxLayout(log_group)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        log_layout.addWidget(self.log_box)
        root.addWidget(log_group, stretch=1)

    def _metric_box(self, title: str, value: str) -> tuple[QGroupBox, QLabel]:
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        label = QLabel(value)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 20px; font-weight: 600;")
        layout.addWidget(label)
        return box, label

    def _connect_mt5(self) -> None:
        try:
            self.connector.connect()
            self.connection_label.setText("MT5: CONNECTED")
            self._log("Connected to MetaTrader 5")
            self.refresh_data()
        except Exception as exc:
            self.connection_label.setText("MT5: ERROR")
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
                count=min(self.bars, 20),
                include_current_bar=False,
            ).tail(10)

            self.candle_table.setRowCount(len(df))
            for row_index, (_, row) in enumerate(df.iterrows()):
                values = [
                    str(row["time"]), row["open"], row["high"], row["low"],
                    row["close"], row["tick_volume"], row["spread"]
                ]
                for col_index, value in enumerate(values):
                    self.candle_table.setItem(
                        row_index,
                        col_index,
                        QTableWidgetItem(str(value)),
                    )
        except Exception as exc:
            self._log(f"Refresh error: {exc}")

    def start_bot(self) -> None:
        self.bot_running = True
        self.bot_status_label.setText("BOT: RUNNING (MONITOR ONLY)")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._log("Bot monitoring started. Order execution is still disabled.")

    def stop_bot(self) -> None:
        self.bot_running = False
        self.bot_status_label.setText("BOT: STOPPED")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._log("Bot stopped")

    def _log(self, message: str) -> None:
        now = datetime.now().strftime("%H:%M:%S")
        self.log_box.append(f"[{now}] {message}")

    def closeEvent(self, event) -> None:  # noqa: N802
        self.connector.disconnect()
        event.accept()
