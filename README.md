# AI MT5 Trading Bot

Python foundation for an AI-assisted MetaTrader 5 trading system.

## Implemented

### Step 1 - MetaTrader 5 connection

- Connect to a running MetaTrader 5 terminal.
- Read account information.
- Select a trading symbol.
- Read latest tick data.
- Download closed OHLC bars.
- Keep credentials outside source control.

### Step 2 - PySide6 HMI

- Desktop dashboard built with PySide6.
- Live MT5 connection status.
- Balance, equity and floating profit display.
- Bid, ask and spread display.
- Latest closed-candle table.
- START BOT / STOP BOT monitoring controls.
- Auto refresh every 2 seconds.
- System log panel.

> The current START BOT button only enables monitoring mode. Order execution is intentionally disabled until the risk manager and trade manager are implemented.

## Requirements

- Windows with MetaTrader 5 desktop installed.
- Python 3.10+ recommended.
- MT5 terminal should be open and logged in before running.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `config/config.yaml` so `symbol` exactly matches the broker symbol in MT5 Market Watch.

## Run the HMI

```bash
python main.py
```

The HMI automatically connects to the open MetaTrader 5 terminal and refreshes account and market information.

## Safety

The current project is still read-only. It does not send, modify or close orders.
