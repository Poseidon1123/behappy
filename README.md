# AI MT5 Trading Bot

Python foundation for an AI-assisted MetaTrader 5 trading system.

## Step 1 implemented

- Connect to a running MetaTrader 5 terminal.
- Read terminal/account information.
- Select a trading symbol.
- Read latest tick data.
- Download closed OHLC bars.
- Keep credentials outside source control.
- Provide a clean base for later AI, risk-management and HMI modules.

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
python main.py
```

Edit `config/config.yaml` so `symbol` exactly matches the broker symbol in MT5 Market Watch.

## Safety

Step 1 is read-only. It does not send, modify or close any order.
