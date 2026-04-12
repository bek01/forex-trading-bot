#!/bin/bash
# Trading Bot Setup Script
# Run once to set up the environment

set -e

echo "=== Trading Bot Setup ==="

# 1. Install Python dependencies
echo "Installing Python packages..."
/home/ubuntu/venv/bin/pip install -e ".[dev]" 2>/dev/null || \
    /home/ubuntu/venv/bin/pip install \
        oandapyV20 pandas numpy pandas-ta httpx aiohttp \
        python-dotenv pydantic pydantic-settings \
        python-telegram-bot apscheduler structlog \
        aiosqlite pytest pytest-asyncio ruff

# 2. Create directories
echo "Creating directories..."
mkdir -p db logs backtest_results

# 3. Copy env template
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from template — EDIT THIS FILE with your API keys"
else
    echo ".env already exists, skipping"
fi

# 4. Run tests
echo "Running tests..."
/home/ubuntu/venv/bin/python3 -m pytest tests/ -v --tb=short 2>&1 | tail -20

# 5. Install systemd service (optional)
read -p "Install systemd service? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo cp trading-bot.service /etc/systemd/system/
    sudo systemctl daemon-reload
    echo "Service installed. Start with: sudo systemctl start trading-bot"
    echo "Enable on boot: sudo systemctl enable trading-bot"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit .env with your OANDA API credentials"
echo "  2. Create a free OANDA practice account: https://www.oanda.com/demo-account/"
echo "  3. Run backtest: /home/ubuntu/venv/bin/python3 -m backtesting.run_backtest"
echo "  4. Start bot: /home/ubuntu/venv/bin/python3 main.py --paper"
echo "  5. Monitor: /home/ubuntu/venv/bin/python3 monitor.py"
