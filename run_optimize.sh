#!/bin/bash
export VIRTUAL_ENV=/home/ubuntu/venv
export PATH="/home/ubuntu/venv/bin:$PATH"
cd /home/ubuntu/trading_bot
PYTHONUNBUFFERED=1 python3 -u param_optimize.py 2>&1
