#!/bin/bash
cd "$(dirname "$0")"

echo "Stopping any existing server on port 8000..."
lsof -ti:8000 | xargs kill -9 2>/dev/null
sleep 1

pip install -r requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
