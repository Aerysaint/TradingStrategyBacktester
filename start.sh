#!/bin/bash

echo "[1/3] Installing Python dependencies..."
pip install -r requirement.txt
if [ $? -ne 0 ]; then
    echo "Pip install failed. Exiting..."
    exit 1
fi

echo "[2/3] Starting Uvicorn backend in the background..."
# The '&' runs Uvicorn in the background so the terminal can move on to the frontend
uvicorn backend.main:app &
BACKEND_PID=$!

echo "[3/3] Starting frontend development server..."
cd frontend
npm run dev

# Optional: Closes the backend process when you Ctrl+C out of the frontend npm script
trap "kill $BACKEND_PID" EXIT