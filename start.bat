@echo off
echo [1/3] Installing Python dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Extraction failed or pip not found. Exiting...
    pause
    exit /b %errorlevel%
)

echo [2/3] Starting Uvicorn backend...
:: Using 'start' keeps Uvicorn running in a separate window so the script can continue to the frontend
start cmd /k "python3 -m uvicorn backend.main:app"

echo [3/3] Starting frontend development server...
cd frontend
npm run dev

pause