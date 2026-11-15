@echo off
echo.
echo ================================================================================
echo BDBV HOST PREDICTION - WEB APPLICATION LAUNCHER
echo ================================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo [1] Installing web dependencies...
pip install -r requirements-web.txt

echo.
echo [2] Starting Flask web server...
echo.
echo ================================================================================
echo WEB APP IS STARTING...
echo ================================================================================
echo.
echo Open your browser and navigate to:
echo   http://localhost:5000
echo.
echo The terminal will show server logs below.
echo Press Ctrl+C to stop the server.
echo.
echo ================================================================================
echo.

python app.py

pause
