@echo off
echo CART-Clin: Clinical LLM Red-Teaming Framework
echo University of Surrey - MSc Dissertation
echo ===========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Create directories if they don't exist
if not exist "results" mkdir results
if not exist "logs" mkdir logs

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Starting CART-Clin framework...
echo.
python cart_clin.py

pause