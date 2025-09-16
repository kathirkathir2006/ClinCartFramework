@echo off
echo CART-Clin Setup Script
echo ======================
echo.

REM Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    echo Make sure to check "Add to PATH" during installation
    pause
    exit /b 1
)

echo Python found: 
python --version

REM Create project directories
echo.
echo Creating project directories...
if not exist "results" mkdir results
if not exist "logs" mkdir logs
echo - results/ directory created
echo - logs/ directory created

REM Install Python dependencies
echo.
echo Installing Python dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    echo Please check your internet connection
    pause
    exit /b 1
)

echo.
echo Setup complete! You can now run:
echo   run.bat           - Start the framework
echo   python cart_clin.py - Run directly with Python
echo.
pause