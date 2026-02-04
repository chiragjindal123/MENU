@echo off
title Menu Detection - First Time Setup
color 0B
cls

echo ================================================
echo    MENU DETECTION - FIRST TIME SETUP
echo ================================================
echo.
echo This will install all required dependencies.
echo Please wait, this may take a few minutes...
echo.
echo ================================================
echo.

REM Check if Python is installed
echo [1/5] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo.
    echo Please install Python 3.10 or higher from:
    echo https://www.python.org/downloads/
    echo.
    echo IMPORTANT: Check "Add Python to PATH" during installation!
    echo.
    pause
    exit /b 1
)

REM Get Python version and check if it's 3.10+
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found Python %PYTHON_VERSION%

REM Extract major and minor version (e.g., 3.10 from 3.10.11)
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set PYTHON_MAJOR=%%a
    set PYTHON_MINOR=%%b
)

REM Check if version is 3.10 or higher
if %PYTHON_MAJOR% LSS 3 (
    echo [ERROR] Python version is too old!
    echo Required: Python 3.10 or higher
    echo Current: Python %PYTHON_VERSION%
    echo.
    echo Please install Python 3.10+ from:
    echo https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

if %PYTHON_MAJOR% EQU 3 if %PYTHON_MINOR% LSS 10 (
    echo [ERROR] Python version is too old!
    echo Required: Python 3.10 or higher
    echo Current: Python %PYTHON_VERSION%
    echo.
    echo Please install Python 3.10+ from:
    echo https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo [OK] Python %PYTHON_VERSION% is compatible
echo.

REM Create virtual environment
echo [2/5] Creating virtual environment...
if exist .venv (
    echo [SKIP] Virtual environment already exists
) else (
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)
echo.

REM Activate virtual environment
echo [3/5] Activating virtual environment...
call .venv\Scripts\activate.bat
echo [OK] Activated
echo.

REM Upgrade pip
echo [4/5] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo [OK] Pip upgraded
echo.

REM Install required packages
echo [5/5] Installing required packages...
echo      - ultralytics (YOLO model)
echo      - opencv-python (Camera support)
echo.
echo This may take 2-5 minutes...
echo.

pip install ultralytics opencv-python --quiet

if errorlevel 1 (
    echo.
    echo [ERROR] Installation failed!
    echo.
    echo Try running this command manually:
    echo pip install ultralytics opencv-python
    echo.
    pause
    exit /b 1
)

echo.
echo ================================================
echo          SETUP COMPLETE!
echo ================================================
echo.
echo All dependencies installed successfully.
echo.
echo To run the application:
echo   - Double-click run.bat
echo.
echo Or run directly:
echo   - python live.py
echo.
echo ================================================
echo.
pause