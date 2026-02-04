@echo off
title Menu Detection - Live Camera
color 0A
cls

echo ================================================
echo         MENU DETECTION - LIVE CAMERA
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed!
    echo.
    echo Please run setup.bat first or install Python from:
    echo https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment if it exists
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
    echo [OK] Virtual environment activated
) else (
    echo [WARNING] No virtual environment found
    echo Run setup.bat first for best results
    echo.
)

echo.
echo Starting application...
echo.
echo ================================================
echo.

REM Run the application
python live.py

REM Check if there was an error
if errorlevel 1 (
    echo.
    echo ================================================
    echo [ERROR] Application stopped with an error
    echo ================================================
    echo.
)

pause