@echo off
REM ============================================
REM  Cloudflare R2 Setup Script for GSplat
REM  Run this to configure cloud storage
REM ============================================

echo.
echo ================================================
echo   CLOUDFLARE R2 CLOUD STORAGE SETUP
echo ================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    pause
    exit /b 1
)

REM Install boto3 if not present
echo Checking boto3...
python -c "import boto3" >nul 2>&1
if errorlevel 1 (
    echo Installing boto3...
    pip install boto3
)

REM Run interactive setup
cd /d "%~dp0"
python pipeline\r2_uploader.py setup

echo.
pause
