@echo off
echo ========================================================
echo ⚡ HyperSplat: All-in-One Environment Setup
echo ========================================================
echo.

:: 1. Setup Windows Environment
echo [1/2] Creating Windows Environment (hypersplat)...
if exist "environment_windows.yml" (
    call conda env create -f environment_windows.yml
) else (
    echo Error: environment_windows.yml not found!
    exit /b 1
)

:: 2. Setup WSL Environment
echo.
echo [2/2] Creating WSL Environment (ace0)...
if exist "environment_wsl.yml" (
    echo Converting path for WSL...
    wsl bash -c "if [ -f environment_wsl.yml ]; then conda env create -f environment_wsl.yml; else echo 'WSL Error: environment_wsl.yml not found inside WSL context!'; fi"
) else (
    echo Error: environment_wsl.yml not found!
    exit /b 1
)

echo.
echo ========================================================
echo ✅ Setup Complete!
echo To activate Windows env: conda activate 3dgrut
echo To activate WSL env:     wsl conda activate ace0
echo ========================================================
pause
