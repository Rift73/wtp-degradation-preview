@echo off
setlocal
cd /d "%~dp0"

echo ========================================
echo  WTP Degradation Preview - Install
echo ========================================
echo.

:: ── Check Python ──
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    echo         Install Python 3.10+ from https://www.python.org
    echo         Make sure "Add to PATH" is checked during install.
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo Found Python %PYVER%

:: ── Create venv ──
if exist "%~dp0venv\Scripts\python.exe" (
    echo Virtual environment already exists, updating packages...
) else (
    echo Creating virtual environment...
    python -m venv "%~dp0venv"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

:: ── Install dependencies ──
echo.
echo Installing dependencies...
"%~dp0venv\Scripts\pip.exe" install --upgrade pip >nul 2>&1
"%~dp0venv\Scripts\pip.exe" install -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo ========================================
echo  Install complete! Run 'run.bat' to start.
echo ========================================
pause
