@echo off
setlocal
cd /d "%~dp0"

echo ========================================
echo  WTP Degradation Preview - Install
echo ========================================
echo.

:: -- Check Python 3.14 (64-bit) --
py -3.14 -c "import sys; sys.exit(0 if sys.maxsize > 2**32 else 1)" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 64-bit Python 3.14 was not found.
    echo         Install it from https://www.python.org/downloads/
    echo         and keep "Python install manager" or "py launcher" ticked.
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%v in ('py -3.14 --version 2^>^&1') do set PYVER=%%v
echo Found Python %PYVER%

:: -- Create venv --
set "VENV_PY=%~dp0venv\Scripts\python.exe"
if exist "%VENV_PY%" (
    "%VENV_PY%" -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 14) else 1)" >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] The existing venv folder was made with another Python version.
        echo         Delete the venv folder, then run install.bat again.
        pause
        exit /b 1
    )
    echo Virtual environment already exists, updating packages...
) else (
    echo Creating virtual environment...
    py -3.14 -m venv "%~dp0venv"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

:: -- Install dependencies --
echo.
echo Installing dependencies (PyTorch with CUDA is about 3 GB)...
"%VENV_PY%" -m pip install --upgrade pip >nul 2>&1
"%VENV_PY%" -m pip install --only-binary=:all: -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

:: -- Vendored chainner_ext (C build for Python 3.14): make vendor\ importable --
"%VENV_PY%" -c "import pathlib, sys, sysconfig; pathlib.Path(sysconfig.get_paths()['purelib'], 'wtp_vendor.pth').write_text(str(pathlib.Path(sys.argv[1]).resolve()) + '\n', encoding='utf-8')" "%~dp0vendor"
if errorlevel 1 (
    echo [ERROR] Failed to register the vendor folder.
    pause
    exit /b 1
)

:: -- Check imports --
"%VENV_PY%" -W ignore -c "import PySide6, cv2, numpy, torch, av, colour, pepeline, pepedpid, chainner_ext"
if errorlevel 1 (
    echo [ERROR] Packages installed, but an import failed - see the error above.
    pause
    exit /b 1
)

echo.
echo ========================================
echo  Install complete! Run 'run.bat' to start.
echo ========================================
pause
