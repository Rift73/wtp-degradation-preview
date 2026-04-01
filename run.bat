@echo off
cd /d "%~dp0"

if not exist "%~dp0venv\Scripts\pythonw.exe" (
    echo Virtual environment not found. Run install.bat first.
    pause
    exit /b 1
)

if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
) else (
    echo WARNING: Could not find vcvars64.bat — CUDA JIT compilation may fail
)
start "" "%~dp0venv\Scripts\pythonw.exe" "%~dp0main.pyw"
