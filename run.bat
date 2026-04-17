@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

:: Find a compatible Python (3.10-3.12). Python 3.13+ lacks prebuilt wheels for pydantic-core.
set PYTHON=
for %%v in (3.12 3.11 3.10) do (
    if not defined PYTHON (
        py -%%v --version >nul 2>&1 && set PYTHON=py -%%v
    )
)
if not defined PYTHON (
    echo ERROR: Python 3.10, 3.11, or 3.12 is required but not found.
    echo Please install Python 3.12 from https://www.python.org/downloads/release/python-3129/
    pause
    exit /b 1
)
echo Using: %PYTHON%

echo Stopping any existing server on port 8001...
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8001 " ^| findstr "LISTENING"') do (
    taskkill /PID %%a /F >nul 2>&1
)
timeout /t 2 /nobreak >nul

echo Installing dependencies...
%PYTHON% -m pip install -r requirements.txt

echo Starting server on http://localhost:8001
%PYTHON% -m uvicorn backend.main:app --host 0.0.0.0 --port 8001 --reload
pause
