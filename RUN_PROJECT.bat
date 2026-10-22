@echo off
title OmniScan - AI Court-Admissible Forensics API
color 0B

echo  ================================================================
echo    OMNISCAN FORENSICS  ^|  AI COURT-ADMISSIBLE FORENSICS ENGINE
echo    Dr. D. Y. Patil College of Engineering and Innovation
echo  ================================================================
echo.

REM ── Locate the project root
set "ROOT=%~dp0"
set "API_DIR=%ROOT%api"
set "INDEX=%ROOT%index.html"

echo  [1/3] Checking dependencies...
python --version >nul 2>&1
if %errorlevel% NEQ 0 (
    echo  [ERROR] Python is not installed or not on PATH!
    pause
    exit /b 1
)

uvicorn --version >nul 2>&1
if %errorlevel% NEQ 0 (
    echo  [INFO] Installing required packages...
    pip install fastapi "uvicorn[standard]" pydantic-settings >nul 2>&1
) else (
    echo  [OK]  Dependencies satisfied.
)

echo.
echo  [2/3] Cleaning up active background ports...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000" 2^>nul') do (
    taskkill /PID %%a /F >nul 2>&1
)
timeout /t 1 >nul 2>&1

echo.
echo  [3/3] Opening Browser...
start "" "%INDEX%"

echo.
echo  ================================================================
echo    Project is LIVE!
echo    API  ^|  http://127.0.0.1:8000
echo    Docs ^|  http://127.0.0.1:8000/docs
echo.
echo    DO NOT close this window. It is the API engine.
echo    Press Ctrl+C to safely shut down.
echo  ================================================================
echo.

cd /d "%API_DIR%"
uvicorn app.main:app --host 0.0.0.0 --port 8000
pause
