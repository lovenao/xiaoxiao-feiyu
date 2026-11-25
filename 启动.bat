@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title XiaoXiaoFatFish

set "PYTHON_EXE="
where py >nul 2>nul
if not errorlevel 1 set "PYTHON_EXE=py -3"

if not defined PYTHON_EXE (
    where python >nul 2>nul
    if not errorlevel 1 set "PYTHON_EXE=python"
)

if not defined PYTHON_EXE (
    echo.
    echo Python 3 was not found. Please install Python 3 first.
    pause
    exit /b 1
)

echo Starting XiaoXiaoFatFish...
echo.
%PYTHON_EXE% app.py %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Program exited with code %EXIT_CODE%.
    pause
)

endlocal
