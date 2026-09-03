@echo off
REM Double-click-friendly launcher for Windows.
cd /d "%~dp0"
where python >nul 2>nul
if %errorlevel% == 0 (
    python start.py
) else (
    echo Python was not found on this computer.
    echo Please install it from https://python.org and try again.
    echo IMPORTANT: during install, check the box that says "Add Python to PATH".
    pause
)
