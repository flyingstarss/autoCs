@echo off
setlocal
cd /d "%~dp0"
echo ==== cs-agent console restart ====
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0restart_console.ps1" %*
echo.
pause
