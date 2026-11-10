@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo  Codex CLI  -  cs-agent-system
echo  当前目录: %CD%
echo ============================================
codex %*
if errorlevel 1 pause
