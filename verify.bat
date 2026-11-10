@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONPATH=%~dp0src
if not defined PYTHON set "PYTHON=python"

> _verify.log echo ==== verify start ====
>> _verify.log echo [1/4] compileall
"%PYTHON%" -m compileall -q src _smoke.py _check_llm.py >> _verify.log 2>&1
>> _verify.log echo exit=%ERRORLEVEL%

>> _verify.log echo.
>> _verify.log echo [2/4] check_llm
"%PYTHON%" _check_llm.py >> _verify.log 2>&1
>> _verify.log echo exit=%ERRORLEVEL%

>> _verify.log echo.
>> _verify.log echo [3/4] smoke
"%PYTHON%" _smoke.py >> _verify.log 2>&1
>> _verify.log echo exit=%ERRORLEVEL%

>> _verify.log echo.
>> _verify.log echo [4/4] demo
"%PYTHON%" -m cs_agent_system.demo >> _verify.log 2>&1
>> _verify.log echo exit=%ERRORLEVEL%

>> _verify.log echo ==== verify end ====
echo.
echo Done. Log: %~dp0_verify.log
type _verify.log
echo.
pause
