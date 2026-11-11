@echo off
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul
where powershell >nul 2>&1
if errorlevel 1 (
  echo PowerShell was not found. Windows PowerShell 5.1 is part of Windows.
  goto :fail
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install.ps1" -Launch
set "ERR=%ERRORLEVEL%"
if "%ERR%"=="0" exit /b 0
:fail
echo %cmdcmdline% | find /I "%~nx0" >nul
if not errorlevel 1 pause
if not defined ERR set "ERR=1"
exit /b %ERR%
