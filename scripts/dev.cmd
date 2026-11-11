@echo off
rem Usage: scripts\dev.cmd <command...>
rem Opens the MSVC x64 environment (cl, cmake, ninja) and sets VCPKG_ROOT when needed.
setlocal EnableExtensions
if not defined VSCMD_VER (
  call :setup_msvc
  if errorlevel 1 exit /b 1
)
if not defined VCPKG_ROOT call :setup_vcpkg
%*
exit /b %ERRORLEVEL%

:setup_msvc
set "WAFFLE_VSINST="
set "WAFFLE_VCVARS="
set "WAFFLE_VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%WAFFLE_VSWHERE%" goto :vswhere_done
"%WAFFLE_VSWHERE%" -latest -products "*" -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -version "[17.0,)" -property installationPath > "%TEMP%\waffle-vswhere.txt" 2>nul
if exist "%TEMP%\waffle-vswhere.txt" set /p WAFFLE_VSINST=<"%TEMP%\waffle-vswhere.txt"
:vswhere_done
if defined WAFFLE_VSINST if exist "%WAFFLE_VSINST%\VC\Auxiliary\Build\vcvars64.bat" set "WAFFLE_VCVARS=%WAFFLE_VSINST%\VC\Auxiliary\Build\vcvars64.bat"
if not defined WAFFLE_VCVARS if exist "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" (
  set "WAFFLE_VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
  set "WAFFLE_VSINST=C:\Program Files\Microsoft Visual Studio\18\Community"
)
if not defined WAFFLE_VCVARS (
  echo Visual Studio C++ was not found.
  echo Install Visual Studio 2022 or newer, with the Desktop development with C++ workload.
  echo https://visualstudio.microsoft.com/downloads/
  exit /b 1
)
call "%WAFFLE_VCVARS%" >nul
if errorlevel 1 exit /b 1
if not defined VCPKG_ROOT if exist "%WAFFLE_VSINST%\VC\vcpkg\vcpkg.exe" set "VCPKG_ROOT=%WAFFLE_VSINST%\VC\vcpkg"
exit /b 0

:setup_vcpkg
if defined VSINSTALLDIR if exist "%VSINSTALLDIR%VC\vcpkg\vcpkg.exe" set "VCPKG_ROOT=%VSINSTALLDIR%VC\vcpkg"
if not defined VCPKG_ROOT if defined VSINSTALLDIR if exist "%VSINSTALLDIR%\VC\vcpkg\vcpkg.exe" set "VCPKG_ROOT=%VSINSTALLDIR%\VC\vcpkg"
if not defined VCPKG_ROOT echo Warning: VCPKG_ROOT was not found. The CMake presets need it.>&2
exit /b 0
