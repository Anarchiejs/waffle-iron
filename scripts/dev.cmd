@echo off
rem Usage : scripts\dev.cmd <commande...>
rem Ouvre l'environnement MSVC x64 (cl, cmake, ninja) et renseigne VCPKG_ROOT si besoin.
setlocal EnableExtensions
if not defined VSCMD_VER (
  call :setup_msvc
  if errorlevel 1 exit /b 1
)
if not defined VCPKG_ROOT call :setup_vcpkg
%*
exit /b %ERRORLEVEL%

:setup_msvc
set "BANC_VSINST="
set "BANC_VCVARS="
set "BANC_VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%BANC_VSWHERE%" goto :vswhere_done
"%BANC_VSWHERE%" -latest -products "*" -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -version "[17.0,)" -property installationPath > "%TEMP%\banc-vswhere.txt" 2>nul
if exist "%TEMP%\banc-vswhere.txt" set /p BANC_VSINST=<"%TEMP%\banc-vswhere.txt"
:vswhere_done
if defined BANC_VSINST if exist "%BANC_VSINST%\VC\Auxiliary\Build\vcvars64.bat" set "BANC_VCVARS=%BANC_VSINST%\VC\Auxiliary\Build\vcvars64.bat"
if not defined BANC_VCVARS if exist "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" (
  set "BANC_VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
  set "BANC_VSINST=C:\Program Files\Microsoft Visual Studio\18\Community"
)
if not defined BANC_VCVARS (
  echo Visual Studio C++ introuvable.
  echo Installez Visual Studio 2022 ou plus recent, charge de travail Developpement Desktop en C++.
  echo https://visualstudio.microsoft.com/fr/downloads/
  exit /b 1
)
call "%BANC_VCVARS%" >nul
if errorlevel 1 exit /b 1
if not defined VCPKG_ROOT if exist "%BANC_VSINST%\VC\vcpkg\vcpkg.exe" set "VCPKG_ROOT=%BANC_VSINST%\VC\vcpkg"
exit /b 0

:setup_vcpkg
if defined VSINSTALLDIR if exist "%VSINSTALLDIR%VC\vcpkg\vcpkg.exe" set "VCPKG_ROOT=%VSINSTALLDIR%VC\vcpkg"
if not defined VCPKG_ROOT if defined VSINSTALLDIR if exist "%VSINSTALLDIR%\VC\vcpkg\vcpkg.exe" set "VCPKG_ROOT=%VSINSTALLDIR%\VC\vcpkg"
if not defined VCPKG_ROOT echo Avertissement : VCPKG_ROOT introuvable. Les presets CMake en ont besoin.>&2
exit /b 0
