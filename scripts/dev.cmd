@echo off
rem Usage : scripts\dev.cmd <commande...>
rem Lance la commande dans l'environnement MSVC x64 (cl, ninja, vcpkg).
if not defined VSCMD_VER (
  call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
)
if not defined VCPKG_ROOT set "VCPKG_ROOT=C:\Program Files\Microsoft Visual Studio\18\Community\VC\vcpkg"
%*
