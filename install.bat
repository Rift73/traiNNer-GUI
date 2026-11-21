@echo off
setlocal DisableDelayedExpansion
set "trainer_distro="
for /f "tokens=1,2 delims=\" %%H in ("%~dp0") do (
    if /i "%%H"=="wsl.localhost" set "trainer_distro=%%I"
    if /i "%%H"=="wsl$" set "trainer_distro=%%I"
)
if defined trainer_distro goto wsl_install

pushd "%~dp0" || exit /b 1
if exist "venv\Scripts\python.exe" goto native_install
if exist "venv" (
    echo venv exists but has no Windows Python executable. Select the checkout for this platform. 1>&2
    popd
    exit /b 1
)
where py.exe >nul 2>&1
if errorlevel 1 (
    python -m venv venv
) else (
    py -3.12 -m venv venv
)
if errorlevel 1 goto done
:native_install
"venv\Scripts\python.exe" scripts\install_runtime.py %*
:done
set "trainer_result=%errorlevel%"
popd
exit /b %trainer_result%

:wsl_install
set "trainer_checkout=%~dp0."
setlocal EnableDelayedExpansion
rem wsl.exe treats unnecessary quotes around a simple distribution name literally.
if not "!trainer_distro: =!"=="!trainer_distro!" set trainer_distro="!trainer_distro!"
wsl.exe -d !trainer_distro! --cd "!trainer_checkout!" --exec bash ./install.sh %*
exit /b %errorlevel%
