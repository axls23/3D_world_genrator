@echo off
setlocal

rem Run from this folder so WSL can use the current directory under /mnt.
pushd "%~dp0"

rem With no arguments, pass a selected USB device through to WSL and start scanning.
if "%~1"=="" (
    call :AttachUsb
    if errorlevel 1 exit /b 1
    set "STREAM_ARGS=--camera usb --wait-usb"
) else (
    set "STREAM_ARGS=%*"
)

wsl.exe -d Ubuntu -- bash -lc "source /home/chron/miniforge3/etc/profile.d/conda.sh && conda activate /home/chron/3dgrut && export HF_HOME=/home/chron/hf_cache HYPERSPLAT_ACEZERO_PYTHON=/home/chron/.conda_envs/ace0/bin/python CUDA_HOME=/home/chron/3dgrut CPATH=/home/chron/3dgrut/targets/x86_64-linux/include CC=/home/chron/3dgrut/bin/x86_64-conda-linux-gnu-gcc CXX=/home/chron/3dgrut/bin/x86_64-conda-linux-gnu-g++ && python3 -c \"from pathlib import Path; p=Path('run_stream_demo.sh'); Path('.run_stream_demo.wsl.sh').write_bytes(p.read_bytes().replace(bytes([13]), b''))\" && trap 'rm -f .run_stream_demo.wsl.sh' EXIT && bash \"$PWD/.run_stream_demo.wsl.sh\" %STREAM_ARGS%"
set "RESULT=%ERRORLEVEL%"

popd
exit /b %RESULT%

:AttachUsb
where.exe usbipd.exe >nul 2>nul
if errorlevel 1 (
    echo [stream-demo] usbipd-win is not installed or is not on PATH.
    echo Install it from an Administrator PowerShell with:
    echo   winget install --exact dorssel.usbipd-win
    exit /b 1
)

rem Keep Ubuntu running so usbipd can attach the device to the WSL distribution.
start "Ubuntu USB camera passthrough" wsl.exe -d Ubuntu
timeout /t 2 /nobreak >nul

echo.
echo Select the camera or phone BUSID from this list, for example 2-3:
usbipd.exe list
set "BUSID="
set /p "BUSID=Camera/phone BUSID (blank to cancel): "
if not defined BUSID (
    echo [stream-demo] USB passthrough cancelled.
    exit /b 1
)

set "CAMERA_BUSID=%BUSID%"
echo [stream-demo] Requesting Administrator approval to bind USB device %BUSID%...
powershell.exe -NoProfile -Command "$usbipd = (Get-Command usbipd.exe -ErrorAction Stop).Source; $p = Start-Process -FilePath $usbipd -ArgumentList @('bind','--busid',$env:CAMERA_BUSID) -Verb RunAs -Wait -PassThru; exit $p.ExitCode"
if errorlevel 1 (
    echo [stream-demo] USB binding failed or Administrator approval was cancelled.
    exit /b 1
)

usbipd.exe attach --wsl --busid "%BUSID%"
if errorlevel 1 (
    echo [stream-demo] USB attach failed. Keep Ubuntu running and check the BUSID.
    exit /b 1
)
exit /b 0
