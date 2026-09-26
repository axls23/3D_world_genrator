@echo off
cd /d "%~dp0"
call conda activate 3dgrut
python demo_server.py
pause
