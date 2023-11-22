@echo off
REM Change directory to the script's location
cd %~dp0

REM Activate the virtual environment
call venv\Scripts\activate.bat


python Download_Model.py
pause
