@echo off
REM Change directory to the script's location
cd %~dp0

REM Activate the virtual environment
call venv\Scripts\activate.bat

REM Start the FastAPI app with uvicorn
REM Make sure to replace 'aethernode' with the actual name of your FastAPI app file without the .py extension
REM Also, ensure that uvicorn is installed in your environment
uvicorn AetherNode:app --host 0.0.0.0 --port 8000 --reload

echo AetherNode.py is now serving the app...
pause
