@echo off
REM Change directory to the script's location
cd %~dp0

REM Activate the virtual environment
call venv\Scripts\activate.bat

REM Check if the model directory exists and contains the model files
python Download_Model.py

REM Start the FastAPI app with uvicorn
REM Make sure to replace 'aethernode' with the actual name of your FastAPI app file without the .py extension
REM Also, ensure that uvicorn is installed in your environment
uvicorn AetherNode_ExLlama2:app --host 127.0.0.1 --port 8000 --reload

echo AetherNode_ExLlama2.py is now serving the app...
pause
