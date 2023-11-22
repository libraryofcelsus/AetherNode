@echo off
REM Change directory to the script's location
cd %~dp0

REM Activate the virtual environment
call venv\Scripts\activate.bat

REM Use Python to read "Use_Public_API" value from settings.json
python -c "import json; print(json.load(open('settings.json'))['Use_Public_API'])" > temp.txt
set /p Use_Public_Api=<temp.txt
del temp.txt


REM Start the FastAPI app with uvicorn
REM Make sure to replace 'aethernode' with the actual name of your FastAPI app file without the .py extension
REM Also, ensure that uvicorn is installed in your environment
start /b uvicorn AetherNode_GPTQ:app --host 127.0.0.1 --port 8000

echo AetherNode_GPTQ.py is now serving the app...


REM Check if Use_Public_API is set to True
if "%Use_Public_Api%"=="True" (
    echo Starting ngrok to create a public URL...
    REM Start ngrok to expose the FastAPI app to a public URL
    REM Ensure ngrok is installed and accessible from the command line
    REM The following line will open a new command prompt window running ngrok
    start ngrok http 8000
) else (
    echo Public API not enabled. Running locally only.
)


pause
