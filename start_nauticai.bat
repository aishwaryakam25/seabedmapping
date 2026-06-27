@echo off
REM One-click launcher for the NautiCAI seabed-mapping app.
REM Double-click this file, then open http://127.0.0.1:8000/review in your browser.
cd /d "%~dp0"

echo Starting the local AI model service (Ollama)...
start "" /min "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve
REM (If Ollama is already running, the line above just exits quietly.)

echo Waiting a few seconds for the AI service to come up...
timeout /t 4 /nobreak >nul

echo.
echo Starting NautiCAI server...
echo When you see "Uvicorn running", open your browser at:
echo     http://127.0.0.1:8000/review        (Data Review + shadow overlay)
echo     http://127.0.0.1:8000/triage         (Triage agent)
echo     http://127.0.0.1:8000/dive-plan      (ROV Dive Plan agent)
echo     http://127.0.0.1:8000/chatbot        (NautiCAI Chatbot)
echo.
echo Keep this window open during the demo. Close it to stop the server.
echo.

".venv\Scripts\python.exe" server.py
pause
