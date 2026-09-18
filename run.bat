@echo off
cd /d "%~dp0"
echo Starting Mastitis AI System...

echo [1/2] Launching Backend API on http://127.0.0.1:8000 ...
start "Mastitis Backend" cmd /k ".venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload"

timeout /t 3 /nobreak >nul

echo [2/2] Launching Frontend on http://127.0.0.1:5173 ...
cd frontend
start "Mastitis Frontend" cmd /k "npm run dev -- --host 0.0.0.0"

timeout /t 2 /nobreak >nul

start http://127.0.0.1:5173

echo ========================================================
echo Mastitis AI application is running!
echo Frontend:         http://127.0.0.1:5173
echo Backend API Docs: http://127.0.0.1:8000/docs
echo Default Login:    farmer / farmer
echo ========================================================

