@echo off
chcp 65001 > nul
echo ========================================================
echo  🚀 SafeStep AI - Night Stair Vision FastAPI (:8000)
echo ========================================================
cd /d %~dp0
c:\project4\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
pause

