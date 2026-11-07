@echo off
chcp 65001 > nul
echo ========================================================
echo  🌙 SafeStep AI - Night Stair Vision React Vite (:5173)
echo ========================================================
cd /d %~dp0\frontend
cmd.exe /c "npm run dev"
pause

