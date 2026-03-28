@echo off
cd /d "%~dp0"
set PYTHON=C:\Users\Admin\AppData\Local\Python\pythoncore-3.14-64\python.exe

echo Closing old Streamlit on port 8501...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8501"') do taskkill /F /PID %%a >nul 2>&1
timeout /t 1 >nul

echo Starting TECHSCAN SET...
echo.
"%PYTHON%" -m streamlit run techscan_app.py --server.port 8501
pause
