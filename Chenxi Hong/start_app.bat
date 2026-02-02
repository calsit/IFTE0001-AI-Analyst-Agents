@echo off
REM Windows startup script

REM Get the directory where the script is located
cd /d "%~dp0"

REM Start Streamlit application
echo 🚀 Starting Fundamental Analysis Web Application...
echo 📁 Working Directory: %CD%
echo.

streamlit run src/app.py

pause

