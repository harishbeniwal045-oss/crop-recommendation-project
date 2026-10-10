@echo off
REM One-click start for Windows: creates a virtual environment on first run.
cd /d "%~dp0"
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate
)
echo.
echo Open http://127.0.0.1:5000 in your browser. Press Ctrl+C to stop.
python app.py
pause
