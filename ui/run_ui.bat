@echo off
REM Launch the Phishing Message Detector web UI (opens in your browser)
cd /d "%~dp0"
python -m streamlit run app.py
pause
