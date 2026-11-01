@echo off
REM Launch the Phishing Message Detector desktop app (no console window)
cd /d "%~dp0"
start "" pythonw main.py
