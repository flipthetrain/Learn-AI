@echo off
REM Windows batch script to check for Python and install if missing
where python >nul 2>nul
if %ERRORLEVEL%==0 (
    echo Python is already installed.
) else (
    echo Python is not installed. Downloading and installing Python 3...
    powershell -Command "Start-Process 'https://www.python.org/ftp/python/3.11.8/python-3.11.8-amd64.exe' -Wait"
    echo Please run the installer and re-run this script after installation.
    exit /b 1
)
python --version

REM Call imports.py to install all required Python packages
python "imports.py"
