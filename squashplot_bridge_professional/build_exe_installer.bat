@echo off
echo SquashPlot Bridge - EXE Installer Builder
echo ==========================================

echo.
echo This script will create a standalone executable installer
echo that end users can run without needing Python installed.
echo.

echo Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo.
echo Installing required packages...
pip install pyinstaller

echo.
echo Building executable installer...
python SquashPlotBridgeEXEInstaller.py

echo.
echo ==========================================
echo EXE Installer Build Complete!
echo ==========================================
echo.
echo The installer package is ready in:
echo executable_installer/package/
echo.
echo Files included:
echo - SquashPlotBridgeInstaller.exe (for end users)
echo - README.txt (user documentation)
echo - QUICK_START.txt (quick start guide)
echo.
echo You can now distribute these files to end users.
echo.

pause



