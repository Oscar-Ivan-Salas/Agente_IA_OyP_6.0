@echo off
echo === Python Direct Check ===

:: Check if Python is in PATH
where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Python is in PATH
    python --version
) else (
    echo Python is NOT in PATH
)

echo.
echo === Virtual Environment Check ===
if exist .venv_clean\Scripts\python.exe (
    echo Found .venv_clean virtual environment
    .venv_clean\Scripts\python.exe --version
) else (
    echo .venv_clean virtual environment not found
)

if exist .venv\Scripts\python.exe (
    echo Found .venv virtual environment
    .venv\Scripts\python.exe --version
) else (
    echo .venv virtual environment not found
)

if exist .venv_new\Scripts\python.exe (
    echo Found .venv_new virtual environment
    .venv_new\Scripts\python.exe --version
) else (
    echo .venv_new virtual environment not found
)

echo.
echo === System Information ===
systeminfo | findstr /B /C:"OS Name" /C:"OS Version" /C:"System Type"

echo.
echo === Environment Variables ===
echo PATH: %PATH%

pause
