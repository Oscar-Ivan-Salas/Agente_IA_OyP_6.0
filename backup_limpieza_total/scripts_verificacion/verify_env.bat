@echo off
echo === Environment Verification ===
echo.

echo 1. System Information:
systeminfo | findstr /B /C:"OS Name" /C:"OS Version" /C:"System Type"
echo.

echo 2. Python Information:
where python
echo.
python --version
echo.

echo 3. Environment Variables:
echo PATH=%PATH%
echo.

echo 4. Directory Contents:
dir /a /b
echo.

echo 5. Virtual Environment Check:
if exist .venv_clean\Scripts\python.exe (
    echo Virtual environment found at .venv_clean
    .venv_clean\Scripts\python.exe --version
) else (
    echo No virtual environment found at .venv_clean
)
echo.

echo 6. Testing Python Execution:
echo print("Hello from Python!") > test.py
python test.py
del test.py
echo.

echo Verification complete!
pause
