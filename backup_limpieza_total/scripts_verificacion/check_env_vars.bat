@echo off
echo === Environment Variables ===
echo.
echo 1. PATH:
echo %PATH%
echo.

echo 2. PYTHONPATH:
echo %PYTHONPATH%
echo.

echo 3. VIRTUAL_ENV:
echo %VIRTUAL_ENV%
echo.

echo 4. System Information:
systeminfo | findstr /B /C:"OS Name" /C:"OS Version" /C:"System Type"
echo.

echo 5. Python Information:
where python
echo.
python --version
echo.

echo 6. Running a simple Python command...
python -c "print('Hello from Python!')"
echo.

echo Check complete!
pause
