@echo off
echo === System Information ===
echo.

echo 1. Operating System:
systeminfo | findstr /B /C:"OS Name" /C:"OS Version" /C:"System Type"
echo.

echo 2. Python Information:
where python
echo.
python --version
echo.

echo 3. Environment Variables:
echo PATH: %PATH%
echo.

echo 4. Current Directory:
cd
echo.

echo 5. Directory Contents:
dir /b
echo.

echo 6. Python Test:
echo print("Hello from Python!") > test.py
python test.py
del test.py
echo.

echo Information gathering complete!
pause
