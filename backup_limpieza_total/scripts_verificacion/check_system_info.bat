@echo off
echo === System Information ===
echo.
systeminfo | findstr /B /C:"OS Name" /C:"OS Version" /C:"System Type" /C:"System Directory"
echo.
echo === Python Information ===
echo.
where python
echo.
python --version
echo.
echo === Environment Variables ===
echo.
echo PATH: %PATH%
echo.
pause
