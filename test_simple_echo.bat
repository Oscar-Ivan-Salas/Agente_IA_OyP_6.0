@echo off
echo === Simple Echo Test ===
echo.
echo 1. Testing echo command...
echo This is a test message.
echo.
echo 2. Creating a test file...
echo This is a test file. > test.txt
type test.txt
del test.txt
echo.
echo 3. Current directory:
cd
echo.
echo Test complete!
pause
