@echo off
echo === Simple Command Test ===
echo.
echo 1. Running 'ver' command...
ver
echo.
echo 2. Running 'echo' command...
echo Hello, World!
echo.
echo 3. Running 'dir' command...
dir /b
echo.
echo 4. Creating a test file...
echo This is a test file > test.txt
type test.txt
del test.txt
echo.
echo Test complete!
pause
