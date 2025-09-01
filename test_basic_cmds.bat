@echo off
echo === Basic Command Test ===
echo.

echo 1. Testing 'ver' command:
ver
echo.

echo 2. Testing 'echo' command:
echo Hello, World!
echo.

echo 3. Testing 'dir' command:
dir /b
echo.

echo 4. Testing 'cd' command:
cd
echo.

echo 5. Testing 'type' command:
echo This is a test file. > test_file.txt
type test_file.txt
del test_file.txt
echo.

echo Test complete!
pause
