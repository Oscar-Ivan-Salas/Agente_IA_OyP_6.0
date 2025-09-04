@echo off
echo === Environment Test ===
echo.

echo 1. Creating test script...
echo import sys > test.py
echo import os >> test.py
echo print("Python version:", sys.version) >> test.py
echo print("Executable:", sys.executable) >> test.py
echo print("Current directory:", os.getcwd()) >> test.py
echo print("Python path:", sys.path) >> test.py
echo.

echo 2. Running test script...
python test.py
echo.

echo 3. Checking for test output...
if exist test_output.txt (
    type test_output.txt
    del test_output.txt
) else (
    echo No output file was created.
)

del test.py
echo.
echo Test complete!
pause
