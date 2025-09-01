@echo off
echo === Running Python Script ===
echo.

echo 1. Creating test script...
echo import sys > test_script.py
echo print("Python version:", sys.version) >> test_script.py
echo print("Executable:", sys.executable) >> test_script.py
echo print("Path:", sys.path) >> test_script.py
echo.

echo 2. Running script with system Python...
python test_script.py
echo.

echo 3. Running script with virtual environment Python...
.venv_clean\Scripts\python.exe test_script.py
echo.

echo 4. Cleaning up...
del test_script.py
echo.

echo Test complete!
pause
