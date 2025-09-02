@echo off
echo === Python Environment Verification ===
echo.

echo [1/4] Checking Python executable...
.venv_clean\Scripts\python.exe --version
echo.

echo [2/4] Checking pip...
.venv_clean\Scripts\pip.exe --version
echo.

echo [3/4] Creating a test script...
echo print("Python is working! Current directory:", __import__('os').getcwd()) > test_script.py
echo print("Python path:", __import__('sys').path) >> test_script.py
echo.

echo [4/4] Running test script...
.venv_clean\Scripts\python.exe test_script.py
del test_script.py
echo.

echo Verification complete!
pause
