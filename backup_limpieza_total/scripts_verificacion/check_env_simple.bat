@echo off
echo === Environment Check ===
echo.
echo [1/3] Checking Python...
.venv_clean\Scripts\python --version
echo.

echo [2/3] Checking pip...
.venv_clean\Scripts\pip --version
echo.

echo [3/3] Testing Python script execution...
echo print("Python is working!") > test_script.py
.venv_clean\Scripts\python test_script.py
del test_script.py
echo.

echo Check complete!
pause
