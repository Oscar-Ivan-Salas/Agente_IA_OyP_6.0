@echo off
echo === Environment Check ===
echo.
echo [1/4] Checking Python installation...
python --version
echo.

echo [2/4] Checking pip installation...
python -m pip --version
echo.

echo [3/4] Checking installed packages...
python -m pip list
echo.

echo [4/4] Testing Python script execution...
echo print("Python is working!") > test_script.py
python test_script.py
del test_script.py
echo.

echo Check complete!
pause
