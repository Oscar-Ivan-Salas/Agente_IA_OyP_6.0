@echo off
echo === Python Path Check ===
echo.
echo 1. Checking Python path...
where python
echo.
echo 2. Checking Python version...
python --version
echo.
echo 3. Checking Python executable...
python -c "import sys; print('Python executable:', sys.executable)"
echo.
echo 4. Checking current directory...
python -c "import os; print('Current directory:', os.getcwd())"
echo.
echo Check complete!
pause
