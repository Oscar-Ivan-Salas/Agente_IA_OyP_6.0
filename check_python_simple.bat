@echo off
echo === Python Environment Check ===
echo.

echo 1. Checking Python executable...
.venv_clean\Scripts\python.exe -c "print('Hello from Python!')"
echo.

echo 2. Creating a test file...
echo print("This is a test file") > test.py
echo print("Current directory:", __import__('os').getcwd()) >> test.py
echo print("Python version:", __import__('sys').version) >> test.py
echo.

echo 3. Running test file...
.venv_clean\Scripts\python.exe test.py
del test.py
echo.

echo 4. Checking installed packages...
.venv_clean\Scripts\pip.exe list
echo.

echo Check complete!
pause
