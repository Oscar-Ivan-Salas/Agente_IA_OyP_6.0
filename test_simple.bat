@echo off
echo === Simple Test ===
echo.
echo 1. Creating test file...
echo print("Hello from Python!") > test.py
echo print("Current directory:", __import__('os').getcwd()) >> test.py
echo print("Python version:", __import__('sys').version) >> test.py
echo.
echo 2. Running test file...
python test.py
echo.
echo 3. Cleaning up...
del test.py
echo.
echo Test complete!
pause
