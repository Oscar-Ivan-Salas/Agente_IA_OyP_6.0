@echo off
echo === Python Environment Test ===
echo.

echo 1. Checking Python version...
python --version
echo.

echo 2. Running a simple Python command...
python -c "print('Hello from Python!')"
echo.

echo 3. Creating and running a test Python script...
echo print("This is a test script") > test_script.py
echo print("Python version:", __import__('sys').version) >> test_script.py
echo print("Current directory:", __import__('os').getcwd()) >> test_script.py
python test_script.py
del test_script.py
echo.

echo 4. Checking pip...
python -m pip --version
echo.

echo Test complete!
pause
