@echo off
echo === Python Installation Verification ===
echo.

echo 1. Checking Python executable...
where python
echo.

echo 2. Checking Python version...
python --version
echo.

echo 3. Running a simple Python command...
python -c "print('Hello from Python!')"
echo.

echo 4. Creating a test script...
echo print("This is a test script") > test_script.py
echo print("Python version:", __import__('sys').version) >> test_script.py
echo print("Current directory:", __import__('os').getcwd()) >> test_script.py
echo.

echo 5. Running the test script...
python test_script.py
del test_script.py
echo.

echo 6. Checking pip...
python -m pip --version
echo.

echo 7. Listing installed packages...
python -m pip list
echo.

echo Verification complete!
pause
