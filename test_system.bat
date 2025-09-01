@echo off
echo === System Information ===
echo.

echo 1. Current Directory:
cd
echo.

echo 2. Directory Contents:
dir /a
echo.

echo 3. Python Information:
where python
echo.
python --version
echo.

echo 4. Environment Variables:
echo PATH: %PATH%
echo.

echo 5. Python Test Script:
echo print("Hello from Python!") > test.py
echo print("Current directory:", __import__('os').getcwd()) >> test.py
echo print("Python version:", __import__('sys').version) >> test.py
echo.
echo 6. Running Test Script:
python test.py
del test.py
echo.

echo Test complete!
pause
