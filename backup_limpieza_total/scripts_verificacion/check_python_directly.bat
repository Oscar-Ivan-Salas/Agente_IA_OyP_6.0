@echo off
echo === Python Direct Check ===
echo.

echo 1. Checking Python in PATH:
where python
echo.

echo 2. Checking Python version:
python --version
echo.

echo 3. Running a simple Python command:
python -c "print('Hello from Python!')"
echo.

echo 4. Checking pip:
python -m pip --version
echo.

echo 5. Listing installed packages:
python -m pip list
echo.

echo Check complete!
pause
