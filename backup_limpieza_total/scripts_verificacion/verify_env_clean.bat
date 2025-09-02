@echo off
echo === Environment Verification ===
echo.

echo 1. Checking Python in PATH...
where python
echo.

echo 2. Checking Python version...
python --version
echo.

echo 3. Running a simple Python command...
python -c "import sys; print('Python executable:', sys.executable)"
echo.

echo 4. Checking virtual environment...
if exist .venv_clean\Scripts\python.exe (
    echo Found .venv_clean virtual environment
    .venv_clean\Scripts\python.exe --version
) else (
    echo .venv_clean virtual environment not found
)
echo.

echo 5. Creating a test script...
echo import sys > test_script.py
echo print("Python version:", sys.version) >> test_script.py
echo print("Executable:", sys.executable) >> test_script.py
echo print("Path:", sys.path) >> test_script.py
echo.

echo 6. Running the test script...
python test_script.py
del test_script.py
echo.

echo 7. Checking installed packages...
python -m pip list
echo.

echo Verification complete!
pause
