@echo off
echo Running Python environment test...
python -c "import sys; print('Python version:', sys.version); print('Executable:', sys.executable)" > test_output.txt 2>&1
type test_output.txt
pause
