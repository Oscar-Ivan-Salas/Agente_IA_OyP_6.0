@echo off
echo Simple test started > test_output.txt
echo Current date: %date% %time% >> test_output.txt
dir /b >> test_output.txt
echo Test completed >> test_output.txt
type test_output.txt
del test_output.txt
pause
