@echo off
echo === Test Output Redirection ===
echo.

echo 1. Output to console:
echo This is a test message.
echo.

echo 2. Output to file:
echo This is a test message. > output.txt
type output.txt
del output.txt
echo.

echo 3. Appending to file:
echo First line > test.txt
echo Second line >> test.txt
type test.txt
del test.txt
echo.

echo 4. Error output:
echo This is an error message 1>&2
echo.

echo 5. Combined output:
echo This is a test message. > combined.txt 2>&1
type combined.txt
del combined.txt
echo.

echo Test complete!
pause
