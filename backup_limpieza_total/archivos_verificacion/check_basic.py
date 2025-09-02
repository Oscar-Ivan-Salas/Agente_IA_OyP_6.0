import sys
import os

print("=== Basic Python Check ===")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print("\nEnvironment variables:")
for key in ['PATH', 'PYTHONPATH', 'VIRTUAL_ENV']:
    print(f"{key}: {os.environ.get(key, 'Not set')}")

print("\nTrying to write to a file...")
try:
    with open('test_write.txt', 'w') as f:
        f.write("Test successful!\n")
    print("Successfully wrote to test_write.txt")
except Exception as e:
    print(f"Error writing to file: {e}")

print("\nBasic Python check complete.")
