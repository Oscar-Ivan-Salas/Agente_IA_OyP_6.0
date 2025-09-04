import sys
import os

print("=== Python Environment Check ===")
print(f"Python Version: {sys.version}")
print(f"\nPython Executable: {sys.executable}")
print(f"Working Directory: {os.getcwd()}")

print("\n=== System Path ===")
for path in sys.path:
    print(f" - {path}")

print("\n=== Environment Variables ===")
for key, value in os.environ.items():
    if 'PYTHON' in key.upper() or 'PATH' in key.upper():
        print(f"{key} = {value}")

print("\n=== Testing Imports ===")
try:
    import fastapi
    print("✅ FastAPI is installed")
except ImportError:
    print("❌ FastAPI is NOT installed")

try:
    import uvicorn
    print("✅ Uvicorn is installed")
except ImportError:
    print("❌ Uvicorn is NOT installed")
