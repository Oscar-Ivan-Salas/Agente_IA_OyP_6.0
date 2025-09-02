print("=== Simple Python Verification ===")
print("If you see this message, Python is working!")

# Try to write to a file
with open('test_output.txt', 'w') as f:
    f.write("This is a test file.\n")
    f.write("If you can see this, file operations are working.\n")

print("\n=== Environment Information ===")
import sys
import os

print(f"Python version: {sys.version}")
print(f"Executable: {sys.executable}")
print(f"Working directory: {os.getcwd()}")
print(f"Virtual environment: {os.environ.get('VIRTUAL_ENV', 'Not in a virtual environment')}")

print("\n=== Simple Import Test ===")
try:
    import fastapi
    print(f"✅ FastAPI is installed: {fastapi.__version__}")
except ImportError:
    print("❌ FastAPI is not installed")

try:
    import uvicorn
    print(f"✅ Uvicorn is installed: {uvicorn.__version__}")
except ImportError:
    print("❌ Uvicorn is not installed")

print("\nVerification complete. Check test_output.txt for file operation test.")
