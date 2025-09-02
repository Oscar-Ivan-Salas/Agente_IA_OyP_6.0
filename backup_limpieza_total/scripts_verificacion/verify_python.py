import sys
import os

def main():
    print("=== Python Environment Verification ===")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    print("\n=== Environment Variables ===")
    print(f"VIRTUAL_ENV: {os.environ.get('VIRTUAL_ENV', 'Not set')}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    
    print("\n=== Python Path ===")
    for i, path in enumerate(sys.path, 1):
        print(f"{i}. {path}")
    
    print("\n=== Testing Imports ===")
    packages = ['fastapi', 'uvicorn', 'websockets', 'pydantic', 'python_dotenv', 'yaml']
    
    for pkg in packages:
        try:
            module = __import__(pkg)
            print(f"✅ {pkg} imported successfully")
            try:
                print(f"   Version: {module.__version__}")
            except AttributeError:
                print("   Version: Not available")
        except ImportError as e:
            print(f"❌ Failed to import {pkg}: {e}")

if __name__ == "__main__":
    main()
