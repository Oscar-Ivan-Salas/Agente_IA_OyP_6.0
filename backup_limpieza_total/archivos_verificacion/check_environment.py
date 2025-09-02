import sys
import os
import platform
import subprocess

def get_package_versions():
    packages = [
        'fastapi',
        'uvicorn',
        'websockets',
        'pydantic',
        'wsproto',
        'watchfiles'
    ]
    
    print("\n=== Package Versions ===")
    for pkg in packages:
        try:
            module = __import__(pkg)
            version = getattr(module, '__version__', 'unknown')
            print(f"{pkg}: {version}")
        except ImportError:
            print(f"{pkg}: Not installed")

def get_python_info():
    print("\n=== Python Information ===")
    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print(f"Current Directory: {os.getcwd()}")

def get_environment_vars():
    print("\n=== Environment Variables ===")
    for key in ['PYTHONPATH', 'PATH', 'VIRTUAL_ENV']:
        print(f"{key}: {os.environ.get(key, 'Not set')}")

def check_websockets():
    print("\n=== WebSockets Check ===")
    try:
        import websockets
        print(f"WebSockets version: {websockets.__version__}")
        print(f"WebSockets path: {websockets.__file__}")
    except Exception as e:
        print(f"Error importing websockets: {e}")

if __name__ == "__main__":
    print("=== Environment Diagnostics ===")
    get_python_info()
    get_environment_vars()
    get_package_versions()
    check_websockets()
