import sys
import os
import subprocess

def check_python():
    print("=== Python Environment Check ===")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    
    print("\n=== Python Path ===")
    for i, path in enumerate(sys.path, 1):
        print(f"{i}. {path}")
    
    print("\n=== Environment Variables ===")
    for key, value in os.environ.items():
        if 'PYTHON' in key or 'PATH' in key:
            print(f"{key} = {value}")
    
    print("\n=== Installed Packages ===")
    try:
        import pkg_resources
        installed_packages = sorted([f"{d.key}=={d.version}" for d in pkg_resources.working_set])
        for pkg in installed_packages:
            print(pkg)
    except Exception as e:
        print(f"Could not list installed packages: {e}")
    
    print("\n=== Pydantic Check ===")
    try:
        import pydantic
        print(f"Pydantic version: {pydantic.__version__}")
        print(f"Pydantic location: {pydantic.__file__}")
        
        try:
            from pydantic import _internal
            print("Pydantic _internal imported successfully!")
        except ImportError as e:
            print(f"Error importing pydantic._internal: {e}")
    except ImportError as e:
        print(f"Pydantic not installed or error: {e}")
        print("\n=== Installing Pydantic ===")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pydantic==1.10.13"])
            print("Pydantic installed successfully!")
        except Exception as e:
            print(f"Error installing pydantic: {e}")

if __name__ == "__main__":
    check_python()
