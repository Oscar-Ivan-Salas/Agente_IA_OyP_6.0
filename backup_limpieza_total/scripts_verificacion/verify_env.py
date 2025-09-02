import sys
import os
import subprocess

def run_command(cmd):
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr.strip()}"

def main():
    python_exe = sys.executable
    print(f"Using Python: {python_exe}")
    print(f"Python version: {sys.version}")
    
    # Check pip version
    print("\n=== pip version ===")
    print(run_command(f'"{python_exe}" -m pip --version'))
    
    # List installed packages
    print("\n=== Installed Packages ===")
    print(run_command(f'"{python_exe}" -m pip list'))
    
    # Check specific packages
    packages = ['fastapi', 'uvicorn', 'websockets', 'pydantic', 'python-dotenv', 'pyyaml']
    print("\n=== Package Versions ===")
    for pkg in packages:
        print(f"{pkg}: {run_command(f'"{python_exe}" -m pip show {pkg} | findstr Version')}")
    
    # Check if we can import them
    print("\n=== Import Tests ===")
    for pkg in packages:
        try:
            module = __import__(pkg)
            print(f"Successfully imported {pkg} (version: {getattr(module, '__version__', 'unknown')})")
        except ImportError as e:
            print(f"Failed to import {pkg}: {e}")

if __name__ == "__main__":
    main()
