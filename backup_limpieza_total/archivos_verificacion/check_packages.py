import sys
import os
import subprocess

def run_command(command):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    print("=== Python Environment ===")
    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}")
    print(f"Working Directory: {os.getcwd()}")
    
    print("\n=== Virtual Environment ===")
    print(f"VIRTUAL_ENV: {os.environ.get('VIRTUAL_ENV', 'Not in a virtual environment')}")
    
    print("\n=== Installed Packages ===")
    print(run_command(f"{sys.executable} -m pip list"))
    
    print("\n=== WebSockets Info ===")
    print(run_command(f"{sys.executable} -m pip show websockets"))

if __name__ == "__main__":
    main()
