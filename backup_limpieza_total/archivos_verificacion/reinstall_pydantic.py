import sys
import subprocess
import os

def reinstall_pydantic():
    print("Uninstalling pydantic...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "pydantic", "pydantic-core", "pydantic-settings"])
    
    print("\nInstalling specific version of pydantic...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pydantic==1.10.13"])
    
    print("\nVerifying installation...")
    result = subprocess.run([sys.executable, "-c", "import pydantic; print(f'Pydantic version: {pydantic.__version__}'); from pydantic import _internal; print('Successfully imported _internal')"], capture_output=True, text=True)
    
    print("\nVerification output:")
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)

if __name__ == "__main__":
    reinstall_pydantic()
