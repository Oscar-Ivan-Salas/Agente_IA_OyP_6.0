import sys
import os

def main():
    print("=== Simple Environment Check ===")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    print("\n=== Environment Variables ===")
    print(f"VIRTUAL_ENV: {os.environ.get('VIRTUAL_ENV', 'Not set')}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    
    print("\n=== Python Path ===")
    for i, path in enumerate(sys.path, 1):
        if 'site-packages' in path or 'Agente_IA_OyP' in path:
            print(f"{i}. {path}")

if __name__ == "__main__":
    main()
