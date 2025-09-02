import sys
import os

def main():
    with open('import_test_output.txt', 'w') as f:
        f.write("=== Import Test ===\n")
        f.write(f"Python: {sys.version}\n")
        f.write(f"Executable: {sys.executable}\n\n")
        
        packages = [
            'fastapi',
            'uvicorn',
            'websockets',
            'pydantic',
            'python_dotenv',
            'yaml'
        ]
        
        for pkg in packages:
            try:
                module = __import__(pkg)
                f.write(f"✓ {pkg} imported successfully")
                try:
                    f.write(f" (version: {module.__version__})\n")
                except AttributeError:
                    f.write(" (version not available)\n")
            except ImportError as e:
                f.write(f"✗ Failed to import {pkg}: {str(e)}\n")

if __name__ == "__main__":
    main()
