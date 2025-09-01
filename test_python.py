import sys
import os

def main():
    with open('python_test_output.txt', 'w') as f:
        f.write("=== Python Test ===\n")
        f.write(f"Python executable: {sys.executable}\n")
        f.write(f"Python version: {sys.version}\n\n")
        
        f.write("=== Environment Variables ===\n")
        for key in ['VIRTUAL_ENV', 'PYTHONPATH', 'PATH']:
            f.write(f"{key}: {os.environ.get(key, 'Not set')}\n")
        
        f.write("\n=== Python Path ===\n")
        for p in sys.path:
            f.write(f"{p}\n")
        
        f.write("\n=== Current Directory Contents ===\n")
        try:
            for item in os.listdir('.'):
                f.write(f"{item}\n")
        except Exception as e:
            f.write(f"Error listing directory: {e}\n")

if __name__ == "__main__":
    main()
