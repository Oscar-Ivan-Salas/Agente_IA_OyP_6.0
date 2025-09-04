import sys
import os

def main():
    print("=== Python Environment Check ===")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Test basic imports
    print("\n=== Testing Imports ===")
    try:
        import fastapi
        print(f"FastAPI version: {fastapi.__version__}")
    except ImportError as e:
        print(f"Error importing FastAPI: {e}")
    
    try:
        import uvicorn
        print(f"Uvicorn version: {uvicorn.__version__}")
    except ImportError as e:
        print(f"Error importing Uvicorn: {e}")
    
    # Test file operations
    print("\n=== Testing File Operations ===")
    test_file = "test_file_operation.txt"
    try:
        with open(test_file, "w") as f:
            f.write("Test successful!\n")
        print(f"Successfully wrote to {test_file}")
        
        with open(test_file, "r") as f:
            content = f.read()
        print(f"Successfully read from {test_file}")
        
        os.remove(test_file)
        print(f"Successfully deleted {test_file}")
    except Exception as e:
        print(f"File operation error: {e}")

if __name__ == "__main__":
    main()
