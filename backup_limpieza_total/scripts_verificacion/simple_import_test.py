import sys
import os

def check_pydantic():
    print("Python version:", sys.version)
    print("Current directory:", os.getcwd())
    
    print("\nPython path:")
    for p in sys.path:
        print(f"- {p}")
    
    print("\nTrying to import pydantic...")
    try:
        import pydantic
        print(f"Success! Pydantic version: {pydantic.__version__}")
        print(f"Pydantic location: {pydantic.__file__}")
    except Exception as e:
        print(f"Error importing pydantic: {e}")
    
    print("\nTrying to import pydantic._internal...")
    try:
        import pydantic._internal
        print("Successfully imported pydantic._internal")
    except Exception as e:
        print(f"Error importing pydantic._internal: {e}")

if __name__ == "__main__":
    check_pydantic()
