import sys
import os

def test_environment():
    print("Python version:", sys.version)
    print("Current working directory:", os.getcwd())
    print("Python path:", sys.path)
    
    # Try to import pytest
    try:
        import pytest
        print("\nPytest version:", pytest.__version__)
        print("Running pytest...\n")
        
        # Run pytest programmatically
        sys.exit(pytest.main(["-v"]))
    except ImportError:
        print("\nError: pytest is not installed. Please install it with: pip install pytest")
        return 1

if __name__ == "__main__":
    test_environment()
