import sys
print("Testing Python environment...")
print("Python executable:", sys.executable)
print("Python version:", sys.version)

try:
    import pydantic
    print("\nPydantic is installed!")
    print("Pydantic version:", pydantic.__version__)
    print("Pydantic location:", pydantic.__file__)
    
    # Try to import _internal directly
    from pydantic import _internal
    print("Successfully imported pydantic._internal")
    
except ImportError as e:
    print("\nError importing pydantic:", str(e))
