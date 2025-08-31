import sys
print("Python version:", sys.version)
print("\nPython path:")
for p in sys.path:
    print(f"- {p}")

print("\nTrying to import pydantic...")
try:
    import pydantic
    print("Pydantic version:", pydantic.__version__)
    print("Pydantic path:", pydantic.__file__)
    
    print("\nTrying to import _internal...")
    from pydantic import _internal
    print("_internal imported successfully!")
    
except ImportError as e:
    print(f"Error importing pydantic: {e}")
    
    print("\nTrying to install pydantic...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pydantic==1.10.13"])
