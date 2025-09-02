print("Python environment test:")
print("-" * 30)

import sys
print(f"Python {sys.version}")
print(f"Executable: {sys.executable}")
print(f"\nCurrent directory: {sys.path[0]}")
print("\nPython path:")
for p in sys.path:
    print(f"  - {p}")

print("\nTesting basic imports:")
try:
    import fastapi
    print(f"✅ fastapi: {fastapi.__version__}")
except Exception as e:
    print(f"❌ fastapi: {e}")

try:
    import uvicorn
    print(f"✅ uvicorn: {uvicorn.__version__}")
except Exception as e:
    print(f"❌ uvicorn: {e}")

print("\nTesting gateway import:")
try:
    import gateway
    print(f"✅ gateway: {gateway.__file__}")
except Exception as e:
    print(f"❌ gateway: {e}")
    print("\nCurrent directory contents:")
    import os
    for item in os.listdir('.'):
        print(f"  - {item}")
