import sys
import os
from pathlib import Path

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("Current working directory:", os.getcwd())
print("\nPython path:")
for p in sys.path:
    print(f"  - {p}")

print("\nContents of gateway directory:")
gateway_path = Path("./gateway")
if gateway_path.exists():
    for item in gateway_path.iterdir():
        print(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
else:
    print("  - gateway directory not found in current directory")

print("\nTrying to import gateway...")
try:
    import gateway
    print(f"✅ Successfully imported gateway from: {gateway.__file__}")
    print("\nContents of gateway module:", dir(gateway))
    
    print("\nTrying to import from gateway.app...")
    try:
        from gateway.app import app
        print("✅ Successfully imported app from gateway.app")
        print("App title:", getattr(app, 'title', 'No title attribute'))
    except Exception as e:
        print(f"❌ Error importing from gateway.app: {e}")
        
except Exception as e:
    print(f"❌ Error importing gateway: {e}")
    
    # Additional debug for common issues
    print("\nChecking for common issues...")
    current_dir = Path.cwd()
    print(f"Current directory: {current_dir}")
    if str(current_dir) not in '\n'.join(sys.path):
        print("⚠️  Current directory not in Python path. This might cause import issues.")
    
    print("\nTrying to add current directory to path and import again...")
    sys.path.insert(0, str(current_dir))
    try:
        import gateway
        print(f"✅ Successfully imported gateway after adding current directory to path")
    except Exception as e2:
        print(f"❌ Still can't import gateway: {e2}")
