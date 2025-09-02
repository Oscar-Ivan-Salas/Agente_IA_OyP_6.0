import sys
import os

print("Python Path:")
for path in sys.path:
    print(f"- {path}")

print("\nCurrent working directory:", os.getcwd())

print("\nTrying to import gateway.config...")
try:
    from gateway.config import settings
    print("Successfully imported gateway.config!")
    print("Settings:", settings.dict())
except ImportError as e:
    print(f"Error importing gateway.config: {e}")
    print("\nTrying to add project root to Python path...")
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        from gateway.config import settings
        print("Successfully imported after adding project root to path!")
        print("Settings:", settings.dict())
    except Exception as e2:
        print(f"Still couldn't import: {e2}")
