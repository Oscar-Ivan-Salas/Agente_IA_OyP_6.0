import sys
import os

print("=== Environment Test ===")
print(f"Python executable: {sys.executable}")
print(f"Working directory: {os.getcwd()}")
print("\nPython path:")
for p in sys.path:
    print(f"  - {p}")

print("\n=== Testing imports ===")
try:
    import fastapi
    print(f"✅ FastAPI version: {fastapi.__version__}")
except ImportError as e:
    print(f"❌ FastAPI import error: {e}")

try:
    import sqlalchemy
    print(f"✅ SQLAlchemy version: {sqlalchemy.__version__}")
except ImportError as e:
    print(f"❌ SQLAlchemy import error: {e}")

print("\n=== Testing project imports ===")
try:
    from gateway import app
    print("✅ Successfully imported gateway.app")
    print(f"App title: {app.title}")
except Exception as e:
    print(f"❌ Error importing gateway.app: {e}")
    import traceback
    traceback.print_exc()
