import os

print("Current directory:", os.getcwd())
print("\nContents:")
for item in os.listdir('.'):
    print(f"- {item}")

print("\nPython version:", os.sys.version)
print("Python executable:", os.sys.executable)
