import os
import sys

print("=== Python Path ===")
for path in sys.path:
    print(f" - {path}")

print("\n=== Current Working Directory ===")
print(os.getcwd())

print("\n=== Directory Contents ===")
for item in os.listdir():
    print(f" - {item}")
