import sys
import os

# Push site-packages to the front to prioritize the real package
for p in list(sys.path):
    if 'site-packages' in p and p in sys.path:
        sys.path.remove(p)
        sys.path.insert(0, p)
