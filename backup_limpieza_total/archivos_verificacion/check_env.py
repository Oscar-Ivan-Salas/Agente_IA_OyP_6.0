import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("\nPython path:")
for p in sys.path:
    if 'site-packages' in p or 'Agente_IA_OyP' in p:
        print(p)
