with open('env_check.txt', 'w') as f:
    import sys
    f.write(f"Python {sys.version}\n")
    f.write(f"Executable: {sys.executable}\n")
    
    try:
        import pydantic
        f.write(f"\nPydantic {pydantic.__version__} is installed at:\n{pydantic.__file__}\n")
    except ImportError as e:
        f.write("\nPydantic is NOT installed or there was an error importing it.\n")
        f.write(f"Error: {str(e)}\n")
    
    f.write("\nPython path:\n")
    for p in sys.path:
        f.write(f"- {p}\n")
