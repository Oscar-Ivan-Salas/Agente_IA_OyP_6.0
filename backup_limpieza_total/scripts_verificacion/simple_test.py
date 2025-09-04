with open('test_output.txt', 'w') as f:
    f.write("Python is working!\n")
    f.write(f"Python version: {sys.version}\n")
    f.write(f"Current directory: {os.getcwd()}\n")
    
    try:
        import pydantic
        f.write(f"Pydantic version: {pydantic.__version__}\n")
    except ImportError as e:
        f.write(f"Pydantic import error: {str(e)}\n")
