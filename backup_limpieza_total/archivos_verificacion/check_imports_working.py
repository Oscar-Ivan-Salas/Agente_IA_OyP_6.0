print("=== Starting Python Environment Check ===")
print("Hello from Python!")

# Test basic Python functionality
try:
    import sys
    print(f"Python version: {sys.version}")
    print(f"Executable: {sys.executable}")
    
    # Test file operations
    with open('test_output.txt', 'w') as f:
        f.write("This is a test file.\n")
    print("Successfully wrote to test_output.txt")
    
    # List current directory
    import os
    print("\nCurrent directory contents:")
    for item in os.listdir('.'):
        if os.path.isfile(item):
            print(f"- File: {item}")
        else:
            print(f"- Directory: {item}")
    
except Exception as e:
    print(f"Error: {str(e)}")

print("=== Environment Check Complete ===")
