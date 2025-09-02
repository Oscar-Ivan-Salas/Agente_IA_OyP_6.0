import sys
import os

def main():
    # Basic info
    print("=== Python Environment ===")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Write to a file
    try:
        with open('test_output.txt', 'w') as f:
            f.write("This is a test file.\n")
            f.write(f"Python version: {sys.version}\n")
        print("\nSuccessfully wrote to test_output.txt")
    except Exception as e:
        print(f"\nError writing to file: {e}")
    
    # List current directory
    try:
        print("\nCurrent directory contents:")
        for item in os.listdir('.'):
            print(f"- {item}")
    except Exception as e:
        print(f"\nError listing directory: {e}")

if __name__ == "__main__":
    main()
