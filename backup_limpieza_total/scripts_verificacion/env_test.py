import sys
import os

def main():
    # Create output file
    with open('env_test_output.txt', 'w', encoding='utf-8') as f:
        f.write("=== Environment Test ===\n\n")
        
        # Basic Python info
        f.write(f"Python Executable: {sys.executable}\n")
        f.write(f"Python Version: {sys.version}\n")
        f.write(f"Working Directory: {os.getcwd()}\n\n")
        
        # Check Pydantic
        try:
            import pydantic
            f.write("Pydantic is installed!\n")
            f.write(f"Version: {pydantic.__version__}\n")
            f.write(f"Location: {pydantic.__file__}\n")
            
            # Try to import _internal
            try:
                from pydantic import _internal
                f.write("Successfully imported pydantic._internal\n")
            except ImportError as e:
                f.write(f"Error importing pydantic._internal: {str(e)}\n")
                
        except ImportError as e:
            f.write(f"Pydantic not installed: {str(e)}\n")
            
        # List site-packages
        f.write("\n=== Site Packages ===\n")
        for path in sys.path:
            if 'site-packages' in path:
                f.write(f"- {path}\n")
                
        # List directory contents
        f.write("\n=== Current Directory Contents ===\n")
        try:
            for item in os.listdir('.'):
                f.write(f"- {item}\n")
        except Exception as e:
            f.write(f"Error listing directory: {str(e)}\n")

if __name__ == "__main__":
    main()
    print("Test complete. Check env_test_output.txt for results.")
