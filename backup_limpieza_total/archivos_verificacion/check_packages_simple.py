import sys
import os

def main():
    print("=== Environment Check ===")
    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}")
    print(f"Working Directory: {os.getcwd()}")
    
    print("\n=== Virtual Environment ===")
    print(f"VIRTUAL_ENV: {os.environ.get('VIRTUAL_ENV', 'Not in a virtual environment')}")
    
    print("\n=== Installed Packages ===")
    try:
        import pkg_resources
        installed_packages = pkg_resources.working_set
        for pkg in installed_packages:
            print(f"{pkg.key}=={pkg.version}")
    except ImportError:
        print("pkg_resources not available")

if __name__ == "__main__":
    main()
