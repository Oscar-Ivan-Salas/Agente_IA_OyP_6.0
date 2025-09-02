import sys
import os
import platform

def main():
    print("=== Environment Check ===")
    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print(f"Current Directory: {os.getcwd()}")
    
    print("\n=== Environment Variables ===")
    for key in ['PYTHONPATH', 'PATH', 'VIRTUAL_ENV']:
        print(f"{key}: {os.environ.get(key, 'Not set')}")
    
    print("\n=== Installed Packages ===")
    try:
        import pkg_resources
        for dist in pkg_resources.working_set:
            print(f"{dist.key}=={dist.version}")
    except ImportError:
        print("pkg_resources not available")

if __name__ == "__main__":
    main()
