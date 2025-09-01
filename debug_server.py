import sys
import os
import subprocess

def run_command(command, cwd=None):
    """Run a command and return the output and error."""
    try:
        result = subprocess.run(
            command,
            cwd=cwd or os.getcwd(),
            shell=True,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return result.stdout, result.stderr, 0
    except subprocess.CalledProcessError as e:
        return e.stdout, e.stderr, e.returncode

def main():
    print("=== DEBUGGING AGENTE IA OYP 6.0 ===\n")
    
    # 1. Check Python version
    print("1. Python version:")
    print(f"   {sys.version}\n")
    
    # 2. Check virtual environment
    print("2. Virtual environment:")
    print(f"   Executable: {sys.executable}")
    print(f"   Prefix: {sys.prefix}\n")
    
    # 3. Check Python path
    print("3. Python path:")
    for p in sys.path:
        print(f"   {p}")
    print()
    
    # 4. Check if we can find gateway module
    print("4. Checking gateway module:")
    try:
        import gateway
        print(f"   ✅ gateway module found at: {gateway.__file__}")
        
        # Try to import the app
        try:
            from gateway.app import app
            print("   ✅ Successfully imported app from gateway.app")
            print(f"   App title: {app.title if hasattr(app, 'title') else 'No title attribute'}")
        except ImportError as e:
            print(f"   ❌ Error importing from gateway.app: {e}")
            print("   Available modules in gateway:", dir(gateway))
    except ImportError as e:
        print(f"   ❌ Error importing gateway: {e}")
        print("   Current directory contents:")
        for item in os.listdir('.'):
            print(f"     - {item}")
    print()
    
    # 5. Try to start the server directly
    print("5. Starting Uvicorn server...")
    try:
        import uvicorn
        print("   Uvicorn version:", uvicorn.__version__)
        print("   Starting server on http://127.0.0.1:8000")
        print("   Press Ctrl+C to stop\n")
        
        uvicorn.run(
            "gateway.app:app",
            host="127.0.0.1",
            port=8000,
            reload=True,
            ws="wsproto",
            log_level="debug"
        )
    except Exception as e:
        print(f"   ❌ Error starting Uvicorn: {e}")
        print("\nTroubleshooting steps:")
        print("1. Make sure you've activated the virtual environment")
        print("2. Run: pip install fastapi uvicorn[standard] wsproto")
        print("3. Ensure your current working directory is the project root")
        print("4. Check for any error messages above")

if __name__ == "__main__":
    main()
