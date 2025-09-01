# Check Python environment and paths

# Go to the root directory
Set-Location C:\Users\USUARIO\Agente_IA_OyP_6.0

# Activate the virtual environment
. \.venv\Scripts\Activate.ps1

# Check Python version and paths
python -c "
import os
import sys

print('='*50)
print('Python Environment Check')
print('='*50)
print(f'Python Executable: {sys.executable}')
print(f'Python Version: {sys.version}')
print(f'Current Working Directory: {os.getcwd()}')
print(f'PYTHONPATH: {os.environ.get("PYTHONPATH", "Not set")}')
print('\nPython Path:')
for p in sys.path:
    print(f'  - {p}')

# Try to import gateway
try:
    import gateway
    print('\n✅ Successfully imported gateway module')
    print(f'Gateway module path: {gateway.__file__}')
    
    # Check config
    try:
        from gateway.core import config
        print('✅ Successfully imported config')
        print(f'Config settings: {config.settings}')
    except Exception as e:
        print(f'❌ Error importing config: {e}')
        
except Exception as e:
    print(f'\n❌ Error importing gateway: {e}')
"
