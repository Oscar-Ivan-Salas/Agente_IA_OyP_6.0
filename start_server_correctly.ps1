# Start the server with all necessary configurations

# Go to the project root
Set-Location C:\Users\USUARIO\Agente_IA_OyP_6.0

# 1. Activate the virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
. \.venv\Scripts\Activate.ps1

# 2. Set Python path to include the root directory
$env:PYTHONPATH = (Resolve-Path .)
Write-Host "PYTHONPATH set to: $env:PYTHONPATH" -ForegroundColor Cyan

# 3. Set WebSocket protocol
$env:UVICORN_WS = "wsproto"

# 4. Start the server with detailed logging
Write-Host "Starting Uvicorn server..." -ForegroundColor Green
Write-Host "Access the API at: http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "API documentation: http://127.0.0.1:8000/docs" -ForegroundColor Green
Write-Host "\nPress Ctrl+C to stop the server\n" -ForegroundColor Yellow

# Start the server with detailed error output
$ErrorActionPreference = "Stop"
try {
    python -c "
import sys
import os
import uvicorn
from pathlib import Path

print('Python executable:', sys.executable)
print('Python version:', sys.version)
print('Current working directory:', os.getcwd())
print('PYTHONPATH:', os.environ.get('PYTHONPATH', 'Not set'))

# Check if gateway module can be imported
try:
    import gateway
    print('Successfully imported gateway from:', gateway.__file__)
    
    # Check if we can import the app
    from gateway.app import app
    print('Successfully imported app from gateway.app')
    
    # Start the server
    print('\nStarting Uvicorn server...')
    uvicorn.run(
        'gateway.app:app',
        host='127.0.0.1',
        port=8000,
        reload=True,
        ws='wsproto',
        log_level='debug'
    )
except ImportError as e:
    print('Error importing gateway:', e)
    print('\nPython path:')
    for p in sys.path:
        print(f'  - {p}')
    raise
"
} catch {
    Write-Host "\nError starting server:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    
    # Additional diagnostics
    Write-Host "\nChecking Python environment..." -ForegroundColor Yellow
    python -c "import sys; print(f'Python {sys.version} on {sys.platform}')"
    
    Write-Host "\nChecking gateway module..." -ForegroundColor Yellow
    python -c "
import sys
print('Python path:')
for p in sys.path:
    print(f'  - {p}')

try:
    import gateway
    print(f'\nGateway module found at: {gateway.__file__}')
    print('Contents of gateway module:', dir(gateway))
    
    try:
        from gateway.app import app
        print('\nSuccessfully imported app from gateway.app')
        print('App instance:', app)
    except ImportError as e:
        print('\nError importing app from gateway.app:', e)
        print('Contents of gateway directory:', dir(gateway))
        
except ImportError as e:
    print('\nError importing gateway:', e)
    import os
    print('\nCurrent directory contents:')
    for item in os.listdir('.'):
        print(f'  - {item}')
    
    if os.path.exists('gateway'):
        print('\ngateway directory contents:')
        for item in os.listdir('gateway'):
            print(f'  - {item}')
"
}
