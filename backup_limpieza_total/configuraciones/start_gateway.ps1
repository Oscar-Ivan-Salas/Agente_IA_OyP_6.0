# Start Gateway with proper configuration

# Go to the root directory
Set-Location C:\Users\USUARIO\Agente_IA_OyP_6.0

# 1. Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
. \.venv\Scripts\Activate.ps1

# 2. Set Python path to include the root directory
$env:PYTHONPATH = (Resolve-Path .)
Write-Host "Set PYTHONPATH to: $env:PYTHONPATH" -ForegroundColor Cyan

# 3. Set WebSocket protocol
$env:UVICORN_WS = "wsproto"

# 4. Check if we need to update imports
Write-Host "Checking for relative imports..." -ForegroundColor Cyan
$imports_updated = python -c "
import os
import re

# Check if any Python files in gateway/ use relative imports
gateway_dir = os.path.join(os.path.dirname(__file__), 'gateway')
has_relative_imports = False

for root, _, files in os.walk(gateway_dir):
    for file in files:
        if file.endswith('.py'):
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if re.search(r'from \s*\.\w+\s+import', content) or \
                   re.search(r'from \s*\.\.\w+\s+import', content):
                    has_relative_imports = True
                    break
    if has_relative_imports:
        break

print('true' if has_relative_imports else 'false')
"

if ($imports_updated -eq 'true') {
    Write-Host "Updating imports to use absolute paths..." -ForegroundColor Yellow
    python update_imports.py
}

# 5. Start the server
Write-Host "\nStarting Uvicorn server..." -ForegroundColor Green
Write-Host "Access the API at: http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "API documentation: http://127.0.0.1:8000/docs" -ForegroundColor Green
Write-Host "\nPress Ctrl+C to stop the server\n" -ForegroundColor Yellow

# Start the server
python -m uvicorn gateway.main:app --host 127.0.0.1 --port 8000 --reload --ws wsproto
