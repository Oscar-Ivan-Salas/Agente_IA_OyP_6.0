# Setup and run script for Agente IA OYP 6.0

# Go to project root
Set-Location C:\Users\USUARIO\Agente_IA_OyP_6.0

# PROMPT A - Install required dependencies
Write-Host "`n=== INSTALANDO DEPENDENCIAS ===" -ForegroundColor Cyan
. \.venv\Scripts\Activate.ps1
pip install "httpx==0.27.2" wsproto "uvicorn[standard]==0.35.0" python-multipart

# PROMPT C - Verify critical imports
Write-Host "`n=== VERIFICANDO IMPORTS CRÍTICOS ===" -ForegroundColor Cyan
python -c "
mods = ['fastapi','starlette','pydantic','uvicorn','httpx','wsproto']
for m in mods:
    try:
        __import__(m)
        print(f'{m}: OK')
    except Exception as e:
        print(f'{m}: FALLO -> {e}')
"

# PROMPT D - Check for websockets collision
Write-Host "`n=== VERIFICANDO CONFLICTOS DE WEBSOCKETS ===" -ForegroundColor Cyan
$websocketsPath = ".\gateway\websockets"
if (Test-Path $websocketsPath) {
    Write-Host "ADVERTENCIA: Se encontró una carpeta 'websockets' local que podría causar conflictos" -ForegroundColor Yellow
    Write-Host "Por favor, renómbrela a 'ws_app' y actualice los imports correspondientes."
    
    Write-Host "`nBuscando imports que podrían estar causando conflictos..."
    Select-String -Path .\**\*.py -Pattern "from\s+gateway\.websockets|from\s+\.websockets" -AllMatches
    
    Write-Host "`nPor favor, actualice los imports a 'ws_app' y luego ejecute este script nuevamente."
    exit 1
} else {
    Write-Host "No se encontró carpeta 'websockets' conflictiva." -ForegroundColor Green
}

# PROMPT B - Start the server
Write-Host "`n=== INICIANDO SERVIDOR ===" -ForegroundColor Green
$env:PYTHONPATH = (Resolve-Path .)
$env:UVICORN_WS = "wsproto"

# Check which app file to use
$appPath = ".\gateway\app.py"
$appModule = "gateway.app:app"

if (-not (Test-Path $appPath)) {
    $appPath = ".\services\gateway\main.py"
    $appModule = "services.gateway.main:app"
    
    if (-not (Test-Path $appPath)) {
        Write-Host "No se pudo encontrar el archivo principal de la aplicación." -ForegroundColor Red
        Write-Host "Busqué en:"
        Write-Host "  - .\gateway\app.py"
        Write-Host "  - .\services\gateway\main.py"
        exit 1
    }
}

Write-Host "Iniciando servidor con: $appModule" -ForegroundColor Green
Write-Host "URL: http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "Documentación: http://127.0.0.1:8000/docs" -ForegroundColor Green
Write-Host "`nPresione Ctrl+C para detener el servidor`n" -ForegroundColor Yellow

# Start the server
python -m uvicorn $appModule --host 127.0.0.1 --port 8000 --reload --ws wsproto
