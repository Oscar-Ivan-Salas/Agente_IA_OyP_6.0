# Script para iniciar el servidor de desarrollo
$env:PYTHONPATH = $PWD
$env:UVICORN_HOST = "0.0.0.0"
$env:UVICORN_PORT = 8000
$env:UVICORN_RELOAD = "--reload"

Write-Host "🚀 Iniciando servidor de desarrollo..." -ForegroundColor Green
Write-Host "📡 Accede a: http://localhost:$env:UVICORN_PORT" -ForegroundColor Cyan

# Ejecutar el servidor
uvicorn gateway.main:app --host $env:UVICORN_HOST --port $env:UVICORN_PORT $env:UVICORN_RELOAD
