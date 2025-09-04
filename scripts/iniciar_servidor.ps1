# Script para iniciar el servidor FastAPI

# Configurar el entorno
$env:PYTHONPATH = "$PWD"

# Instalar dependencias si es necesario
Write-Host "Verificando dependencias..."
pip install -r requirements.txt

# Iniciar el servidor Uvicorn
Write-Host "Iniciando servidor FastAPI..."
python -m uvicorn gateway.main:app --host 0.0.0.0 --port 8000 --reload

# Si hay un error, intentar con wsproto
if ($LASTEXITCODE -ne 0) {
    Write-Host "Intentando con wsproto..."
    python -m uvicorn gateway.main:app --host 0.0.0.0 --port 8000 --reload --ws wsproto
}
