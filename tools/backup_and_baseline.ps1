# Script de backup y línea base
$ErrorActionPreference = "Stop"
$root = (Get-Location).Path
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"

# Crear backup ZIP
Write-Host "Creando backup del código..."
New-Item -ItemType Directory -Force -Path ".\backups" | Out-Null
Compress-Archive -Path * -DestinationPath ".\backups\full_$stamp.zip" -Force

# Inicializar repositorio Git si no existe
if (-not (Test-Path ".git")) {
    Write-Host "Inicializando repositorio Git..."
    git init | Out-Null
}

# Crear .gitignore si no existe
if (-not (Test-Path ".gitignore")) {
    @"
.venv/
venv/
backups/
__pycache__/
node_modules/
*.pyc
.DS_Store
"@ | Set-Content -Encoding UTF8 .gitignore
}

# Hacer commit inicial
try {
    git add -A
    git commit -m "baseline: snapshot antes de auditoría ($stamp)" | Out-Null
    Write-Host "✅ Backup: .\backups\full_$stamp.zip creado"
    Write-Host "✅ Commit baseline creado con éxito"
} catch {
    Write-Host "⚠  No se pudo crear el commit. Asegúrate de que Git esté configurado correctamente."
    Write-Host "   El backup ZIP se creó correctamente en: .\backups\full_$stamp.zip"
}
