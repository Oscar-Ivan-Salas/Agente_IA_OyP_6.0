# Script de auditoría para Agente IA OYP 6.0

Write-Host "=== AUDITORÍA DEL PROYECTO ===`n"

# COMANDO 1 - Ver contenido de archivos principales
Write-Host "=== CONTENIDO DE gateway/app.py ==="
if (Test-Path "gateway\app.py") {
    Get-Content -Path "gateway\app.py" -TotalCount 20
} else {
    Write-Host "El archivo gateway/app.py no existe"
}
Write-Host ""

Write-Host "=== CONTENIDO DE gateway/main.py ==="
if (Test-Path "gateway\main.py") {
    Get-Content -Path "gateway\main.py" -TotalCount 20
} else {
    Write-Host "El archivo gateway/main.py no existe"
}
Write-Host ""

# COMANDO 2 - Verificar servicios
Write-Host "=== ARCHIVOS EN services/ai-engine/ ==="
if (Test-Path "services\ai-engine") {
    Get-ChildItem -Path "services\ai-engine" -Force
} else {
    Write-Host "El directorio services/ai-engine/ no existe"
}
Write-Host ""

Write-Host "=== ARCHIVOS EN services/analytics-engine/ ==="
if (Test-Path "services\analytics-engine") {
    Get-ChildItem -Path "services\analytics-engine" -Force
} else {
    Write-Host "El directorio services/analytics-engine/ no existe"
}
Write-Host ""

Write-Host "=== ARCHIVOS EN services/document-processor/ ==="
if (Test-Path "services\document-processor") {
    Get-ChildItem -Path "services\document-processor" -Force
} else {
    Write-Host "El directorio services/document-processor/ no existe"
}
Write-Host ""

# COMANDO 3 - Buscar archivos con código real (no vacíos)
Write-Host "=== ARCHIVOS .PY CON CONTENIDO (primeros 20) ==="
Get-ChildItem -Path . -Recurse -Filter "*.py" -File | Where-Object { $_.Length -gt 1KB } | Select-Object -First 20 FullName, Length, LastWriteTime | Format-Table -AutoSize
