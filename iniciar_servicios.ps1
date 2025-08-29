# Script para iniciar los servicios uno por uno

# Función para mostrar un menú
function Show-Menu {
    param (
        [string]$Title = 'Gestor de Servicios - Agente IA OYP 6.0'
    )
    Clear-Host
    Write-Host "===== $Title ====="
    Write-Host "1. Iniciar Gateway Principal (puerto 8000)"
    Write-Host "2. Iniciar Motor de IA (puerto 8001)"
    Write-Host "3. Iniciar Procesador de Documentos (puerto 8002)"
    Write-Host "4. Iniciar Motor de Análisis (puerto 8003)"
    Write-Host "5. Iniciar Generador de Reportes (puerto 8004)"
    Write-Host "6. Iniciar TODOS los servicios"
    Write-Host "7. Ver estado de los servicios"
    Write-Host "Q. Salir"
}

# Función para iniciar un servicio
function Start-ServiceWithLog {
    param (
        [string]$serviceName,
        [string]$displayName
    )
    
    Write-Host "`nIniciando $displayName..." -ForegroundColor Cyan
    python service_manager.py start --service $serviceName
    
    # Esperar y verificar estado
    Start-Sleep -Seconds 2
    $status = python service_manager.py status --service $serviceName 2>&1
    
    if ($status -match "running") {
        Write-Host "✅ $displayName iniciado correctamente" -ForegroundColor Green
    } else {
        Write-Host "❌ Error al iniciar $displayName" -ForegroundColor Red
        Write-Host "Intentando ver logs..."
        python service_manager.py logs --service $serviceName --lines 20
    }
    
    Write-Host "`nPresiona cualquier tecla para continuar..."
    $null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
}

# Menú principal
do {
    Show-Menu
    $input = Read-Host "`nSelecciona una opción"
    
    switch ($input) {
        '1' { Start-ServiceWithLog -serviceName "gateway" -displayName "Gateway Principal" }
        '2' { Start-ServiceWithLog -serviceName "ai-engine" -displayName "Motor de IA" }
        '3' { Start-ServiceWithLog -serviceName "document-processor" -displayName "Procesador de Documentos" }
        '4' { Start-ServiceWithLog -serviceName "analytics-engine" -displayName "Motor de Análisis" }
        '5' { Start-ServiceWithLog -serviceName "report-generator" -displayName "Generador de Reportes" }
        '6' {
            # Iniciar todos los servicios en orden
            $services = @(
                @{Name="gateway"; DisplayName="Gateway Principal"},
                @{Name="ai-engine"; DisplayName="Motor de IA"},
                @{Name="document-processor"; DisplayName="Procesador de Documentos"},
                @{Name="analytics-engine"; DisplayName="Motor de Análisis"},
                @{Name="report-generator"; DisplayName="Generador de Reportes"}
            )
            
            foreach ($service in $services) {
                Start-ServiceWithLog -serviceName $service.Name -displayName $service.DisplayName
            }
        }
        '7' { 
            Clear-Host
            Write-Host "=== Estado de los servicios ===`n" -ForegroundColor Cyan
            python service_manager.py status
            Write-Host "`nPresiona cualquier tecla para continuar..."
            $null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
        }
        'q' { Write-Host "Saliendo..." -ForegroundColor Yellow }
        default { Write-Host "Opción no válida. Intenta de nuevo." -ForegroundColor Red }
    }
} until ($input -eq 'q')
