# Ensure all Python packages have __init__.py files

# Go to the root directory
Set-Location C:\Users\USUARIO\Agente_IA_OyP_6.0

# In gateway/
if (-not (Test-Path .\gateway\__init__.py)) { 
    New-Item -ItemType File .\gateway\__init__.py | Out-Null 
    Write-Host "Created gateway/__init__.py"
}

# In gateway subdirectories
Get-ChildItem .\gateway -Directory -Recurse | ForEach-Object {
    $init = Join-Path $_.FullName "__init__.py"
    if (-not (Test-Path $init)) { 
        New-Item -ItemType File $init | Out-Null
        Write-Host "Created $init"
    }
}

# In services/gateway/ if it exists
if (Test-Path .\services\gateway) {
    if (-not (Test-Path .\services\gateway\__init__.py)) { 
        New-Item -ItemType File .\services\gateway\__init__.py | Out-Null
        Write-Host "Created services/gateway/__init__.py"
    }
    
    # In services/gateway subdirectories
    Get-ChildItem .\services\gateway -Directory -Recurse | ForEach-Object {
        $init = Join-Path $_.FullName "__init__.py"
        if (-not (Test-Path $init)) { 
            New-Item -ItemType File $init | Out-Null
            Write-Host "Created $init"
        }
    }
}

Write-Host "Package structure verification complete!" -ForegroundColor Green
