# ORDEN 1 — Matar procesos y quedarnos solo con .venv

# 1) RAÍZ del repo
Set-Location C:\Users\USUARIO\Agente_IA_OyP_6.0

# 2) Mata procesos Python/uvicorn (si no hay, no pasa nada)
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process uvicorn -ErrorAction SilentlyContinue | Stop-Process -Force

# 3) Activa el venv correcto
deactivate 2>$null
.venv\Scripts\Activate.ps1

# 4) Elimina venvs sobrantes para que el reloader NO se equivoque
if (Test-Path .\.venv_new)   { Remove-Item -Recurse -Force .\.venv_new }
if (Test-Path .\.venv_clean) { Remove-Item -Recurse -Force .\.venv_clean }
if (Test-Path .\gateway\.venv) { Remove-Item -Recurse -Force .\gateway\.venv }

# 5) Comprueba que FastAPI/Pydantic son los esperados **en este shell**
python -c "import sys, fastapi, pydantic, starlette, uvicorn
print('Python:', sys.version.split()[0])
print('fastapi:', fastapi.__version__)
print('pydantic:', pydantic.__version__)
print('starlette:', starlette.__version__)
print('uvicorn:', uvicorn.__version__)"

Write-Host "`nORDEN 1 completada. Presiona cualquier tecla para continuar con ORDEN 2..." -NoNewline
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
