@echo off
echo Ejecutando ORDEN 1 - Limpieza de entorno
echo =======================================

:: 1) Ir a la raíz del repositorio
cd /d "C:\Users\USUARIO\Agente_IA_OyP_6.0"

echo.
echo 1. Deteniendo procesos de Python y Uvicorn...
taskkill /F /IM python.exe /T >nul 2>&1
taskkill /F /IM uvicorn.exe /T >nul 2>&1

echo 2. Activando entorno virtual .venv...
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else (
    echo ERROR: No se encontró el entorno virtual .venv
    pause
    exit /b 1
)

echo 3. Eliminando entornos virtuales sobrantes...
if exist .venv_new (
    echo   - Eliminando .venv_new...
    rmdir /s /q .venv_new
)
if exist .venv_clean (
    echo   - Eliminando .venv_clean...
    rmdir /s /q .venv_clean
)
if exist gateway\.venv (
    echo   - Eliminando gateway\.venv...
    rmdir /s /q gateway\.venv
)

echo.
echo 4. Verificando versiones de paquetes...
python -c "import sys, fastapi, pydantic, starlette, uvicorn; print(f'Python: {sys.version.split()[0]}'); print(f'fastapi: {fastapi.__version__}'); print(f'pydantic: {pydantic.__version__}'); print(f'starlette: {starlette.__version__}'); print(f'uvicorn: {uvicorn.__version__}')"

echo.
echo ORDEN 1 completada. Presiona una tecla para continuar...
pause >nul
