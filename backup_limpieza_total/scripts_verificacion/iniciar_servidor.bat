@echo off
echo Iniciando servidor...
echo ====================

:: 1. Ir al directorio del proyecto
cd /d "C:\Users\USUARIO\Agente_IA_OyP_6.0"

:: 2. Detener procesos de Python/uvicorn si están en ejecución
taskkill /F /IM python.exe /T >nul 2>&1
taskkill /F /IM uvicorn.exe /T >nul 2>&1

:: 3. Activar el entorno virtual
call .venv\Scripts\activate.bat

:: 4. Configurar variables de entorno
set UVICORN_WS=wsproto

:: 5. Iniciar el servidor Uvicorn
echo.
echo Iniciando servidor Uvicorn...
echo.
echo URL: http://127.0.0.1:8000
echo.

python -m uvicorn gateway.main:app --host 127.0.0.1 --port 8000 --reload --ws wsproto

:: 6. Si hay un error, mostrar información de depuración
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ===========================================
    echo ERROR al iniciar el servidor
    echo ===========================================
    echo.
    echo Verificando versiones de paquetes...
    python -c "import sys, pydantic, fastapi; print('Python:', sys.executable); print('pydantic:', pydantic.__version__); print('fastapi:', fastapi.__version__)"
    
    echo.
    echo Si ves versiones de Pydantic v1 o rutas incorrectas, por favor:
    echo 1. Cierra todas las terminales de Python/CMD
    echo 2. Vuelve a abrir una nueva terminal
    echo 3. Ejecuta 'ejecutar_orden1.bat' primero
    echo 4. Luego ejecuta este archivo otra vez
)

pause
