@echo off
echo Ejecutando ORDEN 4 - Iniciando servidor Uvicorn
echo ===========================================

:: Ir a la raíz del repositorio
cd /d "C:\Users\USUARIO\Agente_IA_OyP_6.0"

echo.
echo Configurando variables de entorno...
set UVICORN_WS=wsproto

echo.
echo Iniciando servidor Uvicorn...
echo.
echo URL: http://127.0.0.1:8000
echo.

:: Ejecutar Uvicorn
python -m uvicorn gateway.main:app --host 127.0.0.1 --port 8000 --reload --ws wsproto

:: Si hay un error, mostrar solo las primeras 20 líneas
if errorlevel 1 (
    echo.
    echo ===========================================
    echo ERROR al iniciar el servidor. Últimas 20 líneas:
    echo ===========================================
    for /f "tokens=1-20" %%i in ('python -m uvicorn gateway.main:app --host 127.0.0.1 --port 8000 --reload --ws wsproto 2^>^&1 ^| findstr /n "^" ^| findstr /b "[0-9][0-9]*:" ^| findstr /v "[0-9][0-9]*:$" ^| findstr /v "^[0-9][0-9]*: *$" ^| findstr /v "^[0-9][0-9]*: *[0-9][0-9]*$" ^| findstr /v "^[0-9][0-9]*: *[0-9][0-9]* *$" ^| findstr /v "^[0-9][0-9]*: *[0-9][0-9]* *[0-9][0-9]*$" ^| findstr /v "^[0-9][0-9]*: *[0-9][0-9]* *[0-9][0-9]* *$" ^| findstr /v "^[0-9][0-9]*: *[0-9][0-9]* *[0-9][0-9]* *[0-9][0-9]*$" ^| findstr /v "^[0-9][0-9]*: *[0-9][0-9]* *[0-9][0-9]* *[0-9][0-9]* *$" ^| findstr /v "^[0-9][0-9]*: *[0-9][0-9]* *[0-9][0-9]* *[0-9][0-9]* *[0-9][0-9]*$"') do (
        set "line=%%i"
        setlocal enabledelayedexpansion
        set "line=!line:*:=!"
        echo !line!
        endlocal
    )
    
    echo.
    echo ===========================================
    echo Verificando versiones en el entorno virtual:
    echo ===========================================
    python -c "import sys, pydantic, fastapi; print('Python:', sys.executable); print('pydantic:', pydantic.__version__, pydantic.__file__); print('fastapi:', fastapi.__version__, fastapi.__file__)"
    
    echo.
    echo Si ves versiones de Pydantic v1 o rutas incorrectas, por favor:
    echo 1. Cierra todas las terminales de Python/CMD
    echo 2. Vuelve a abrir una nueva terminal
    echo 3. Ejecuta ejecutar_orden1.bat de nuevo
    echo 4. Luego ejecuta este archivo otra vez
)

echo.
pause
