@echo off
echo Ejecutando ORDEN 2 - Arreglar colisi¢n del paquete websockets
echo =============================================================

:: Ir a la ra¡z del repositorio
cd /d "C:\Users\USUARIO\Agente_IA_OyP_6.0"

:: Asegurar que gateway sea un paquete Python
echo.
echo 1. Verificando paquete gateway...
if not exist gateway\__init__.py (
    echo   - Creando archivo __init__.py en gateway
    echo. > gateway\__init__.py
) else (
    echo   - El archivo __init__.py ya existe en gateway
)

:: Renombrar carpeta websockets si existe
echo.
echo 2. Verificando carpeta websockets...
if exist gateway\websockets (
    echo   - Renombrando gateway\websockets a gateway\ws_app
    ren "gateway\websockets" "ws_app"
) else (
    echo   - No se encontr¢ la carpeta gateway\websockets
)

echo.
echo 3. Buscando y reemplazando imports en archivos Python...
setlocal enabledelayedexpansion
set "count=0"

for /r gateway\ %%f in (*.py) do (
    set "file=%%f"
    set "modified=0"
    
    :: Crear archivo temporal
    set "tempfile=%%~dpnf.tmp"
    
    (
    for /f "usebackq delims=" %%a in ("%%f") do (
        set "line=%%a"
        
        :: Reemplazar los patrones de import
        echo !line! | findstr /i "from gateway import websockets" >nul
        if not errorlevel 1 (
            set "line=!line:from gateway import websockets=from gateway import ws_app!"
            set /a modified+=1
        )
        
        echo !line! | findstr /i "from .websockets import" >nul
        if not errorlevel 1 (
            set "line=!line:from .websockets import=from .ws_app import!"
            set /a modified+=1
        )
        
        echo !line! | findstr /i "from gateway.websockets import" >nul
        if not errorlevel 1 (
            set "line=!line:from gateway.websockets import=from gateway.ws_app import!"
            set /a modified+=1
        )
        
        echo(!line!
    )
    ) > "!tempfile!"
    
    :: Si se hicieron cambios, reemplazar el archivo original
    if !modified! gtr 0 (
        set /a count+=1
        echo   - Modificados !modified! imports en: %%f
        move /y "!tempfile!" "%%f" >nul
    ) else (
        del "!tempfile!" >nul 2>&1
    )
)

:: Limpiar archivos compilados
echo.
echo 4. Limpiando archivos compilados...
for /r . %%d in (__pycache__) do if exist "%%d" (
    echo   - Eliminando directorio: %%d
    rmdir /s /q "%%d" 2>nul
)
del /s /q *.pyc 2>nul

echo.
echo ORDEN 2 completada. Se modificaron %count% archivos.
echo Presiona una tecla para continuar...
pause >nul
