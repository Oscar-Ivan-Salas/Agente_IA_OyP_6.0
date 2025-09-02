@echo off
echo Ejecutando ORDEN 3 - Verificando importaciones de websockets
echo =========================================================

:: Ir a la raíz del repositorio
cd /d "C:\Users\USUARIO\Agente_IA_OyP_6.0"

echo.
echo Verificando rutas de importación...
python -c "import websockets, sys, importlib; print('1. websockets(lib) ->', websockets.__file__); import gateway; print('2. gateway pkg ->', gateway.__path__); from gateway.main import app; print('3. gateway.main:app -> OK')"

echo.
echo Análisis de rutas:
echo - La ruta de websockets debe terminar en \.venv\Lib\site-packages\websockets\...
echo - Si ves gateway\websockets o similar, hay que corregir más imports.

echo.
echo Presiona una tecla para continuar...
pause >nul
