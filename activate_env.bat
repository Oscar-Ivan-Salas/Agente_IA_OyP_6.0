@echo off
echo 🐍 Activando entorno virtual de Agente IA OyP 6.0...
cd /d "C:\Users\USUARIO\Agente_IA_OyP_6.0"
call venv\Scripts\activate.bat

echo ✅ Entorno virtual activado
echo 📂 Directorio del proyecto: %CD%
echo 🐍 Python: %CD%\venv\Scripts\python.exe

echo.
echo 🚀 Comandos disponibles:
echo   python manage.py dev       - Iniciar desarrollo
echo   python manage.py status    - Ver status de servicios
echo   python manage.py test      - Ejecutar tests
echo.
