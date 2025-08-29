@echo off

REM Activar el entorno virtual
call .venv\Scripts\activate.bat

REM Instalar dependencias
echo Instalando dependencias...
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install pydantic-settings

REM Ejecutar las pruebas
echo Ejecutando pruebas...
python -m pytest -v

pause
