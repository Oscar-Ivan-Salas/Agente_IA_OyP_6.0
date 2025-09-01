#!/usr/bin/env python3
"""
Setup para Procesador de Documentos
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🔧 Configurando Procesador de Documentos...")
    
    service_dir = Path(__file__).parent
    project_root = service_dir.parent.parent
    
    # Verificar entorno virtual
    if os.name == 'nt':  # Windows
        pip_exe = project_root / "venv" / "Scripts" / "pip.exe"
        python_exe = project_root / "venv" / "Scripts" / "python.exe"
    else:  # Linux/macOS
        pip_exe = project_root / "venv" / "bin" / "pip"
        python_exe = project_root / "venv" / "bin" / "python"
    
    if not pip_exe.exists():
        print("❌ Entorno virtual no encontrado")
        print("   Ejecuta primero el setup principal")
        return False
    
    # Instalar dependencias específicas
    # Instalar dependencias específicas
    print("⬆️ Actualizando pip y setuptools...")
    try:
        subprocess.run([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([str(python_exe), "-m", "pip", "install", "--upgrade", "setuptools"], check=True)
        print("✅ pip y setuptools actualizados")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error actualizando pip o setuptools: {e}")
        return False
    print("📦 Instalando numpy por separado...")
    try:
        subprocess.run([str(pip_exe), "install", "numpy==1.24.3"], check=True)
        print("✅ numpy instalado")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando numpy: {e}")
        return False
    print("📦 Instalando dependencias de document-processor...")
    try:
        subprocess.run([
            str(pip_exe), "install", "-r", 
            str(service_dir / "requirements.txt")
        ], check=True)
        print("✅ Dependencias instaladas")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias: {e}")
        return False
    
    # Crear configuración por defecto
    config_content = """# Configuración de Procesador de Documentos
SERVICE_NAME=document-processor
SERVICE_PORT=8002
SERVICE_DESCRIPTION=Procesador de Documentos
DEBUG=true
LOG_LEVEL=INFO
"""
    
    with open(service_dir / "config" / "default.env", "w") as f:
        f.write(config_content)
    
    print("✅ Procesador de Documentos configurado correctamente")
    print(f"🌐 Puerto: 8002")
    print(f"🚀 Para iniciar: python src/main.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
