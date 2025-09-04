#!/usr/bin/env python3
"""
Setup para Generador de Reportes
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🔧 Configurando Generador de Reportes...")
    
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
    print("📦 Instalando dependencias de report-generator...")
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
    config_content = """# Configuración de Generador de Reportes
SERVICE_NAME=report-generator
SERVICE_PORT=8004
SERVICE_DESCRIPTION=Generador de Reportes
DEBUG=true
LOG_LEVEL=INFO
"""
    
    with open(service_dir / "config" / "default.env", "w") as f:
        f.write(config_content)
    
    print("✅ Generador de Reportes configurado correctamente")
    print(f"🌐 Puerto: 8004")
    print(f"🚀 Para iniciar: python src/main.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
