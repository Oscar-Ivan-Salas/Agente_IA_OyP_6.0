#!/usr/bin/env python3
"""
Setup para Motor de IA
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🔧 Configurando Motor de IA...")
    
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
    print("📦 Instalando dependencias de ai-engine...")
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
    config_content = """# Configuración de Motor de IA
SERVICE_NAME=ai-engine
SERVICE_PORT=8001
SERVICE_DESCRIPTION=Motor de IA
DEBUG=true
LOG_LEVEL=INFO
"""
    
    with open(service_dir / "config" / "default.env", "w") as f:
        f.write(config_content)
    
    print("✅ Motor de IA configurado correctamente")
    print(f"🌐 Puerto: 8001")
    print(f"🚀 Para iniciar: python src/main.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
