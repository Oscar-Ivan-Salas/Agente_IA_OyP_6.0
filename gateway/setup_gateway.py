#!/usr/bin/env python3
"""
Setup para API Gateway
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🌐 Configurando API Gateway...")
    
    gateway_dir = Path(__file__).parent
    project_root = gateway_dir.parent
    
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
    
    # Instalar dependencias del gateway
    print("📦 Instalando dependencias del gateway...")
    try:
        subprocess.run([
            str(pip_exe), "install", "-r", 
            str(gateway_dir / "requirements.txt")
        ], check=True)
        print("✅ Dependencias instaladas")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias: {e}")
        return False
    
    # Crear directorio static si no existe
    static_dir = gateway_dir / "static"
    static_dir.mkdir(exist_ok=True)
    
    print("✅ API Gateway configurado correctamente")
    print("🌐 Puerto: 8080")
    print("🚀 Para iniciar: python app.py")
    print("🌍 Dashboard: http://localhost:8080")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
