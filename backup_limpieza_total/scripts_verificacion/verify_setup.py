#!/usr/bin/env python3
"""
Script de verificación del setup inicial
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Verificar versión de Python"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} no compatible")
        return False

def check_venv():
    """Verificar entorno virtual"""
    venv_path = Path("venv")
    
    if os.name == 'nt':  # Windows
        python_exe = venv_path / "Scripts" / "python.exe"
    else:  # Linux/macOS
        python_exe = venv_path / "bin" / "python"
    
    if python_exe.exists():
        print("✅ Entorno virtual existe")
        return True
    else:
        print("❌ Entorno virtual no encontrado")
        return False

def check_directories():
    """Verificar directorios principales"""
    required_dirs = [
        "gateway", "services", "data", "docs", "tests", "logs"
    ]
    
    all_exist = True
    for dirname in required_dirs:
        if Path(dirname).exists():
            print(f"✅ Directorio {dirname}")
        else:
            print(f"❌ Directorio {dirname} faltante")
            all_exist = False
    
    return all_exist

def check_files():
    """Verificar archivos principales"""
    required_files = [
        "requirements.txt", "manage.py", ".gitignore", "README.md"
    ]
    
    all_exist = True
    for filename in required_files:
        if Path(filename).exists():
            print(f"✅ Archivo {filename}")
        else:
            print(f"❌ Archivo {filename} faltante")
            all_exist = False
    
    return all_exist

def main():
    """Función principal"""
    print("🧪 VERIFICACIÓN DEL SETUP")
    print("=" * 30)
    
    checks = [
        ("Versión de Python", check_python_version),
        ("Entorno Virtual", check_venv),
        ("Directorios", check_directories),
        ("Archivos", check_files)
    ]
    
    all_passed = True
    for name, check_func in checks:
        print(f"\n🔍 Verificando {name}...")
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 30)
    if all_passed:
        print("✅ VERIFICACIÓN EXITOSA")
        print("\n📋 Próximos pasos:")
        print("   1. Activar entorno virtual")
        print("   2. Instalar servicios individuales")
        print("   3. python manage.py dev")
        return True
    else:
        print("❌ VERIFICACIÓN FALLIDA")
        print("   Ejecuta de nuevo: python setup_project.py")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
