#!/usr/bin/env python3
"""
INSTALADOR MAESTRO - AGENTE IA OYP 6.0
Instala todos los servicios en orden
"""

import os
import sys
import subprocess
from pathlib import Path

def run_service_setup(service_path, service_name):
    """Ejecutar setup de un servicio"""
    print(f"\n🔧 Configurando {service_name}...")
    
    try:
        os.chdir(service_path)
        result = subprocess.run([sys.executable, f"setup_{service_name.replace('-', '_')}.py"], 
                              check=True, capture_output=True, text=True)
        print(f"✅ {service_name} configurado")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error configurando {service_name}")
        print(f"   {e.stdout}")
        print(f"   {e.stderr}")
        return False
    finally:
        os.chdir(Path(__file__).parent)

def main():
    """Función principal"""
    print("🚀 INSTALADOR MAESTRO - AGENTE IA OYP 6.0")
    print("=" * 50)
    
    # Servicios a instalar
    services = [
        ("services/ai-engine", "ai-engine"),
        ("services/document-processor", "document-processor"),
        ("services/analytics-engine", "analytics-engine"),
        ("services/report-generator", "report-generator"),
        ("gateway", "gateway")
    ]
    
    failed_services = []
    
    for service_path, service_name in services:
        if not run_service_setup(service_path, service_name):
            failed_services.append(service_name)
    
    print("\n" + "=" * 50)
    if not failed_services:
        print("✅ TODOS LOS SERVICIOS INSTALADOS CORRECTAMENTE")
        print("\n🚀 Para iniciar el sistema:")
        print("   python gateway/app.py")
        print("\n🌍 Dashboard disponible en:")
        print("   http://localhost:8080")
    else:
        print(f"⚠️  SERVICIOS CON ERRORES: {', '.join(failed_services)}")
        print("   Revisa los errores arriba y vuelve a ejecutar")
    
    return len(failed_services) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
