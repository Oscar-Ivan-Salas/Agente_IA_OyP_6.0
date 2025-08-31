#!/usr/bin/env python3
"""
AGENTE IA OYP 6.0 - SCRIPT DE GESTIÓN
Comandos principales para manejar el proyecto
"""

import os
import sys
import subprocess
import argparse
import signal
import time
from pathlib import Path

class ProjectManager:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.venv_path = self.project_root / "venv"
        
        if os.name == 'nt':  # Windows
            self.python_exe = self.venv_path / "Scripts" / "python.exe"
            self.pip_exe = self.venv_path / "Scripts" / "pip.exe"
        else:  # Linux/macOS
            self.python_exe = self.venv_path / "bin" / "python"
            self.pip_exe = self.venv_path / "bin" / "pip"
    
    def check_venv(self):
        """Verificar que el entorno virtual existe"""
        if not self.python_exe.exists():
            print("[ERROR] Entorno virtual no encontrado")
            print("   Ejecuta primero: python setup_project.py")
            return False
        return True
    
    def dev(self):
        """Iniciar en modo desarrollo"""
        if not self.check_venv():
            return
        
        print("Iniciando en modo desarrollo...")
        print(f"Dashboard disponible en: http://localhost:8080")
        
        gateway_path = self.project_root / "gateway"
        if gateway_path.exists():
            os.chdir(gateway_path)
            # NOTE: This will block until the gateway process is terminated.
            subprocess.run([str(self.python_exe), "app.py"])
        else:
            print("[ERROR] Gateway no encontrado. Instala primero los servicios.")
    
    def status(self):
        """Ver status de servicios"""
        print("Verificando servicios...")
        
        services = [
            ("Gateway", "http://localhost:8080/health"),
            ("AI Engine", "http://localhost:8001/health"),
            ("Document Processor", "http://localhost:8002/health"),
            ("Analytics Engine", "http://localhost:8003/health"),
            ("Report Generator", "http://localhost:8004/health")
        ]
        
        try:
            import requests
            for name, url in services:
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        print(f"[OK] {name}: ONLINE")
                    else:
                        print(f"[WARN] {name}: ERROR ({response.status_code})")
                except Exception:
                    print(f"[FAIL] {name}: OFFLINE")
        except ImportError:
            print("[WARN] requests no instalado. No se puede verificar servicios.")
    
    def logs(self):
        """Ver logs"""
        logs_dir = self.project_root / "logs"
        if logs_dir.exists():
            log_files = list(logs_dir.glob("*.log"))
            if log_files:
                latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
                print(f"Mostrando logs de: {latest_log.name}")
                try:
                    with open(latest_log, 'r', encoding='utf-8') as f:
                        print(f.read())
                except Exception as e:
                    print(f"Error leyendo logs: {e}")
            else:
                print("No hay archivos de log")
        else:
            print("Directorio de logs no existe")
    
    def clean(self):
        """Limpiar archivos temporales"""
        print("Limpiando archivos temporales...")
        
        for root, dirs, files in os.walk(self.project_root):
            for d in dirs:
                if d == "__pycache__":
                    cache_dir = Path(root) / d
                    import shutil
                    shutil.rmtree(cache_dir)
                    print(f"Eliminado: {cache_dir}")
        
        for pyc_file in self.project_root.rglob("*.pyc"):
            pyc_file.unlink()
            print(f"Eliminado: {pyc_file}")
        
        print("Limpieza completada")
    
    def test(self):
        """Ejecutar tests"""
        if not self.check_venv():
            return
        
        print("Ejecutando tests...")
        subprocess.run([
            str(self.python_exe), "-m", "pytest", 
            "tests/", "-v", "--tb=short"
        ])

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Gestor del proyecto Agente IA OyP 6.0")
    parser.add_argument('command', choices=['dev', 'status', 'logs', 'clean', 'test'], 
                       help='Comando a ejecutar')
    
    args = parser.parse_args()
    
    manager = ProjectManager()
    
    if args.command == 'dev':
        manager.dev()
    elif args.command == 'status':
        manager.status()
    elif args.command == 'logs':
        manager.logs()
    elif args.command == 'clean':
        manager.clean()
    elif args.command == 'test':
        manager.test()

if __name__ == "__main__":
    main()