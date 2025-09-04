#!/usr/bin/env python3
"""
📡 Gestor de Servicios - Agente IA OYP 6.0
=======================================
Script para gestionar todos los servicios del sistema.

Uso:
    python service_manager.py [comando] [opciones]

Comandos disponibles:
    start     Iniciar todos los servicios
    stop      Detener todos los servicios
    restart   Reiniciar todos los servicios
    status    Mostrar estado de los servicios
    logs      Mostrar logs de un servicio

Opciones:
    --service <nombre>  Especificar un servicio específico
    --help             Mostrar este mensaje de ayuda
"""

import os
import sys
import subprocess
import time
import signal
import psutil
from typing import Dict, List, Optional
import platform
import argparse

# Configuración de servicios
SERVICES = {
    "gateway": {
        "name": "Gateway Principal",
        "path": "gateway",
        "command": "uvicorn app:app --host 0.0.0.0 --port 8000 --reload",
        "port": 8000,
        "process": None
    },
    "ai-engine": {
        "name": "Motor de IA",
        "path": "services/ai-engine",
        "command": "uvicorn main:app --host 0.0.0.0 --port 8001 --reload",
        "port": 8001,
        "process": None
    },
    "document-processor": {
        "name": "Procesador de Documentos",
        "path": "services/document-processor",
        "command": "uvicorn main:app --host 0.0.0.0 --port 8002 --reload",
        "port": 8002,
        "process": None
    },
    "analytics-engine": {
        "name": "Motor de Análisis",
        "path": "services/analytics-engine",
        "command": "uvicorn main:app --host 0.0.0.0 --port 8003 --reload",
        "port": 8003,
        "process": None
    },
    "report-generator": {
        "name": "Generador de Reportes",
        "path": "services/report-generator",
        "command": "uvicorn main:app --host 0.0.0.0 --port 8004 --reload",
        "port": 8004,
        "process": None
    }
}

def is_windows() -> bool:
    """Verificar si el sistema operativo es Windows"""
    return platform.system() == "Windows"

def get_process_on_port(port: int) -> Optional[psutil.Process]:
    """Obtener proceso que está usando un puerto específico"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            for conns in proc.connections(kind='inet'):
                if conns.laddr.port == port:
                    return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return None

def stop_service(service_name: str) -> bool:
    """Detener un servicio específico"""
    if service_name not in SERVICES:
        print(f"❌ Error: Servicio '{service_name}' no encontrado")
        return False
    
    service = SERVICES[service_name]
    port = service["port"]
    process = get_process_on_port(port)
    
    if process:
        try:
            print(f"🛑 Deteniendo {service['name']} (PID: {process.pid})...")
            if is_windows():
                subprocess.run(["taskkill", "/F", "/PID", str(process.pid)], 
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except psutil.TimeoutExpired:
                    process.kill()
            print(f"✅ {service['name']} detenido correctamente")
            return True
        except Exception as e:
            print(f"❌ Error al detener {service['name']}: {str(e)}")
            return False
    else:
        print(f"ℹ️  {service['name']} no está en ejecución")
        return True

def start_service(service_name: str) -> bool:
    """Iniciar un servicio específico"""
    if service_name not in SERVICES:
        print(f"❌ Error: Servicio '{service_name}' no encontrado")
        return False
    
    service = SERVICES[service_name]
    port = service["port"]
    
    # Verificar si el puerto ya está en uso
    if get_process_on_port(port):
        print(f"⚠️  {service['name']} ya está en ejecución en el puerto {port}")
        return True
    
    try:
        print(f"🚀 Iniciando {service['name']}...")
        
        # Configurar el entorno
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.abspath(".")
        
        # Iniciar el proceso
        if is_windows():
            process = subprocess.Popen(
                service["command"],
                cwd=service["path"],
                shell=True,
                env=env,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:
            process = subprocess.Popen(
                service["command"],
                cwd=service["path"],
                shell=True,
                env=env,
                start_new_session=True
            )
        
        # Esperar un momento para verificar si el servicio se inicia correctamente
        time.sleep(2)
        
        if process.poll() is not None:
            print(f"❌ Error al iniciar {service['name']}")
            return False
        
        print(f"✅ {service['name']} iniciado correctamente (PID: {process.pid})")
        service["process"] = process
        return True
    except Exception as e:
        print(f"❌ Error al iniciar {service['name']}: {str(e)}")
        return False

def restart_service(service_name: str) -> bool:
    """Reiniciar un servicio específico"""
    if stop_service(service_name):
        time.sleep(1)  # Esperar un momento antes de reiniciar
        return start_service(service_name)
    return False

def check_service_status(service_name: str) -> Dict[str, str]:
    """Verificar el estado de un servicio"""
    if service_name not in SERVICES:
        return {"status": "error", "message": f"Servicio '{service_name}' no encontrado"}
    
    service = SERVICES[service_name]
    port = service["port"]
    process = get_process_on_port(port)
    
    if process:
        return {
            "status": "running",
            "name": service["name"],
            "pid": process.pid,
            "port": port,
            "uptime": time.time() - process.create_time()
        }
    else:
        return {
            "status": "stopped",
            "name": service["name"],
            "port": port
        }

def show_status(service_name: Optional[str] = None):
    """Mostrar el estado de los servicios"""
    print("\n📊 Estado de los servicios:\n" + "="*50)
    
    services_to_check = [service_name] if service_name else SERVICES.keys()
    
    for name in services_to_check:
        status = check_service_status(name)
        if status["status"] == "running":
            uptime = int(status["uptime"])
            hours, remainder = divmod(uptime, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"✅ {status['name']} (PID: {status['pid']}, Puerto: {status['port']})")
            print(f"   ⏱️  Tiempo activo: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        else:
            print(f"❌ {status['name']} (Detenido, Puerto: {status['port']})")
    
    print("\n" + "="*50)

def show_logs(service_name: str, lines: int = 50):
    """Mostrar logs de un servicio"""
    if service_name not in SERVICES:
        print(f"❌ Error: Servicio '{service_name}' no encontrado")
        return
    
    service = SERVICES[service_name]
    log_file = os.path.join(service["path"], "logs", "app.log")
    
    if not os.path.exists(log_file):
        print(f"ℹ️  No se encontró archivo de log para {service['name']}")
        return
    
    try:
        print(f"📜 Últimas {lines} líneas del log de {service['name']}:")
        print("="*70)
        with open(log_file, 'r', encoding='utf-8') as f:
            lines_to_show = f.readlines()[-lines:]
            print(''.join(lines_to_show))
        print("="*70)
    except Exception as e:
        print(f"❌ Error al leer el archivo de log: {str(e)}")

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Gestor de servicios de Agente IA OYP 6.0')
    subparsers = parser.add_subparsers(dest='command', help='Comandos disponibles')
    
    # Comando start
    start_parser = subparsers.add_parser('start', help='Iniciar servicios')
    start_parser.add_argument('--service', help='Nombre del servicio a iniciar')
    
    # Comando stop
    stop_parser = subparsers.add_parser('stop', help='Detener servicios')
    stop_parser.add_argument('--service', help='Nombre del servicio a detener')
    
    # Comando restart
    restart_parser = subparsers.add_parser('restart', help='Reiniciar servicios')
    restart_parser.add_argument('--service', help='Nombre del servicio a reiniciar')
    
    # Comando status
    status_parser = subparsers.add_parser('status', help='Mostrar estado de los servicios')
    status_parser.add_argument('--service', help='Nombre del servicio a consultar')
    
    # Comando logs
    logs_parser = subparsers.add_parser('logs', help='Mostrar logs de un servicio')
    logs_parser.add_argument('--service', required=True, help='Nombre del servicio')
    logs_parser.add_argument('--lines', type=int, default=50, help='Número de líneas a mostrar')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'start':
            if args.service:
                start_service(args.service)
            else:
                for service in SERVICES:
                    start_service(service)
                    time.sleep(1)  # Pequeña pausa entre servicios
        
        elif args.command == 'stop':
            if args.service:
                stop_service(args.service)
            else:
                # Detener en orden inverso al de inicio
                for service in reversed(list(SERVICES.keys())):
                    stop_service(service)
                    time.sleep(1)  # Pequeña pausa entre servicios
        
        elif args.command == 'restart':
            if args.service:
                restart_service(args.service)
            else:
                for service in SERVICES:
                    restart_service(service)
                    time.sleep(1)  # Pequeña pausa entre servicios
        
        elif args.command == 'status':
            show_status(args.service)
        
        elif args.command == 'logs':
            if args.service:
                show_logs(args.service, args.lines)
            else:
                print("❌ Debe especificar un servicio con --service")
        
    except KeyboardInterrupt:
        print("\nOperación cancelada por el usuario")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
