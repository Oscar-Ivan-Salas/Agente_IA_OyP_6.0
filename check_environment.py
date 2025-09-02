#!/usr/bin/env python3
"""
Script para verificar el entorno de Python.
"""
import sys
import os
import platform
import pkg_resources

def print_section(title):
    print("\n" + "="*80)
    print(f" {title}".ljust(80, '='))
    print("="*80)

print_section("INFORMACIÓN DEL SISTEMA")
print(f"Sistema operativo: {platform.system()} {platform.release()}")
print(f"Versión de Python: {sys.version}")
print(f"Ejecutable de Python: {sys.executable}")
print(f"Directorio de trabajo: {os.getcwd()}")

print_section("VARIABLES DE ENTORNO")
for key, value in sorted(os.environ.items()):
    if 'python' in key.lower() or 'path' in key.lower():
        print(f"{key}: {value}")

print_section("PAQUETES INSTALADOS")
installed_packages = sorted([f"{pkg.key}=={pkg.version}" for pkg in pkg_resources.working_set])
for pkg in installed_packages:
    print(pkg)

print_section("ESTRUCTURA DE DIRECTORIOS")
project_root = os.path.dirname(os.path.abspath(__file__))
print(f"Directorio del proyecto: {project_root}")

def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files[:10]:  # Mostrar solo los primeros 10 archivos por directorio
            print(f"{subindent}{f}")
        if len(files) > 10:
            print(f"{subindent}... y {len(files) - 10} archivos más")

list_files(project_root)
