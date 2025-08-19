#!/usr/bin/env python3
"""
🚀 SETUP CONTINUACIÓN - AGENTE IA OYP 6.0
EJECUTAR DENTRO del directorio Agente_IA_OyP_6.0/
Completa la estructura y prepara para instalación de servicios
"""

import os
import sys
import platform
from pathlib import Path

class ContinuationSetup:
    def __init__(self):
        self.current_dir = Path.cwd()
        self.is_windows = platform.system() == "Windows"
        
        # Verificar que estamos en el directorio correcto
        if not self.current_dir.name == "Agente_IA_OyP_6.0":
            print("❌ EJECUTAR DENTRO del directorio Agente_IA_OyP_6.0/")
            print(f"   Directorio actual: {self.current_dir}")
            print("   cd Agente_IA_OyP_6.0")
            sys.exit(1)
        
        # Verificar entorno virtual
        venv_path = self.current_dir / "venv"
        if not venv_path.exists():
            print("❌ Entorno virtual no encontrado")
            print("   Ejecuta primero: python setup_project.py")
            sys.exit(1)
        
        print("🚀 CONTINUACIÓN SETUP - AGENTE IA OYP 6.0")
        print("=" * 50)
        print(f"📁 Directorio: {self.current_dir}")
        print("✅ Entorno virtual detectado")
    
    def create_additional_directories(self):
        """Crear directorios adicionales necesarios"""
        print("📁 Creando estructura adicional...")
        
        additional_dirs = [
            # Configuraciones específicas
            "configs",
            "configs/environments",
            "configs/models",
            "configs/apis",
            
            # Scripts de utilidad
            "scripts",
            "scripts/deployment",
            "scripts/monitoring",
            "scripts/backup",
            
            # Templates
            "templates",
            "templates/reports",
            "templates/emails",
            
            # Estructura detallada por servicio
            "services/ai-engine/src",
            "services/ai-engine/config",
            "services/ai-engine/models",
            "services/ai-engine/tests",
            
            "services/document-processor/src", 
            "services/document-processor/config",
            "services/document-processor/parsers",
            "services/document-processor/tests",
            
            "services/analytics-engine/src",
            "services/analytics-engine/config", 
            "services/analytics-engine/algorithms",
            "services/analytics-engine/tests",
            
            "services/report-generator/src",
            "services/report-generator/config",
            "services/report-generator/templates",
            "services/report-generator/tests",
            
            # Gateway con estructura completa
            "gateway/src",
            "gateway/config",
            "gateway/static", 
            "gateway/templates",
            "gateway/middleware",
            "gateway/routes",
            
            # Docker detallado
            "docker/images",
            "docker/compose",
            "docker/configs",
            
            # Data específica
            "data/exports",
            "data/imports", 
            "data/backups",
            "data/temp"
        ]
        
        for dir_path in additional_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
            # Crear .gitkeep para directorios que pueden estar vacíos
            if any(word in dir_path for word in ["temp", "exports", "imports", "backups"]):
                (Path(dir_path) / ".gitkeep").touch()
        
        print("✅ Estructura adicional creada")
    
    def create_service_setup_scripts(self):
        """Crear scripts de setup individuales para cada servicio"""
        print("🔧 Creando scripts de setup por servicio...")
        
        # Lista de servicios
        services = [
            ("ai-engine", 8001, "Motor de IA"),
            ("document-processor", 8002, "Procesador de Documentos"),
            ("analytics-engine", 8003, "Motor de Analytics"), 
            ("report-generator", 8004, "Generador de Reportes")
        ]
        
        for service_name, port, description in services:
            self.create_individual_service_setup(service_name, port, description)
        
        # Gateway setup
        self.create_gateway_setup()
        
        print("✅ Scripts de setup por servicio creados")
    
    def create_individual_service_setup(self, service_name, port, description):
        """Crear script de setup para un servicio individual"""
        
        service_dir = Path(f"services/{service_name}")
        
        # requirements.txt específico por servicio
        if service_name == "ai-engine":
            requirements = """# AI Engine - Dependencias
transformers==4.35.2
torch==2.1.1
sentence-transformers==2.2.2
openai==1.3.5
anthropic==0.7.7
google-generativeai==0.3.1
spacy==3.7.2
numpy==1.24.3
scikit-learn==1.3.2
accelerate==0.24.1
"""
        elif service_name == "document-processor":
            requirements = """# Document Processor - Dependencias
pypdf2==3.0.1
python-docx==1.1.0
openpyxl==3.1.2
pandas==2.1.3
pillow==10.1.0
pytesseract==0.3.10
pdfplumber==0.9.0
textract==1.6.5
"""
        elif service_name == "analytics-engine":
            requirements = """# Analytics Engine - Dependencias
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.17.0
pandas==2.1.3
numpy==1.24.3
scipy==1.11.4
statsmodels==0.14.0
"""
        else:  # report-generator
            requirements = """# Report Generator - Dependencias
jinja2==3.1.2
weasyprint==60.2
reportlab==4.0.7
markdown==3.5.1
beautifulsoup4==4.12.2
xlsxwriter==3.1.9
python-pptx==0.6.23
"""
        
        # Crear requirements.txt
        with open(service_dir / "requirements.txt", "w", encoding="utf-8") as f:
            f.write(requirements)
        
        # Crear main.py básico
        main_content = f'''"""
{description} - Servicio Principal
Puerto: {port}
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from pathlib import Path

app = FastAPI(
    title="{description}",
    description="Microservicio del Agente IA OyP 6.0",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {{
        "message": "Bienvenido a {description}",
        "service": "{service_name}",
        "version": "1.0.0",
        "status": "active"
    }}

@app.get("/health")
async def health_check():
    return {{
        "status": "healthy",
        "service": "{service_name}",
        "port": {port}
    }}

@app.get("/info")
async def service_info():
    return {{
        "name": "{service_name}",
        "description": "{description}",
        "port": {port},
        "endpoints": ["/", "/health", "/info"]
    }}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port={port},
        reload=True
    )
'''
        
        with open(service_dir / "src" / "main.py", "w", encoding="utf-8") as f:
            f.write(main_content)
        
        # Crear __init__.py
        (service_dir / "src" / "__init__.py").touch()
        
        # Script de setup específico
        setup_script_content = f'''#!/usr/bin/env python3
"""
Setup para {description}
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🔧 Configurando {description}...")
    
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
    print("📦 Instalando dependencias de {service_name}...")
    try:
        subprocess.run([
            str(pip_exe), "install", "-r", 
            str(service_dir / "requirements.txt")
        ], check=True)
        print("✅ Dependencias instaladas")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias: {{e}}")
        return False
    
    # Crear configuración por defecto
    config_content = """# Configuración de {description}
SERVICE_NAME={service_name}
SERVICE_PORT={port}
SERVICE_DESCRIPTION={description}
DEBUG=true
LOG_LEVEL=INFO
"""
    
    with open(service_dir / "config" / "default.env", "w") as f:
        f.write(config_content)
    
    print("✅ {description} configurado correctamente")
    print(f"🌐 Puerto: {port}")
    print(f"🚀 Para iniciar: python src/main.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
'''
        
        with open(service_dir / f"setup_{service_name.replace('-', '_')}.py", "w", encoding="utf-8") as f:
            f.write(setup_script_content)
        
        # Hacer ejecutable en sistemas Unix
        if not self.is_windows:
            os.chmod(service_dir / f"setup_{service_name.replace('-', '_')}.py", 0o755)
    
    def create_gateway_setup(self):
        """Crear setup específico para el gateway"""
        print("🌐 Creando setup del Gateway...")
        
        gateway_dir = Path("gateway")
        
        # requirements.txt para gateway
        gateway_requirements = """# Gateway - Dependencias
fastapi==0.104.1
uvicorn[standard]==0.24.0
jinja2==3.1.2
aiofiles==23.2.1
python-multipart==0.0.6
websockets==12.0
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
"""
        
        with open(gateway_dir / "requirements.txt", "w", encoding="utf-8") as f:
            f.write(gateway_requirements)
        
        # app.py principal del gateway
        app_content = '''"""
API Gateway - Agente IA OyP 6.0
Puerto: 8080
"""

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import httpx
import os
from pathlib import Path

app = FastAPI(
    title="Agente IA OyP 6.0 - Gateway",
    description="API Gateway principal del sistema",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurar archivos estáticos y templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# URLs de microservicios
SERVICES = {
    "ai-engine": "http://localhost:8001",
    "document-processor": "http://localhost:8002", 
    "analytics-engine": "http://localhost:8003",
    "report-generator": "http://localhost:8004"
}

@app.get("/")
async def dashboard(request: Request):
    """Dashboard principal"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "Agente IA OyP 6.0",
        "services": SERVICES
    })

@app.get("/health")
async def health_check():
    """Health check del gateway"""
    return {
        "status": "healthy",
        "service": "gateway",
        "port": 8080
    }

@app.get("/services/status")
async def services_status():
    """Estado de todos los microservicios"""
    status = {}
    
    async with httpx.AsyncClient() as client:
        for service_name, service_url in SERVICES.items():
            try:
                response = await client.get(f"{service_url}/health", timeout=5.0)
                status[service_name] = {
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "url": service_url
                }
            except Exception as e:
                status[service_name] = {
                    "status": "offline",
                    "error": str(e),
                    "url": service_url
                }
    
    return status

# Proxy endpoints para microservicios
@app.api_route("/api/{service_name}/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_to_service(service_name: str, path: str, request: Request):
    """Proxy a microservicios"""
    
    if service_name not in SERVICES:
        return {"error": f"Servicio {service_name} no encontrado"}
    
    service_url = SERVICES[service_name]
    url = f"{service_url}/{path}"
    
    async with httpx.AsyncClient() as client:
        try:
            # Reenviar request al microservicio
            response = await client.request(
                method=request.method,
                url=url,
                params=request.query_params,
                content=await request.body(),
                headers=dict(request.headers)
            )
            return response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
        except Exception as e:
            return {"error": f"Error conectando con {service_name}: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        reload=True
    )
'''
        
        with open(gateway_dir / "app.py", "w", encoding="utf-8") as f:
            f.write(app_content)
        
        # Template HTML básico
        templates_dir = gateway_dir / "templates"
        templates_dir.mkdir(exist_ok=True)
        
        html_template = '''<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-5">
        <h1 class="text-center mb-4">🤖 {{ title }}</h1>
        
        <div class="row">
            <div class="col-md-8 mx-auto">
                <div class="card">
                    <div class="card-header">
                        <h3>🌐 Estado de Servicios</h3>
                    </div>
                    <div class="card-body">
                        <div id="services-status">
                            <p>Cargando estado de servicios...</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mt-4">
            <div class="col-md-8 mx-auto">
                <div class="card">
                    <div class="card-header">
                        <h3>📋 Acciones Rápidas</h3>
                    </div>
                    <div class="card-body">
                        <div class="d-grid gap-2">
                            <button class="btn btn-primary" onclick="checkServices()">🔍 Verificar Servicios</button>
                            <button class="btn btn-success" onclick="openDocs()">📚 Ver Documentación</button>
                            <button class="btn btn-info" onclick="viewLogs()">📊 Ver Logs</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        async function checkServices() {
            const response = await fetch('/services/status');
            const status = await response.json();
            
            let html = '';
            for (const [service, info] of Object.entries(status)) {
                const badgeClass = info.status === 'healthy' ? 'bg-success' : 
                                 info.status === 'offline' ? 'bg-danger' : 'bg-warning';
                html += `
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <span><strong>${service}:</strong> ${info.url}</span>
                        <span class="badge ${badgeClass}">${info.status}</span>
                    </div>
                `;
            }
            
            document.getElementById('services-status').innerHTML = html;
        }
        
        function openDocs() {
            window.open('/docs', '_blank');
        }
        
        function viewLogs() {
            alert('Funcionalidad de logs próximamente');
        }
        
        // Cargar estado inicial
        checkServices();
        
        // Actualizar cada 30 segundos
        setInterval(checkServices, 30000);
    </script>
</body>
</html>'''
        
        with open(templates_dir / "index.html", "w", encoding="utf-8") as f:
            f.write(html_template)
        
        # Setup script para gateway
        gateway_setup = '''#!/usr/bin/env python3
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
'''
        
        with open(gateway_dir / "setup_gateway.py", "w", encoding="utf-8") as f:
            f.write(gateway_setup)
        
        if not self.is_windows:
            os.chmod(gateway_dir / "setup_gateway.py", 0o755)
    
    def create_master_installer(self):
        """Crear script maestro para instalar todos los servicios"""
        print("🔧 Creando instalador maestro...")
        
        installer_content = '''#!/usr/bin/env python3
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
    print(f"\\n🔧 Configurando {service_name}...")
    
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
    
    print("\\n" + "=" * 50)
    if not failed_services:
        print("✅ TODOS LOS SERVICIOS INSTALADOS CORRECTAMENTE")
        print("\\n🚀 Para iniciar el sistema:")
        print("   python gateway/app.py")
        print("\\n🌍 Dashboard disponible en:")
        print("   http://localhost:8080")
    else:
        print(f"⚠️  SERVICIOS CON ERRORES: {', '.join(failed_services)}")
        print("   Revisa los errores arriba y vuelve a ejecutar")
    
    return len(failed_services) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
'''
        
        with open("install_all_services.py", "w", encoding="utf-8") as f:
            f.write(installer_content)
        
        if not self.is_windows:
            os.chmod("install_all_services.py", 0o755)
        
        print("✅ Instalador maestro creado")
    
    def create_quick_start_guide(self):
        """Crear guía de inicio rápido"""
        print("📚 Creando guía de inicio rápido...")
        
        quick_start_content = '''# 🚀 GUÍA DE INICIO RÁPIDO - AGENTE IA OYP 6.0

## ✅ Prerequisitos Completados
- [x] Script maestro ejecutado
- [x] Entorno virtual creado
- [x] Estructura base generada

## 🎯 Próximos Pasos

### 1. Instalar Todos los Servicios (RECOMENDADO)
```bash
python install_all_services.py
```

### 2. O Instalar Servicios Individualmente
```bash
# AI Engine
cd services/ai-engine && python setup_ai_engine.py

# Document Processor
cd ../document-processor && python setup_document_processor.py

# Analytics Engine  
cd ../analytics-engine && python setup_analytics_engine.py

# Report Generator
cd ../report-generator && python setup_report_generator.py

# Gateway (IMPORTANTE - instalar al final)
cd ../../gateway && python setup_gateway.py
```

### 3. Iniciar el Sistema
```bash
# Opción 1: Solo Gateway (para desarrollo)
python gateway/app.py

# Opción 2: Gateway + servicios en paralelo (uso completo)
python manage.py dev
```

### 4. Verificar Funcionamiento
- **Dashboard**: http://localhost:8080
- **API Docs**: http://localhost:8080/docs
- **Status**: http://localhost:8080/services/status

## 🌐 Puertos de Servicios

- **Gateway**: 8080 (Principal)
- **AI Engine**: 8001
- **Document Processor**: 8002
- **Analytics Engine**: 8003  
- **Report Generator**: 8004

## 🔧 Configuración Adicional

### Variables de Entorno (Opcional)
```bash
cp .env.example .env
# Editar .env con tus API keys si tienes
```

### APIs Soportadas (Todas Opcionales)
- OpenAI (GPT-4)
- Anthropic (Claude)
- Google (Gemini)
- **Modelos locales** (funcionan sin API keys)

## 🧪 Testing

```bash
# Verificar setup
python verify_setup.py

# Tests básicos
python manage.py test
```

## 🆘 Solución de Problemas

### Error: "Entorno virtual no encontrado"
```bash
# Volver al directorio padre y ejecutar setup inicial
cd ..
python setup_project.py
```

### Error: "Puerto en uso"
```bash
# Verificar qué está usando el puerto
# Windows:
netstat -ano | findstr :8080

# Linux/macOS:
lsof -i :8080
```

### Error de dependencias
```bash
# Actualizar pip
python -m pip install --upgrade pip

# Reinstalar dependencias
pip install -r requirements.txt
```

## 📞 Soporte

Si encuentras problemas:
1. Verifica que Python 3.8+ esté instalado
2. Asegúrate de estar en el entorno virtual activado
3. Revisa los logs en el directorio `logs/`

---
**¡Listo para usar Agente IA OyP 6.0! 🚀**
'''
        
        with open("QUICK_START.md", "w", encoding="utf-8") as f:
            f.write(quick_start_content)
        
        print("✅ Guía de inicio rápido creada")
    
    def run(self):
        """Ejecutar continuación del setup"""
        try:
            self.create_additional_directories()
            self.create_service_setup_scripts()
            self.create_master_installer()
            self.create_quick_start_guide()
            
            print("\n" + "=" * 50)
            print("✅ CONTINUACIÓN COMPLETADA EXITOSAMENTE")
            print("=" * 50)
            print("\n📋 Próximos pasos RECOMENDADOS:")
            print("   1. python install_all_services.py  # Instalar todos los servicios")
            print("   2. python gateway/app.py          # Iniciar el sistema")
            print("   3. Abrir: http://localhost:8080    # Ver dashboard")
            print("\n📚 Guías disponibles:")
            print("   • QUICK_START.md - Guía completa")
            print("   • README.md - Documentación general")
            print("\n🔧 Comandos útiles:")
            print("   • python manage.py status - Ver servicios")
            print("   • python verify_setup.py - Verificar instalación")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error durante la continuación: {str(e)}")
            return False

def main():
    """Función principal"""
    setup = ContinuationSetup()
    success = setup.run()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)