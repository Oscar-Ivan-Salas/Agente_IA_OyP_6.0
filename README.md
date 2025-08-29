# 🤖 Agente IA OyP 6.0

## 🎯 Descripción

Plataforma de Inteligencia Documental con IA híbrida (local + cloud) construida con arquitectura de microservicios. Este sistema integra múltiples servicios especializados que funcionan de manera independiente pero coordinada a través de un Gateway central.

## 🏗️ Arquitectura del Sistema

El sistema sigue una arquitectura de microservicios con los siguientes componentes principales:

1. **Gateway API** (Puerto 8080) - Punto de entrada principal para todas las peticiones
2. **AI Engine** (Puerto 8001) - Motor de IA para procesamiento de lenguaje natural
3. **Document Processor** (Puerto 8002) - Procesamiento de documentos (PDF, DOCX, etc.)
4. **Analytics Engine** (Puerto 8003) - Análisis de datos y generación de estadísticas
5. **Report Generator** (Puerto 8004) - Generación de reportes en múltiples formatos
6. **Chat AI Service** (Puerto 8005) - Servicio de chat en tiempo real

## 🚀 Instalación Rápida

### 1. Requisitos Previos
- Python 3.9+
- pip
- Git

### 2. Clonar el Repositorio
```bash
git clone https://github.com/tu-usuario/Agente_IA_OyP_6.0.git
cd Agente_IA_OyP_6.0
```

### 3. Configuración del Entorno
1. Copiar el archivo de configuración de ejemplo:
   ```bash
   cp .env.example .env
   ```
2. Configurar las variables de entorno en el archivo `.env` según sea necesario

### 4. Crear y Activar Entorno Virtual
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 5. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 6. Iniciar los Servicios
```bash
# Iniciar el Gateway
python gateway/app.py

# En otra terminal, iniciar los microservicios
python services/ai_engine/app.py
python services/document_processor/app.py
# ... y así para cada servicio
```

## 📁 Estructura del Proyecto

```
Agente_IA_OyP_6.0/
├── gateway/               # API Gateway principal
├── services/              # Microservicios
│   ├── ai_engine/        # Motor de IA
│   ├── document_processor/ # Procesador de documentos
│   ├── analytics_engine/  # Motor de análisis
│   ├── report_generator/  # Generador de reportes
│   └── chat_ai/          # Servicio de chat
├── data/                 # Datos del sistema
├── tests/                # Pruebas automatizadas
├── docs/                 # Documentación
└── scripts/              # Scripts de utilidad
```

## 🔧 Variables de Entorno

Copia `.env.example` a `.env` y configura las siguientes variables:

- `DEBUG`: Modo depuración (True/False)
- `DATABASE_URL`: URL de conexión a la base de datos
- `OPENAI_API_KEY`: Clave API de OpenAI (opcional)
- `ANTHROPIC_API_KEY`: Clave API de Anthropic (opcional)
- `REDIS_URL`: URL de Redis para caché (opcional)

## 🧪 Ejecutando Pruebas

```bash
# Ejecutar todas las pruebas
pytest tests/

# Ejecutar pruebas específicas
pytest tests/unit/
pytest tests/integration/
```

## 🐳 Ejecución con Docker

```bash
# Construir las imágenes
docker-compose build

# Iniciar los contenedores
docker-compose up -d
```

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.
```

### 3. Próximos Pasos
- Configurar variables de entorno (.env)
- Instalar servicios individuales
- Iniciar desarrollo

## 🛠️ Comandos Básicos

```bash
# Iniciar desarrollo
python manage.py dev

# Ver estado de servicios
python manage.py status

# Ver logs
python manage.py logs

# Ejecutar tests
python manage.py test

# Limpiar archivos temporales
python manage.py clean
```

## 📁 Estructura del Proyecto

```
Agente_IA_OyP_6.0/
├── 📁 gateway/              # API Gateway principal
├── 📁 services/             # Microservicios
│   ├── ai-engine/           # Motor de IA
│   ├── document-processor/  # Procesador de documentos
│   ├── analytics-engine/    # Motor de análisis
│   └── report-generator/    # Generador de reportes
├── 📁 data/                 # Datos y archivos
├── 📁 docs/                 # Documentación
├── 📁 tests/                # Tests
├── 📁 docker/               # Configuración Docker
└── 📁 logs/                 # Logs del sistema
```

## 🌐 URLs de Acceso

- **Dashboard Principal**: http://localhost:8080
- **API Docs**: http://localhost:8080/docs
- **Health Checks**: http://localhost:8080/health

---

**Desarrollado con ❤️ para la comunidad de IA en español**
# 📋 DOCUMENTACIÓN ARQUITECTURA - Agente IA OyP 6.0

## 🎯 **VISIÓN GENERAL**

El **Agente IA OyP 6.0** es una plataforma de inteligencia documental basada en **arquitectura de microservicios** donde:

- **1 Dashboard Central** (interfaz gráfica unificada)
- **5 Microservicios Independientes** (cada uno con su lógica propia)
- **Todos los servicios se ven en una sola página** pero funcionan independientemente

## 🏗️ **ARQUITECTURA TÉCNICA**

### **Capa de Presentación (Frontend)**
```
📱 Dashboard Tabler 1.4 (Puerto 8080)
├── 📊 Sección: Analytics SPSS
├── 💬 Sección: Chat IA  
├── 📄 Sección: Documentos
├── 📋 Sección: Reportes
├── 🔧 Sección: Servicios
└── 🎛️ Sección: Dashboard Principal
```

### **Capa de Gateway (Coordinación)**
```
🌐 API Gateway FastAPI (Puerto 8080)
├── 🔄 Proxy a microservicios
├── 🛡️ Autenticación y CORS
├── 📡 WebSocket para tiempo real
├── 📁 Gestión de archivos estáticos
└── 🔗 Coordinación entre servicios
```

### **Capa de Microservicios (Backend)**
```
🔄 Microservicios Independientes:
├── 🤖 AI Engine (Puerto 8001)
│   ├── Modelos locales (Llama, Mistral)
│   ├── APIs cloud (GPT-4, Claude, Gemini)
│   └── Procesamiento de lenguaje natural
│
├── 📄 Document Processor (Puerto 8002)  
│   ├── OCR de imágenes
│   ├── Extracción PDF/DOCX
│   └── Clasificación automática
│
├── 📊 Analytics Engine (Puerto 8003)
│   ├── Estadísticas descriptivas
│   ├── Análisis de correlaciones
│   ├── Machine Learning básico
│   └── Visualizaciones (Plotly)
│
├── 📋 Report Generator (Puerto 8004)
│   ├── Templates dinámicos
│   ├── PDFs profesionales
│   └── Exports múltiples formatos
│
└── 💬 Chat AI Service (Puerto 8005)
    ├── WebSocket tiempo real
    ├── Contexto conversacional
    └── Integración con AI Engine
```

### **Capa de Datos**
```
💾 Sistema de Datos:
├── 🗄️ SQLite (datos principales)
├── 📁 File System (uploads, cache)
├── 🔄 Redis (sesiones, cache)
└── 💾 Memoria (estado WebSocket)
```

## 📁 **ESTRUCTURA DE DIRECTORIOS**

```
Agente_IA_OyP_6.0/
├── 📂 gateway/                    # API Gateway Principal
│   ├── 📂 templates/              # Dashboard HTML
│   │   └── index.html            # Dashboard Tabler completo
│   ├── 📂 static/                # Assets estáticos
│   │   ├── css/                  # Estilos personalizados
│   │   └── js/                   # JavaScript del dashboard
│   ├── 📂 src/                   # Código del gateway
│   │   ├── routers/              # Rutas FastAPI
│   │   ├── middleware/           # Middleware personalizado
│   │   └── utils/                # Utilidades
│   ├── app.py                    # Aplicación principal
│   └── requirements.txt          # Dependencias gateway
│
├── 📂 services/                   # Microservicios
│   ├── 📂 ai-engine/             # Motor de IA
│   ├── 📂 document-processor/    # Procesador documentos
│   ├── 📂 analytics-engine/      # Motor análisis
│   ├── 📂 report-generator/      # Generador reportes
│   └── 📂 chat-ai/              # Chat inteligente
│
├── 📂 data/                      # Datos del sistema
│   ├── uploads/                  # Archivos subidos
│   ├── processed/                # Archivos procesados
│   ├── models/                   # Modelos ML
│   └── cache/                    # Cache del sistema
│
├── 📂 logs/                      # Logs del sistema
├── 📂 docs/                      # Documentación
├── 📂 tests/                     # Tests automatizados
└── 📂 docker/                    # Configuración Docker
```

## 🔌 **COMUNICACIÓN ENTRE SERVICIOS**

### **1. Dashboard → Gateway**
```javascript
// Frontend (Dashboard) → Gateway
fetch('/api/services/analytics/analyze', {
    method: 'POST',
    body: formData
})
```

### **2. Gateway → Microservicio**
```python
# Gateway → Microservicio específico
async with httpx.AsyncClient() as client:
    response = await client.post(
        f"http://localhost:8003/analyze", 
        json=data
    )
```

### **3. WebSocket Tiempo Real**
```javascript
// Comunicación bidireccional
const ws = new WebSocket('ws://localhost:8080/ws');
ws.onmessage = (event) => {
    updateDashboard(JSON.parse(event.data));
};
```

## 🎨 **TECNOLOGÍAS POR CAPA**

### **Frontend (Dashboard)**
- **Framework UI**: Tabler 1.4 (Bootstrap-based)
- **JavaScript**: Vanilla JS + WebSocket API
- **Visualizaciones**: Plotly.js integrado
- **Estilos**: CSS3 + Tabler components

### **Gateway (Coordinación)**
- **Framework**: FastAPI (Python 3.11+)
- **WebSocket**: FastAPI WebSocket support
- **Proxy**: httpx async client
- **Templates**: Jinja2
- **Middleware**: CORS, Auth, Logging

### **Microservicios**
- **AI Engine**: Python + Transformers + OpenAI API
- **Document Processor**: Python + PyPDF2 + python-docx
- **Analytics**: Python + Pandas + Scikit-learn + Plotly
- **Report Generator**: Python + ReportLab + Jinja2
- **Chat AI**: Python + WebSocket + IA contextual

### **Base de Datos**
- **Principal**: SQLite (desarrollo) / PostgreSQL (producción)
- **Cache**: Redis (opcional)
- **Files**: Sistema de archivos local

## 🔄 **FLUJO DE DATOS TÍPICO**

### **Análisis de Documentos**
```
📱 Usuario sube archivo en Dashboard
    ↓
🌐 Gateway recibe archivo y lo almacena
    ↓
📄 Document Processor extrae contenido
    ↓
🤖 AI Engine analiza contenido
    ↓
📊 Analytics Engine genera estadísticas
    ↓
📋 Report Generator crea reporte
    ↓
📱 Dashboard muestra resultados
```

### **Chat Interactivo**
```
📱 Usuario escribe mensaje en chat
    ↓
🌐 Gateway recibe vía WebSocket
    ↓
💬 Chat AI Service procesa contexto
    ↓
🤖 AI Engine genera respuesta
    ↓
📱 Dashboard muestra respuesta en tiempo real
```

## ⚙️ **CONFIGURACIÓN DEL SISTEMA**

### **Puertos de Servicios**
```
Gateway Principal:     8080
AI Engine:            8001
Document Processor:   8002
Analytics Engine:     8003
Report Generator:     8004
Chat AI Service:      8005
```

### **Variables de Entorno**
```bash
# .env
DEBUG=true
DATABASE_URL=sqlite:///./agente_ia.db
OPENAI_API_KEY=optional
ANTHROPIC_API_KEY=optional
```

### **Dependencias Principales**
```
FastAPI (gateway)
Pandas + Plotly (analytics)
Transformers (IA)
ReportLab (reportes)
WebSocket (tiempo real)
```

## 🚀 **FLUJO DE DESARROLLO**

### **1. Estructura Actual**
- ✅ Gateway funcionando (puerto 8080)
- ✅ Dashboard HTML base creado
- ❓ Microservicios en desarrollo
- ❓ Integración dashboard-servicios pendiente

### **2. Próximos Pasos Sugeridos**
1. **Completar Dashboard** con todas las secciones
2. **Implementar microservicios** uno por uno
3. **Integrar WebSocket** para tiempo real
4. **Testing** y optimización

### **3. Principios de Desarrollo**
- **Separation of Concerns**: Cada servicio tiene una responsabilidad
- **Single Page Application**: Todo se ve en una página
- **Microservices Pattern**: Servicios independientes y escalables
- **API-First**: Todas las funcionalidades vía API
- **Real-time**: WebSocket para actualizaciones inmediatas

## 🔧 **COMANDOS DE GESTIÓN**

### **Desarrollo**
```bash
# Activar entorno
source venv/bin/activate

# Iniciar gateway
cd gateway && python app.py

# Iniciar servicio específico
cd services/ai-engine && python app.py
```

### **Verificación**
```bash
# Health check
curl http://localhost:8080/health

# Estado servicios
curl http://localhost:8080/api/services/status
```

---

## 📝 **CONCLUSIÓN**

Esta arquitectura permite:

✅ **Dashboard único** que coordina todos los servicios
✅ **Microservicios independientes** cada uno con su lógica
✅ **Escalabilidad** - cada servicio puede crecer independientemente  
✅ **Mantenibilidad** - cambios en un servicio no afectan otros
✅ **Tiempo real** - WebSocket para actualizaciones inmediatas
✅ **Flexibilidad** - fácil agregar nuevos servicios

La clave está en el **Gateway** que actúa como **orquestador** mostrando todo en una interfaz unificada mientras mantiene la **independencia** de cada microservicio.

Agente_IA_OyP_6.0/
├── 📁 configs/                     # Configuraciones
│   ├── 📁 apis/                   # Configuraciones de APIs externas
│   ├── 📁 environments/           # Configuraciones por entorno
│   └── 📁 models/                 # Configuraciones de modelos
│
├── 📁 data/                       # Datos de la aplicación
│   ├── 📁 backups/                # Copias de seguridad
│   ├── 📁 cache/                  # Datos en caché
│   ├── 📁 exports/                # Datos exportados
│   ├── 📁 imports/                # Datos para importar
│   ├── 📁 models/                 # Modelos de IA
│   ├── 📁 processed/              # Datos procesados
│   ├── 📁 temp/                   # Archivos temporales
│   └── 📁 uploads/                # Archivos subidos
│
├── 📁 docker/                     # Configuración de Docker
│   ├── 📁 compose/                # Archivos docker-compose
│   ├── 📁 configs/                # Configuraciones para contenedores
│   ├── 📁 images/                 # Imágenes personalizadas
│   └── 📁 volumes/                # Volúmenes de datos
│
├── 📁 docs/                       # Documentación
│   ├── 📁 api/                    # Documentación de la API
│   ├── 📁 deployment/             # Guías de despliegue
│   └── 📁 guides/                 # Guías de usuario
│
├── 📁 gateway/                    # Gateway principal
│   ├── 📁 config/                 # Configuración del gateway
│   ├── 📁 middleware/             # Middlewares
│   ├── 📁 routes/                 # Rutas de la API
│   ├── 📁 src/                    # Código fuente
│   ├── 📁 static/                 # Archivos estáticos
│   │   ├── 📁 css/                # Hojas de estilo
│   │   ├── 📁 img/                # Imágenes
│   │   └── 📁 js/                 # JavaScript del frontend
│   ├── 📁 templates/              # Plantillas HTML
│   └── 📁 uploads/                # Archivos subidos temporalmente
│
├── 📁 logs/                       # Archivos de registro
│
├── 📁 scripts/                    # Scripts de utilidad
│   ├── 📁 backup/                 # Scripts de respaldo
│   ├── 📁 deployment/             # Scripts de despliegue
│   └── 📁 monitoring/             # Scripts de monitoreo
│
├── 📁 services/                   # Microservicios
│   ├── 📁 ai-engine/              # Motor de IA
│   │   ├── 📁 cache/              # Caché del motor
│   │   ├── 📁 config/             # Configuración
│   │   ├── 📁 data/               # Datos del motor
│   │   ├── 📁 logs/               # Registros
│   │   └── 📁 models/             # Modelos de IA
│   │
│   ├── 📁 analytics-engine/       # Motor de análisis
│   ├── 📁 document-processor/     # Procesador de documentos
│   └── 📁 report-generator/       # Generador de reportes
│
├── 📁 templates/                  # Plantillas globales
│   ├── 📁 emails/                 # Plantillas de correo
│   └── 📁 reports/                # Plantillas de reportes
│
├── 📁 tests/                      # Pruebas
│   ├── 📁 e2e/                    # Pruebas de extremo a extremo
│   ├── 📁 integration/            # Pruebas de integración
│   └── 📁 unit/                   # Pruebas unitarias
│
├── 📄 .env.example               # Variables de entorno de ejemplo
├── 📄 manage.py                  # Punto de entrada de la aplicación
├── 📄 pytest.ini                 # Configuración de pytest
├── 📄 README.md                  # Documentación principal
├── 📄 requirements.txt           # Dependencias de producción
└── 📄 requirements-dev.txt       # Dependencias de desarrollo