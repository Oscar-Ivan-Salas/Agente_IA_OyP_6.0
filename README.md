# 🤖 Agente IA OyP 6.0

## 🎯 Descripción

Plataforma de Inteligencia Documental con IA híbrida (local + cloud) construida con arquitectura de microservicios.

## 🚀 Instalación Rápida

### 1. Script Maestro (EJECUTAR PRIMERO)
```bash
python setup_project.py
```

### 2. Activar Entorno Virtual
```bash
# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
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
