# 🚀 GUÍA DE INICIO RÁPIDO - AGENTE IA OYP 6.0

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
