# Gap Analysis Report

Este documento describe la diferencia entre el estado actual del repositorio (ver `AUDIT.md`) y los requisitos del proyecto "Vibe Code" descritos en los 20 prompts.

## Resumen de Gaps

El proyecto tiene una estructura de carpetas base, pero la mayoría de las funcionalidades clave son inexistentes y deben ser construidas desde cero.

- **Servicios Principales:**
  - **Gateway (Prompt 6):** El `gateway/app.py` existente es un placeholder. La lógica de orquestación de jobs, la cola en SQLite y el WebSocket no existen.
  - **Document Processor (Prompt 9):** El servicio en `services/document-processor/` está vacío. No hay lógica para ingesta de PDF/DOCX ni para la creación de "digital twins".
  - **AI Engine (Prompt 10):** El servicio en `services/ai-engine/` está vacío. No hay implementación para carga de modelos, indexación con ChromaDB, ni endpoints de embedding/search/NER.
  - **Analytics Engine (Prompt 11):** El servicio en `services/analytics-engine/` está vacío. No hay lógica para análisis estadístico de CSVs.
  - **Report Generator (Prompt 12):** El servicio en `services/report-generator/` está vacío. No hay implementación de `docxtpl` para renderizar reportes.
  - **Gantt Service (Prompt 13):** Completamente ausente.

- **Base de Datos (Prompt 8):**
  - No existe el archivo de base de datos `db/oyp.sqlite`.
  - No se encontraron scripts de migración (e.g., con Alembic).
  - Las tablas `projects`, `tasks`, `daily_logs`, etc., no existen.

- **API y Contratos (Prompt 7):**
  - El `AUDIT.md` muestra solo 6 endpoints genéricos en el gateway.
  - El 95% de los endpoints definidos en `openapi/gateway.yaml` (CRUD de proyectos, tareas, reportes, AI, etc.) no están implementados.
  - El archivo `openapi/gateway.yaml` no existe.

- **Lógica de Negocio:**
  - **CRUDs (Prompt 14):** La lógica para gestionar proyectos, tareas, diarios y riesgos no existe.
  - **Cálculo de KPIs (Prompt 15):** El módulo `progress.py` y la lógica para calcular avances y KPIs son inexistentes.
  - **Orquestación (Prompt 16):** Los flujos de "reporte semanal" y "reporte final" no están implementados.

- **Contenido y Scripts:**
  - **Plantillas (Prompt 17):** El directorio `data/templates/` está vacío. Faltan todas las plantillas base (`.docx`).
  - **Scripts de Ejecución (Prompt 20):** Los scripts `run_all.sh`/`.bat` para levantar el ecosistema de servicios no existen.
  - **Seeds (Prompt 20):** No hay script `seed_demo.py` para poblar la base de datos.
  - **Guía de Demo (Prompt 20):** El archivo `DEMO.md` no existe.

- **Riesgos y Calidad:**
  - **Pruebas Rotas:** Como se detalla en `AUDIT.md`, la suite de pruebas no funciona, lo cual es un gap crítico de calidad y estabilidad. Se necesitan tests funcionales (Prompt 20).
