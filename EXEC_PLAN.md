# Plan de Ejecución: MVP Orquestación

Checklist ordenado para la implementación del proyecto "Vibe Code". Cada ítem corresponde a un prompt del plan maestro.

- [x] **Fase 0: Preparación y Análisis** (COMPLETADO)
  - [x] **Prompt 1: Auditoría integral previa.** (DONE)
    - _Resultado:_ `AUDIT.md` generado.
  - [x] **Prompt 2: Gap analysis + plan de ejecución.** (DONE)
    - _Entregables:_ `GAP_ANALYSIS.md`, `EXEC_PLAN.md` generados.
  - [x] **Prompt 3: Política de cambios y Go/No-Go.** (DONE)
    - _Entregables:_ `CHANGE_POLICY.md`, `RISK_REPORT.md` generados y revisados.
    - _Riesgo crítico identificado y mitigado:_ Se resolvió el problema de compatibilidad de pydantic.

- [ ] **Fase 1: Estructura y Entorno Base** (EN PROGRESO)
  - [x] **Prompt 4: Sincronización del plan y bitácora.** (COMPLETADO)
    - _Entregables:_ `CHANGELOG.md` inicial creado con los cambios realizados hasta el momento.
    - _Archivos actualizados:_ `EXEC_PLAN.md`, `RISK_REPORT.md`
  - [x] **Prompt 5: Entorno y estructura base.** (COMPLETADO)
    - _Entregables:_ 
      - `.env.example` verificado y actualizado con configuraciones necesarias.
      - Estructura de directorios `data/` y `db/` verificada y completada.
      - Directorios creados: `data/twins/`, `data/outputs/`, `data/vectors/`, `db/migrations/`.
  - [x] **Prompt 8: DB SQLite y migraciones mínimas.** (COMPLETADO)
    - _Entregables:_ 
      - Modelos de base de datos creados en `gateway/models/`
      - Configuración de base de datos en `gateway/database.py`
      - Configuración de migraciones con Alembic
      - Script de inicialización en `scripts/iniciar_bd.py`
      - Base de datos SQLite inicializada

- [ ] **Fase 2: Servicios Core** (PRÓXIMA)
  - [ ] **Prompt 6: Gateway + WebSocket + orquestador de jobs.** (TODO)
    - _Entregables:_ `main.py`, `jobs.py`, `db.py`, `progress.py` en `services/gateway/`.
  - [ ] **Prompt 9: Servicio Document Processor.** (TODO)
    - _Entregables:_ `main.py` en `services/docproc/`.
  - [ ] **Prompt 10: Servicio AI Engine.** (TODO)
    - _Entregables:_ `main.py` en `services/ai/`.
  - [ ] **Prompt 11: Servicio Analytics.** (TODO)
    - _Entregables:_ `main.py` en `services/analytics/`.
  - [ ] **Prompt 12: Servicio Report Generator.** (TODO)
    - _Entregables:_ `main.py` en `services/reports/`.
  - [ ] **Prompt 13: Servicio Gantt.** (TODO)
    - _Entregables:_ Módulo de generación de Gantt en PNG.

- [ ] **Fase 3: Lógica de Negocio y Orquestación**
  - [ ] **Prompt 7: Contratos OpenAPI del Gateway.** (TODO)
    - _Entregables:_ `openapi/gateway.yaml`.
  - [ ] **Prompt 14: CRUD Proyectos/Tareas/Diarios/Riesgos.** (TODO)
    - _Entregables:_ Implementación de endpoints CRUD en el Gateway.
  - [ ] **Prompt 15: Cálculo de avance y KPIs.** (TODO)
    - _Entregables:_ Implementación de `progress.py` y endpoint de KPIs.
  - [ ] **Prompt 16: Orquestación “reporte semanal” y “final”.** (TODO)
    - _Entregables:_ Endpoint `/api/agent/command` con la lógica de orquestación.
  - [ ] **Prompt 18: Auto-indexado Documento → AI.** (TODO)
    - _Entregables:_ Hook para invocar `/ai/embed` tras la extracción.

- [ ] **Fase 4: Contenido, Seguridad y Finalización**
  - [ ] **Prompt 17: Plantillas base.** (TODO)
    - _Entregables:_ Archivos `.docx` en `data/templates/`.
  - [ ] **Prompt 19: Seguridad, CORS y límites.** (TODO)
    - _Entregables:_ Middleware de CORS y API Key en el Gateway.
  - [ ] **Prompt 20: Tests, seeds, scripts y guía de demo.** (TODO)
    - _Entregables:_ `tests/`, `scripts/run_all.*`, `scripts/seed_demo.py`, `DEMO.md`.
