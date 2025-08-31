# Política de Cambios del Proyecto

Este documento establece las reglas y directrices que deben seguirse al realizar cambios en el código base para garantizar la estabilidad, consistencia y retrocompatibilidad del sistema.

## Reglas Fundamentales

1.  **No Tocar la UI:** Queda estrictamente prohibido modificar cualquier archivo relacionado con la interfaz de usuario (UI) del dashboard. Esto incluye archivos HTML, CSS, JavaScript del frontend y cualquier otro elemento visible por el usuario final.

2.  **No Borrar ni Renombrar:** No se deben borrar ni renombrar archivos o carpetas existentes que ya estén en producción o en la rama principal. La estabilidad del sistema depende de la consistencia de su estructura.

3.  **Compatibilidad Retroactiva:** Todos los cambios en la API y en la estructura de datos deben ser retrocompatibles. Si se necesita un cambio disruptivo, se debe crear una nueva versión del endpoint o del recurso (e.g., `/api/v2/...`).

4.  **Commits Atómicos:** Cada commit debe representar una unidad de trabajo lógica, pequeña y completa. Los mensajes de commit deben ser claros, concisos y seguir un formato estándar (e.g., `feat(gateway): Add endpoint for project creation`).

5.  **Mecanismo de Rollback:** Todos los cambios deben ser desplegables y, en caso de fallo, debe existir un plan para revertirlos rápidamente. El uso de ramas de feature y commits atómicos facilita este proceso.

6.  **Versionado de Archivos Nuevos:** Cuando se modifique un archivo existente de forma significativa y se quiera preservar el original, o cuando se creen nuevas versiones de artefactos (como plantillas), se utilizarán sufijos para indicar la nueva versión.
    - **Sufijo de propiedad:** `_oyp` (e.g., `plantilla_cliente_oyp.docx`).
    - **Sufijo de versión:** `_v2`, `_v3`, etc. (e.g., `informe_final_v2.docx`).

## Proceso de Cambio

1.  **Rama de Feature:** Todo cambio debe realizarse en una rama de feature dedicada, partiendo de la rama de desarrollo principal (e.g., `feat/nombre-feature`).
2.  **Pull Request (PR):** Una vez completado el desarrollo, se abrirá un Pull Request hacia la rama principal.
3.  **Revisión y Pruebas:** El PR debe ser revisado por al menos otro miembro del equipo y debe pasar todas las pruebas automatizadas (unitarias, integración, e2e).
4.  **Merge:** Solo después de la aprobación y la validación de las pruebas, el código podrá ser fusionado.
