/* ==========================================================================
   Compatibilidad temprana AgenteIA (NO BORRAR)
   - Provee AgenteIA.utils y alias en la raíz (AgenteIA.loadScripts / showAlert)
   - No interfiere con implementaciones existentes (solo rellena huecos)
   ========================================================================== */
(function(w){
  const A = w.AgenteIA = w.AgenteIA || {};

  // Base mínima segura
  A.util = A.util || {};
  A.utils = A.utils || {};

  // Implementaciones de respaldo (no invasivas)
  A.util.mostrarAlerta = A.util.mostrarAlerta || function(mensaje, tipo){
    try { console.log(`[${tipo||'info'}] ${mensaje}`); } catch(_){}
  };

  A.util.cargarScripts = A.util.cargarScripts || (async function(scripts){
    if (!Array.isArray(scripts)) return;
    for (const src of scripts){
      await new Promise((ok, fail)=>{
        const s = document.createElement('script');
        s.src = src;
        s.onload = ok;
        s.onerror = ()=>fail(new Error(`Error cargando ${src}`));
        document.head.appendChild(s);
      });
    }
  });

  // Mapear a utils (si faltan)
  A.utils.showAlert   = A.utils.showAlert   || A.util.mostrarAlerta;
  A.utils.loadScripts = A.utils.loadScripts || A.util.cargarScripts;

  // Alias de nivel raíz exigidos por la UI
  A.showAlert   = A.showAlert   || A.utils.showAlert;
  A.loadScripts = A.loadScripts || A.utils.loadScripts;

  // Alias en español (por si otros módulos los esperan)
  A.utilidades = A.utilidades || A.util;
  A.utilidades.mostrarAlerta = A.utilidades.mostrarAlerta || A.utils.showAlert;
  A.utilidades.cargarScripts = A.utilidades.cargarScripts || A.utils.loadScripts;

  // Señal
  A.__compat_ok__ = true;
})(window);
