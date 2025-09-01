import json
import time
import sys
from datetime import datetime
import httpx
from typing import Dict, Any, Optional

def http_get(client: httpx.Client, url: str) -> Dict[str, Any]:
    """Realiza una petición HTTP GET y devuelve los resultados."""
    try:
        start_time = time.time()
        r = client.get(url, timeout=5)
        elapsed = (time.time() - start_time) * 1000  # ms
        return {
            "status": r.status_code, 
            "ok": r.is_success, 
            "time_ms": f"{elapsed:.1f}ms",
            "body": r.text[:300]  # Solo primeros 300 caracteres
        }
    except Exception as e:
        return {"error": str(e), "ok": False}

def http_post(client: httpx.Client, url: str, json_body: Optional[Dict] = None) -> Dict[str, Any]:
    """Realiza una petición HTTP POST y devuelve los resultados."""
    try:
        start_time = time.time()
        r = client.post(url, json=json_body or {}, timeout=5)
        elapsed = (time.time() - start_time) * 1000  # ms
        return {
            "status": r.status_code, 
            "ok": r.is_success,
            "time_ms": f"{elapsed:.1f}ms",
            "body": r.text[:300]
        }
    except Exception as e:
        return {"error": str(e), "ok": False}

def test_websocket(url: str) -> Dict[str, Any]:
    """Prueba la conexión a un WebSocket."""
    try:
        import websockets
        from websockets.sync.client import connect
        
        start_time = time.time()
        with connect(url, open_timeout=3, close_timeout=3) as ws:
            elapsed = (time.time() - start_time) * 1000
            return {
                "ok": True, 
                "time_ms": f"{elapsed:.1f}ms"
            }
    except ImportError:
        return {"error": "websockets no instalado. Ejecuta: pip install websockets", "ok": False}
    except Exception as e:
        return {"error": str(e), "ok": False}

def main():
    # Configuración
    BASE_URL = "http://127.0.0.1:8000"
    
    print(f"🚀 Iniciando auditoría de {BASE_URL}")
    print("=" * 60)
    
    result = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "base_url": BASE_URL,
        "http": {},
        "websockets": {}
    }
    
    # Probar endpoints HTTP
    with httpx.Client(base_url=BASE_URL, timeout=10.0) as client:
        # Endpoints principales
        endpoints = [
            "/",
            "/salud",
            "/api/dashboard/stats",
            "/api/training/projects",
            "/api/agent/history",
            "/static/js/main.js",
            "/static/js/compat.js"
        ]
        
        print("🔍 Probando endpoints HTTP...")
        for endpoint in endpoints:
            print(f"  - GET {endpoint}")
            result["http"][endpoint] = http_get(client, endpoint)
        
        # Endpoints POST
        post_endpoints = [
            ("/api/ai/analyze", {"text": "ping"}),
            ("/api/documents/upload", {"name": "test_audit"}),
            ("/api/analytics/upload_dataset", {"name": "test_dataset"}),
            ("/api/reports/generate", {"q": "test"})
        ]
        
        print("\n📤 Probando endpoints POST...")
        for endpoint, data in post_endpoints:
            print(f"  - POST {endpoint}")
            result["http"][endpoint] = http_post(client, endpoint, data)
    
    # Probar WebSockets
    print("\n🌐 Probando conexiones WebSocket...")
    ws_urls = [
        "ws://127.0.0.1:8000/ws",
        "ws://127.0.0.1:8000/ws/chat"
    ]
    
    for ws_url in ws_urls:
        print(f"  - {ws_url}")
        result["websockets"][ws_url] = test_websocket(ws_url)
    
    # Guardar resultados
    output_file = "audit_runtime.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    # Resumen
    print("\n" + "=" * 60)
    print(f"✅ Auditoría completada. Resultados guardados en: {output_file}")
    
    # Estadísticas
    http_ok = sum(1 for r in result["http"].values() if r.get("ok"))
    ws_ok = sum(1 for r in result["websockets"].values() if r.get("ok"))
    
    print(f"\n📊 Resumen:")
    print(f"- HTTP: {http_ok} de {len(result['http'])} endpoints respondieron correctamente")
    print(f"- WebSockets: {ws_ok} de {len(result['websockets'])} conectaron exitosamente")
    
    # Mostrar advertencias
    if http_ok < len(result['http']) or ws_ok < len(result['websockets']):
        print("\n⚠  Se encontraron problemas:")
        
        # Mostrar endpoints HTTP con error
        for endpoint, res in result["http"].items():
            if not res.get("ok"):
                print(f"  - {endpoint}: {res.get('error', f'HTTP {res.get("status", "error")}')}")
        
        # Mostrar WebSockets con error
        for ws_url, res in result["websockets"].items():
            if not res.get("ok"):
                print(f"  - {ws_url}: {res.get('error', 'falló la conexión')}")

if __name__ == "__main__":
    main()
