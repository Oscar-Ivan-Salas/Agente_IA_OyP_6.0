import os, re, sys, json, hashlib, ast
from pathlib import Path
from datetime import datetime

# Configuración
IGNORE_DIRS = {".git", "__pycache__", "node_modules", "dist", "build", 
               "backups", ".venv", "venv", ".idea", ".vscode", "tools"}
PY_ENDPOINT_DECOS = {"get", "post", "put", "delete", "patch", "options", "head"}

def sha256_file(p: Path, max_mb=50):
    try:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024*1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

def read_text_safe(p: Path, limit=200_000):
    try:
        data = p.read_bytes()
        if len(data) > limit: 
            data = data[:limit]
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""

def analyze_python(p: Path, text: str):
    info = {
        "creates_fastapi_app": False,
        "defines_apirouter": [],
        "endpoints": [],
        "websockets": [],
        "mounts": [],
        "templates": False,
        "imports": []
    }
    try:
        tree = ast.parse(text or "")
    except Exception:
        return info

    # Analizar imports
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                info["imports"].append(a.name)
        elif isinstance(n, ast.ImportFrom):
            mod = n.module or ""
            info["imports"].append(mod)

    # Analizar llamadas y decoradores
    for n in ast.walk(tree):
        try:
            # FastAPI() o APIRouter()
            if isinstance(n, ast.Call):
                fn = n.func
                fn_name = None
                if isinstance(fn, ast.Name):
                    fn_name = fn.id
                elif isinstance(fn, ast.Attribute):
                    fn_name = fn.attr

                if fn_name == "FastAPI":
                    info["creates_fastapi_app"] = True

                if fn_name == "APIRouter":
                    prefix_val = None
                    for kw in n.keywords:
                        if kw.arg == "prefix" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                            prefix_val = kw.value.value
                    info["defines_apirouter"].append({"prefix": prefix_val})

                # app.mount("/static", ...)
                if isinstance(fn, ast.Attribute) and fn.attr == "mount":
                    if n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
                        info["mounts"].append(n.args[0].value)

                # Jinja2Templates(...)
                if fn_name == "Jinja2Templates":
                    info["templates"] = True

            # Decoradores: @app.get("/x") o @router.post("/y") o @app.websocket("/ws")
            if hasattr(n, "decorator_list"):
                for d in n.decorator_list:
                    try:
                        if isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute):
                            deco = d.func.attr
                            if deco in PY_ENDPOINT_DECOS:
                                path = None
                                if d.args and isinstance(d.args[0], ast.Constant) and isinstance(d.args[0].value, str):
                                    path = d.args[0].value
                                info["endpoints"].append({"method": deco.upper(), "path": path})
                            if d.func.attr == "websocket":
                                ws_path = None
                                if d.args and isinstance(d.args[0], ast.Constant) and isinstance(d.args[0].value, str):
                                    ws_path = d.args[0].value
                                info["websockets"].append(ws_path)
                    except Exception:
                        pass
        except Exception:
            pass
    return info

def analyze_js(p: Path, text: str):
    info = {
        "ws_urls": re.findall(r'new\s+WebSocket\s*\(\s*([\'\"])(.*?)\1\s*\)', text, flags=re.I),
        "api_calls": re.findall(r'["\'](/api/[\w\-/\.]+)["\']', text),
        "uses_agenteia_utils": bool(re.search(r'AgenteIA\.utils', text)),
        "expects_loadScripts": bool(re.search('\.loadScripts\s*\\(', text)),
        "expects_showAlert": bool(re.search('\.showAlert\s*\\(', text)),
    }
    info["ws_urls"] = [m[1] for m in info["ws_urls"]]
    return info

def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "audit_report.json").resolve()
    
    result = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "root": str(root),
        "files": [],
        "entrypoints": [],
        "routers": [],
        "route_map": {},
        "websockets": [],
        "mounts": [],
        "templates_in_use": [],
        "duplicates_by_hash": {},
        "duplicates_by_name": {},
        "warnings": []
    }

    by_hash = {}
    by_name = {}

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        for fn in filenames:
            p = Path(dirpath) / fn
            rel = str(p.relative_to(root))
            
            # Ignorar archivos grandes o no relevantes
            if p.suffix.lower() in {'.zip', '.pyc', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.pdf'}:
                continue
                
            size = p.stat().st_size
            mtime = datetime.fromtimestamp(p.stat().st_mtime).isoformat()
            ext = p.suffix.lower()
            meta = {"path": rel, "size": size, "mtime": mtime, "ext": ext}

            # Calcular hash
            h = sha256_file(p)
            if h:
                meta["sha256"] = h
                by_hash.setdefault(h, []).append(rel)

            by_name.setdefault(p.name, []).append(rel)

            # Analizar contenido de archivos relevantes
            text = ""
            if ext in {".py", ".js", ".ts", ".json", ".yml", ".yaml", ".html", ".css"}:
                text = read_text_safe(p)

            if ext == ".py":
                py = analyze_python(p, text)
                meta["py"] = py
                if py["creates_fastapi_app"]:
                    result["entrypoints"].append(rel)
                for r in py["defines_apirouter"]:
                    result["routers"].append({"file": rel, **r})
                for ep in py["endpoints"]:
                    k = ep["path"] or "UNSPECIFIED"
                    result["route_map"].setdefault(k, []).append({"method": ep["method"], "file": rel})
                for ws in py["websockets"]:
                    result["websockets"].append({"path": ws, "file": rel})
                for m in py["mounts"]:
                    result["mounts"].append({"mount": m, "file": rel})
                if py["templates"]:
                    result["templates_in_use"].append(rel)

            elif ext in {".js", ".ts"}:
                js = analyze_js(p, text)
                meta["js"] = js
                for api in js["api_calls"]:
                    result["route_map"].setdefault(api, []).append({"method": "CLIENT", "file": rel})
                for wsu in js["ws_urls"]:
                    result["websockets"].append({"path_ref": wsu, "file": rel})

            result["files"].append(meta)

    # Identificar duplicados
    result["duplicates_by_hash"] = {h: paths for h, paths in by_hash.items() if len(paths) > 1}
    result["duplicates_by_name"] = {n: paths for n, paths in by_name.items() if len(paths) > 1}

    # Generar advertencias
    if len(result["entrypoints"]) > 1:
        result["warnings"].append(f"Múltiples puntos de entrada FastAPI: {result['entrypoints']}")
    
    gw_dirs = [f for f in result["files"] if re.search(r'(^|/)gateway(/|$)', f["path"])]
    if not gw_dirs:
        result["warnings"].append("No se encontró carpeta 'gateway/'.")

    # Guardar resultados
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Informe generado: {out}")
    print(f"- Archivos analizados: {len(result['files'])}")
    print(f"- Puntos de entrada: {len(result['entrypoints'])}")
    print(f"- Rutas definidas: {len(result['route_map'])}")
    print(f"- WebSockets: {len([w for w in result['websockets'] if 'path' in w])}")
    print(f"- Advertencias: {len(result['warnings'])}")

if __name__ == "__main__":
    main()
