print("Probando importaciones...")
try:
    from services.gateway.app import app
    print("¡Importación exitosa!")
    print(f"Aplicación FastAPI: {app}")
except Exception as e:
    print(f"Error al importar: {e}")
    import traceback
    traceback.print_exc()
