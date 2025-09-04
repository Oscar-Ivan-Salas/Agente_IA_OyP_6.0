"""
Script para iniciar el servidor Gateway
"""
import uvicorn

if __name__ == "__main__":
    print("Iniciando servidor Gateway...")
    uvicorn.run(
        "gateway.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug"
    )
