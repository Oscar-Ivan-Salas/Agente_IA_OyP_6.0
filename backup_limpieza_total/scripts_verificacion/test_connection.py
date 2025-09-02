import requests
import sys

def test_connection():
    try:
        # Probamos el endpoint de salud
        response = requests.get('http://localhost:8000/health')
        print("✅ Conexión exitosa al servidor")
        print(f"Respuesta: {response.status_code} - {response.json()}")
        
        # Probamos el endpoint de WebSocket (solo verificación de conexión)
        print("\nProbando conexión WebSocket...")
        import websockets
        import asyncio
        
        async def test_ws():
            try:
                async with websockets.connect('ws://localhost:8000/ws') as ws:
                    await ws.send("test")
                    response = await ws.recv()
                    print(f"✅ WebSocket funcionando. Respuesta: {response}")
            except Exception as e:
                print(f"❌ Error en WebSocket: {e}")
        
        asyncio.get_event_loop().run_until_complete(test_ws())
        
    except requests.exceptions.ConnectionError:
        print("❌ No se pudo conectar al servidor. Asegúrate de que el servidor esté en ejecución.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_connection()
