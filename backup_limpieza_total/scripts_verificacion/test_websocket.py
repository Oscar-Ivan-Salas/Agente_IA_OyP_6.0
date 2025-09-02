import asyncio
import websockets

async def test_websocket():
    uri = "ws://localhost:8080/ws"
    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected to {uri}")
            
            # Send a test message
            test_message = "ping"
            print(f"Sending: {test_message}")
            await websocket.send(test_message)
            
            # Wait for the response
            response = await websocket.recv()
            print(f"Received: {response}")
            
            if response == f"echo:{test_message}":
                print("✅ WebSocket test passed!")
            else:
                print(f"❌ Unexpected response: {response}")
                
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())
