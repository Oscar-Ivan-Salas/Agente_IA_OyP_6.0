"""
Test script for WebSocket functionality.
"""
import asyncio
import websockets
import json
import uuid
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# WebSocket server URL
WS_URL = "ws://localhost:8000/ws"

async def test_websocket_connection():
    """Test WebSocket connection and message exchange."""
    client_id = str(uuid.uuid4())
    logger.info(f"Connecting to WebSocket with client ID: {client_id}")
    
    try:
        async with websockets.connect(f"{WS_URL}/{client_id}", ping_interval=20, ping_timeout=20) as websocket:
            logger.info("✅ Connected to WebSocket server")
            
            # Test echo functionality
            test_message = {"type": "echo", "data": "Hello, WebSocket!"}
            await websocket.send(json.dumps(test_message))
            logger.info(f"📤 Sent: {test_message}")
            
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            logger.info(f"📥 Received: {response}")
            
            # Test subscription
            subscribe_msg = {
                "type": "subscribe",
                "channels": ["updates", "notifications"]
            }
            await websocket.send(json.dumps(subscribe_msg))
            logger.info(f"📤 Sent subscription: {subscribe_msg}")
            
            # Wait for any potential responses
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                logger.info(f"📥 Received: {response}")
            except asyncio.TimeoutError:
                logger.info("ℹ️ No immediate response to subscription")
            
            # Keep connection open for a moment to test stability
            await asyncio.sleep(1)
            
    except websockets.exceptions.ConnectionClosed as e:
        logger.error(f"❌ Connection closed: {e}")
    except asyncio.TimeoutError:
        logger.error("❌ Connection or operation timed out")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}", exc_info=True)
    finally:
        logger.info("🔌 Disconnected from WebSocket server")

if __name__ == "__main__":
    try:
        asyncio.run(test_websocket_connection())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise
