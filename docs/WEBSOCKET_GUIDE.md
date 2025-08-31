# WebSocket Implementation Guide

This document provides an overview of the WebSocket implementation in the Agente IA OYP 6.0 Gateway.

## Architecture

The WebSocket implementation consists of:

1. **Connection Manager**: Handles WebSocket connections and message routing
2. **WebSocket Endpoint**: Main WebSocket route for client connections
3. **Message Types**: Standardized message format for different operations

## Setup

### Dependencies

Ensure these dependencies are installed:

```bash
pip install websockets fastapi uvicorn
```

### Running the Server

Start the FastAPI server with WebSocket support:

```bash
uvicorn gateway.main:app --reload --host 0.0.0.0 --port 8000
```

## Testing

### Test Script

A test script is available at `tests/test_websocket.py`:

```bash
# Install test dependencies
pip install -r tests/requirements-test.txt

# Run the test
python -m tests.test_websocket
```

## API Reference

### WebSocket Endpoint

- **URL**: `ws://localhost:8000/ws/{client_id}`
- **Protocol**: WebSocket

### Message Types

#### 1. Echo (Test)
```json
{
  "type": "echo",
  "data": "Test message"
}
```

#### 2. Subscribe to Channels
```json
{
  "type": "subscribe",
  "channels": ["updates", "notifications"]
}
```

#### 3. Unsubscribe from Channels
```json
{
  "type": "unsubscribe",
  "channels": ["updates"]
}
```

## Error Handling

- Invalid JSON: Returns `{"type": "error", "message": "Invalid JSON format"}`
- Disconnection: Automatically cleans up resources

## Security Considerations

- Client authentication should be implemented for production use
- Validate all incoming messages
- Implement rate limiting to prevent abuse
