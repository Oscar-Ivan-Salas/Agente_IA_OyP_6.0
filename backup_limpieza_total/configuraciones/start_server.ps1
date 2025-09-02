# Start the Uvicorn server with the correct Python path

# Go to the root directory
Set-Location C:\Users\USUARIO\Agente_IA_OyP_6.0

# Activate the virtual environment
. \.venv\Scripts\Activate.ps1

# Set the Python path to include the root directory
$env:PYTHONPATH = (Resolve-Path .)

# Set the websocket protocol
$env:UVICORN_WS = "wsproto"

# Start the server
python -m uvicorn gateway.main:app --host 127.0.0.1 --port 8000 --reload --ws wsproto
