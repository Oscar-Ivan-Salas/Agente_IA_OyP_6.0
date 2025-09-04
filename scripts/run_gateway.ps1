Set-Location "$PSScriptRoot\.."
$env:PYTHONPATH = "$PWD"
python -m uvicorn gateway.main:app --host 0.0.0.0 --port 8080 --reload --ws websockets
