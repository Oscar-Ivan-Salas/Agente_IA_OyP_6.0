@echo off
echo Starting Gateway...
python -u -c "import uvicorn; uvicorn.run('gateway.main:app', host='127.0.0.1', port=8000, log_level='info')" > gateway.log 2>&1
pause
