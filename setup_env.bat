@echo off
echo === Environment Setup Script ===
echo.

echo [1/4] Checking Python version...
python --version
echo.

echo [2/4] Checking pip version...
python -m pip --version
echo.

echo [3/4] Installing required packages...
python -m pip install fastapi==0.116.1 uvicorn[standard]==0.35.0 websockets==14.1 python-dotenv==1.0.1 PyYAML==6.0.2 pydantic==2.11.7
echo.

echo [4/4] Verifying installations...
python -c "import fastapi, uvicorn, websockets, pydantic; print(f'FastAPI: {fastapi.__version__}'); print(f'Uvicorn: {uvicorn.__version__}'); print(f'WebSockets: {websockets.__version__}'); print(f'Pydantic: {pydantic.__version__}')"
echo.

echo Setup complete!
pause
