import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello, World! This is a test FastAPI server."}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    with open("simple_api.log", "w") as f:
        f.write("Starting simple API server...\n")
    
    uvicorn.run(
        "simple_api:app",
        host="127.0.0.1",
        port=8000,
        log_level="info",
        reload=False
    )
