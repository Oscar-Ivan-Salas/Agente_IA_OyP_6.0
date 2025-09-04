from fastapi import FastAPI
import uvicorn

app = FastAPI(title="Test Server", description="A simple test server")

@app.get("/")
async def read_root():
    return {"message": "Hello, World! This is a test server."}

if __name__ == "__main__":
    print("Starting test server on http://localhost:8000")
    uvicorn.run("test_server:app", host="0.0.0.0", port=8000, reload=True)
