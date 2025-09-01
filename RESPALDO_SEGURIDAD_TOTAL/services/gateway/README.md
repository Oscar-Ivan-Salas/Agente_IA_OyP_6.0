# Gateway Service

Gateway service for the Agente IA OYP 6.0 platform, built with FastAPI. This service acts as the main entry point for all client requests and coordinates communication between different microservices.

## Features

- **RESTful API** with OpenAPI/Swagger documentation
- **WebSocket** support for real-time updates
- **Job Queue** for background task processing
- **Authentication & Authorization** (JWT-based)
- **CORS** enabled for cross-origin requests
- **Logging** and error handling
- **Health checks** and monitoring endpoints

## Prerequisites

- Python 3.9+
- SQLite (for development)
- Redis (for production job queue)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/agente-ia-oyp-6.0.git
   cd agente-ia-oyp-6.0/services/gateway
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Copy the example environment file and update the values:

```bash
cp ../../.env.example .env
```

Edit the `.env` file with your configuration:

```env
# Server settings
ENVIRONMENT=development
DEBUG=true
HOST=0.0.0.0
PORT=8080

# Database
DATABASE_URL=sqlite:///./oyp.sqlite

# Security
SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=1440  # 24 hours

# CORS (comma-separated list of origins)
BACKEND_CORS_ORIGINS=["http://localhost:3000","http://localhost:8080"]

# Service URLs
AI_SERVICE_URL=http://ai:8001
DOCS_SERVICE_URL=http://docproc:8002
ANALYTICS_SERVICE_URL=http://analytics:8003
REPORTS_SERVICE_URL=http://reports:8004
CHAT_SERVICE_URL=http://chat:8005
```

## Running the Service

### Development

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8080
```

The API documentation will be available at:
- Swagger UI: http://localhost:8080/docs
- ReDoc: http://localhost:8080/redoc

### Production

For production, use a production-grade ASGI server like Uvicorn with Gunicorn:

```bash
gunicorn -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:8080 main:app
```

## API Endpoints

### Authentication

- `POST /api/v1/auth/login` - Authenticate and get an access token
- `POST /api/v1/auth/refresh` - Refresh an access token
- `GET /api/v1/auth/me` - Get current user information

### Projects

- `GET /api/v1/projects` - List all projects
- `POST /api/v1/projects` - Create a new project
- `GET /api/v1/projects/{project_id}` - Get project details
- `PUT /api/v1/projects/{project_id}` - Update a project
- `DELETE /api/v1/projects/{project_id}` - Delete a project

### Tasks

- `GET /api/v1/tasks` - List all tasks
- `POST /api/v1/tasks` - Create a new task
- `GET /api/v1/tasks/{task_id}` - Get task details
- `PUT /api/v1/tasks/{task_id}` - Update a task
- `DELETE /api/v1/tasks/{task_id}` - Delete a task

### Jobs

- `GET /api/v1/jobs` - List all jobs
- `POST /api/v1/jobs` - Create a new job
- `GET /api/v1/jobs/{job_id}` - Get job status
- `DELETE /api/v1/jobs/{job_id}` - Cancel a job
- `WS /ws/jobs/{job_id}` - WebSocket for job updates

## WebSocket API

The WebSocket API allows real-time updates for long-running operations:

```javascript
const ws = new WebSocket('ws://localhost:8080/ws/jobs/{job_id}');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Job update:', data);
  // Update UI with job progress
};

// Send ping every 30 seconds to keep connection alive
setInterval(() => ws.send('ping'), 30000);
```

## Testing

Run the test suite:

```bash
pytest
```

## Deployment

### Docker

Build the Docker image:

```bash
docker build -t agente-ia-oyp-gateway .
```

Run the container:

```bash
docker run -d --name gateway -p 8080:8080 --env-file .env agente-ia-oyp-gateway
```

### Kubernetes

Example deployment manifest:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gateway
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gateway
  template:
    metadata:
      labels:
        app: gateway
    spec:
      containers:
      - name: gateway
        image: agente-ia-oyp-gateway:latest
        ports:
        - containerPort: 8080
        envFrom:
        - secretRef:
            name: gateway-secrets
---
apiVersion: v1
kind: Service
metadata:
  name: gateway
spec:
  selector:
    app: gateway
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
