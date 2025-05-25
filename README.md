# dxee-china
Bidirectional information bridge providing official China data to Asia-Europe users and verified global information to mainland China users. Features search for news, official facts, and technical insights from developers and engineers.

# dxee-china Application

This application consists of three main components:
- Frontend (Flask)
- Backend (Tornado)
- Meilisearch (Search Engine)

## New Architecture

The application now uses local virtual environments for the frontend and backend, while Meilisearch runs in a Docker container.

## Setup Instructions

### 1. Set up Virtual Environments

Run the setup script to create virtual environments for both frontend and backend:

```bash
./setup_venvs.sh
```

This script will:
- Create virtual environments for both frontend and backend
- Install all required dependencies
- Make the run scripts executable

### 2. Start Meilisearch

Start the Meilisearch container using Docker Compose:

```bash
docker-compose up -d
```

### 3. Start the Backend

```bash
cd backend
./run.sh
```

The backend will be available at http://localhost:8100

### 4. Start the Frontend

```bash
cd frontend
./run.sh
```

The frontend will be available at http://localhost:5101

## Ports

- Frontend: 5101
- Backend: 8100
- Meilisearch: 7701

## Environment Variables

The following environment variables are set in the run scripts:

### Backend
- PYTHONUNBUFFERED=1
- MEILISEARCH_URL=http://localhost:7701

### Frontend
- FLASK_APP=apps
- FLASK_ENV=development
- MEILISEARCH_URL=http://localhost:7701

## Notes

- The nginx container has been removed as it's no longer needed. The Flask frontend can handle static file serving directly.
- Both frontend and backend now connect directly to Meilisearch at http://localhost:7701.
