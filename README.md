# dxee-china
Bidirectional information bridge providing official China data to Asia-Europe users and verified global information to mainland China users. Features search for news, official facts, and technical insights from developers and engineers.

# dxee-china Application

This application consists of two main components:
- Backend (Tornado)
- Meilisearch (Search Engine)

## Architecture

The application uses a local virtual environment for the backend, while Meilisearch runs in a Docker container. The backend provides API endpoints for search and chat functionality, with AI-powered responses using the DeepSeek API.

## Setup Instructions

### 1. Set up Virtual Environment

Run the setup script to create a virtual environment for the backend:

```bash
./setup_venvs.sh
```

This script will:
- Create a virtual environment for the backend
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

The backend will be available at http://localhost:8888 (or the port specified in your .env file)

## API Endpoints

The backend provides the following API endpoints:

- `/search` - Search for documents
- `/chat/message` - Send a message to the AI chat service
- `/chat/history/{chat_id}` - Get chat history for a specific chat
- `/health` - Health check endpoint

## Ports

- Backend: 8888 (default, configurable in .env)
- Meilisearch: 7701

## Environment Variables

The following environment variables can be configured in the backend/.env file:

### Backend
- PORT=8888
- DEBUG=True
- MEILISEARCH_HOST=http://localhost:7701
- MEILISEARCH_API_KEY=masterKey
- DEEPSEEK_API_KEY=your_deepseek_api_key_here
- DEEPSEEK_API_URL=https://api.deepseek.com
- LOG_LEVEL=INFO

## Features

### Search
The application uses Meilisearch to provide fast and relevant search results. The search functionality is implemented in the `SearchService` class and exposed through the `/search` endpoint.

### Chat
The application provides an AI-powered chat functionality using the DeepSeek API. The chat functionality is implemented in the `DeepSeekService` class and exposed through the `/chat/message` and `/chat/history/{chat_id}` endpoints.

## Project Structure

- `backend/` - Backend code (Tornado)
  - `app/` - Application code
    - `handler/` - Request handlers
    - `service/` - Service classes
  - `templates/` - HTML templates
- `example/` - Example code
- `web/` - Nginx configuration
- `scripts/` - Utility scripts

## Notes

- The application uses Tornado as the web framework for the backend.
- Chat messages and responses are stored in Meilisearch for persistence.
- The DeepSeek API is used for generating AI responses to chat messages.
