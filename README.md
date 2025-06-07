# dxee-china
Bidirectional information bridge providing official China data to Asia-Europe users and verified global information to mainland China users. Features search for news, official facts, and technical insights from developers and engineers.

# dxee-china Application

This application consists of three main components:
- Backend (Tornado)
- MongoDB (Primary Database)
- Meilisearch (Search Engine)

## Architecture

The application uses a local virtual environment for the backend and a local MongoDB installation, while Meilisearch runs in a Docker container. The backend provides API endpoints for search and chat functionality, with AI-powered responses using the DeepSeek API.

- **MongoDB** (installed locally) stores user information and chat history for all users
- **Meilisearch** stores selected chats that users choose to share, enabling search functionality across shared content
- **User Authentication** supports multiple login methods including email, Gmail, Microsoft 365, and Apple accounts

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

### 2. Install and Configure MongoDB

Install MongoDB locally following the official documentation for your operating system:
- [MongoDB Installation Guide](https://docs.mongodb.com/manual/installation/)

Start the MongoDB service:
```bash
# For Ubuntu/Debian
sudo systemctl start mongod

# For macOS (if installed via Homebrew)
brew services start mongodb-community

# For Windows
# MongoDB should be running as a service
```

### 3. Start Meilisearch

Start the Meilisearch container using Docker Compose:

```bash
docker-compose up -d
```

### 4. Start the Backend

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
- MongoDB: 27017 (default for local MongoDB)
- Meilisearch: 7701

## Environment Variables

The following environment variables can be configured in the backend/.env file:

### Backend
- PORT=8888
- DEBUG=True
- MONGODB_URI=mongodb://localhost:27017/deepchina
- MONGODB_DB_NAME=deepchina
- MEILISEARCH_HOST=http://localhost:7701
- MEILISEARCH_API_KEY=masterKey
- DEEPSEEK_API_KEY=your_deepseek_api_key_here
- DEEPSEEK_API_URL=https://api.deepseek.com
- LOG_LEVEL=INFO
- AUTH_SECRET_KEY=your_auth_secret_key_here

## Features

### User Authentication
The application supports multiple authentication methods:
- Email with password
- OAuth login with Google (Gmail)
- OAuth login with Microsoft 365
- OAuth login with Apple

### Data Storage
- **MongoDB**: Primary database for storing user information and all chat history
- **Meilisearch**: Search engine for storing and searching shared chats

### Search
The application uses Meilisearch to provide fast and relevant search results. When a user searches for a query, the system will match it against shared chats in Meilisearch. If matches are found, the system will retrieve the complete data from MongoDB and display it to the user.

### Chat
The application provides an AI-powered chat functionality using the DeepSeek API. Users can:
- Start new chat sessions
- Continue previous conversations
- View their chat history
- Choose to share specific chats, which will be indexed in Meilisearch for other users to discover

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
- User data and all chat history are stored in MongoDB for persistence.
- Selected chats that users choose to share are stored in Meilisearch for search functionality.
- The DeepSeek API is used for generating AI responses to chat messages.
- The application is designed to support up to 2000 users with approximately 200 chat entries per user.
