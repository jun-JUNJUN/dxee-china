# DeepSchina Project

A bidirectional information bridge application providing official China data to Asia-Europe users and verified global information to mainland China users.

## Architecture

- **Backend**: Tornado (Python async web framework)
- **Database**: MongoDB (local installation)
- **Search**: Meilisearch (Docker container)
- **AI**: DeepSeek API for chat completions
- **Auth**: Multi-provider OAuth + email/password
- **Python**: >=3.11 with UV package manager

## Quick Start

1. **Setup environment**: `./setup_venvs.sh`
2. **Activate backend**: `./activate_backend.sh`
3. **Start search engine**: `docker-compose up -d`
4. **Start backend**: `cd backend && ./run.sh`

## Project Structure

### Backend (`/backend/`)
- **app/handler/**: Request handlers (auth, chat, search, health)
- **app/service/**: Business logic (deepseek, mongodb, search)
- **tornado_main.py**: Main application entry point
- **wsgi.py**: WSGI server configuration

### Key Files
- **pyproject.toml**: Modern Python project configuration
- **requirements.txt**: Python dependencies
- **docker-compose.yml**: Meilisearch container setup
- **.env.example**: Environment variables template

## Environment Variables

```bash
PORT=8888
MONGODB_URI=mongodb://localhost:27017
MEILISEARCH_HOST=http://localhost:7701
DEEPSEEK_API_KEY=your_key_here
AUTH_SECRET_KEY=your_secret_here
```

## API Endpoints

- `/auth/*` - Authentication (register, login, OAuth)
- `/chat/*` - Chat messaging and history
- `/search` - Content search
- `/health` - Service health check

## Development Commands

### Backend
```bash
# Setup and activate environment
./setup_venvs.sh
./activate_backend.sh

# Start development server
cd backend && ./run.sh

# Test API
./backend/test_api.sh
```

### Dependencies
- **tornado**: 6.4.2 (web framework)
- **pymongo/motor**: MongoDB drivers
- **meilisearch-python-sdk**: Search integration
- **openai**: DeepSeek API client
- **bcrypt**: Password hashing
- **PyJWT**: JWT tokens
- **gunicorn**: WSGI server

## Features

### Authentication
- OAuth providers: Google, Microsoft, Apple
- Email/password registration
- JWT token sessions
- Secure cookie management

### Chat System
- AI conversations via DeepSeek API
- Chat history in MongoDB
- Message sharing functionality
- Async processing queues

### Search
- Meilisearch for fast content search
- MongoDB for complete data retrieval
- Shared content indexing

## Data Architecture

- **MongoDB**: Primary storage (users, chat history)
- **Meilisearch**: Search index (shared chats)
- **Dual storage**: Complete data + searchable subset

## Deployment

The application uses Gunicorn with Tornado workers for production deployment. Nginx configuration is included for reverse proxy setup.

## Testing

- **test_api.sh**: API endpoint testing
- **test_deepseek_api.py**: AI service testing

## Notes

- Supports ~2000 users with ~200 chats each
- Modern Python packaging with pyproject.toml
- Async-first architecture
- Local database for privacy/control
- Container-based search engine