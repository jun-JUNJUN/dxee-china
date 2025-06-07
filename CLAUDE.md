# dxee-china Project

A bidirectional information bridge application providing official China data to Asia-Europe users and verified global information to mainland China users.

## Architecture

- **Backend**: Tornado (Python async web framework)
- **Database**: MongoDB (local installation)
- **Search**: Meilisearch (Docker container)
- **AI**: DeepSeek API for chat completions with streaming support
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
PORT=8100
MONGODB_URI=mongodb://localhost:27017
MEILISEARCH_HOST=http://localhost:7701
DEEPSEEK_API_KEY=your_key_here
AUTH_SECRET_KEY=your_secret_here
```

## API Endpoints

### Chat System
- `/chat/stream` - **Real-time streaming chat** (Server-Sent Events)
- `/chat/message` - Legacy non-streaming chat
- `/chat/history/{chat_id}` - Chat conversation history
- `/chat/user` - User's chat list
- `/chat/share/{message_id}` - Share/unshare messages
- `/chat/shared` - Browse shared messages

### Authentication
- `/auth/register` - Email registration
- `/auth/login` - Email login
- `/auth/google` - Google OAuth
- `/auth/microsoft` - Microsoft OAuth
- `/auth/apple` - Apple OAuth
- `/auth/profile` - User profile
- `/auth/logout` - Logout

### Other
- `/search` - Content search
- `/health` - Service health check
- `/` - Main chat interface

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
- **openai**: DeepSeek API client with streaming support
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
- **Real-time streaming responses**: Word-by-word AI responses like ChatGPT
- AI conversations via DeepSeek API with streaming support
- Chat history in MongoDB
- Message sharing functionality
- Async processing queues
- Stream queues for real-time communication

### Search
- Meilisearch for fast content search
- MongoDB for complete data retrieval
- Shared content indexing

## Streaming Implementation

### How It Works
1. **Frontend**: Uses Fetch API with ReadableStream for Server-Sent Events
2. **Backend**: ChatStreamHandler manages real-time streaming via SSE
3. **AI Service**: DeepSeek API called with `stream=True` parameter
4. **Processing**: Async stream queues handle real-time chunks
5. **Storage**: Complete messages saved to MongoDB after streaming

### Technical Details
- **Streaming endpoint**: `/chat/stream` (default in UI)
- **Legacy fallback**: `/chat/message` (non-streaming)
- **Protocol**: Server-Sent Events (SSE) over HTTP
- **Queue management**: Per-chat stream queues with cleanup
- **Error handling**: Graceful degradation to legacy mode

## Data Architecture

- **MongoDB**: Primary storage (users, chat history)
- **Meilisearch**: Search index (shared chats)
- **Dual storage**: Complete data + searchable subset
- **Stream queues**: Real-time response chunks (memory-based)

## Deployment

The application uses Gunicorn with Tornado workers for production deployment. Nginx configuration is included for reverse proxy setup.

### Production Command
```bash
cd backend
uv run gunicorn --bind 0.0.0.0:8100 --workers=1 --worker-class=tornado wsgi:application
```

## Testing

- **test_api.sh**: API endpoint testing
- **test_deepseek_api.py**: AI service testing
- **Manual testing**: Chat streaming, authentication, search functionality

## Performance Notes

- Supports ~2000 users with ~200 chats each
- Modern Python packaging with pyproject.toml
- Async-first architecture for concurrent requests
- Local database for privacy/control
- Container-based search engine
- Memory-efficient streaming with automatic cleanup

## Streaming vs Non-Streaming

### Streaming Mode (Default)
- **Endpoint**: `/chat/stream`
- **Protocol**: Server-Sent Events
- **Experience**: Real-time word-by-word responses
- **Latency**: Immediate feedback, progressive display

### Legacy Mode (Fallback)
- **Endpoint**: `/chat/message`
- **Protocol**: Standard HTTP POST/Response
- **Experience**: Wait for complete response
- **Compatibility**: Fallback for streaming failures

## Development Guidelines

1. **Async-first**: Use async/await for all I/O operations
2. **Streaming support**: Maintain both streaming and legacy endpoints
3. **Error handling**: Implement graceful degradation
4. **Memory management**: Clean up stream queues properly
5. **Testing**: Test both streaming and non-streaming modes
6. **Logging**: Use structured logging for debugging

## Environment Setup Notes

- **UV package manager**: Modern Python dependency management
- **Virtual environment**: Backend uses `.venv` directory
- **Database**: Local MongoDB for privacy and control
- **Search**: Dockerized Meilisearch for easy deployment
- **Configuration**: Environment variables in `.env` file