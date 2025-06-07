# dxee-china

A bidirectional information bridge application providing official China data to Asia-Europe users and verified global information to mainland China users. Features real-time streaming AI chat, search functionality, and multi-provider authentication.

## 🚀 Key Features

- **🔄 Real-time Streaming Chat**: AI responses stream word-by-word like ChatGPT
- **🔍 Fast Search**: Meilisearch-powered content discovery
- **🔐 Multi-auth**: Google, Microsoft, Apple OAuth + email/password
- **💾 Persistent Storage**: MongoDB for chat history and user data
- **⚡ Modern Stack**: Tornado, UV package manager, async-first design

## Architecture

- **Backend**: Tornado (Python async web framework)
- **Database**: MongoDB (local installation)
- **Search**: Meilisearch (Docker container)
- **AI**: DeepSeek API with streaming support
- **Auth**: Multi-provider OAuth + email/password
- **Python**: >=3.11 with UV package manager

## Quick Start

### 1. Environment Setup
```bash
# Setup virtual environments and dependencies
./setup_venvs.sh

# Activate backend environment
./activate_backend.sh
```

### 2. Services Setup
```bash
# Start MongoDB (install locally first)
# macOS: brew services start mongodb-community
# Ubuntu: sudo systemctl start mongod

# Start Meilisearch (Docker)
docker-compose up -d

# Start backend server
cd backend && ./run.sh
```

### 3. Access Application
- **Main Interface**: http://localhost:8100
- **Health Check**: http://localhost:8100/health

## Project Structure

```
backend/
├── app/
│   ├── handler/           # Request handlers
│   │   ├── auth_handler.py      # Authentication (OAuth, login)
│   │   ├── chat_handler.py      # Chat (streaming + legacy)
│   │   ├── search_handler.py    # Content search
│   │   ├── health_handler.py    # Health checks
│   │   └── main_handler.py      # Main page
│   ├── service/           # Business logic
│   │   ├── deepseek_service.py  # AI streaming service
│   │   ├── mongodb_service.py   # Database operations
│   │   └── search_service.py    # Search functionality
│   └── tornado_main.py    # Application entry point
├── templates/
│   └── index.html         # Chat interface with streaming
├── pyproject.toml         # Modern Python config
├── requirements.txt       # Dependencies
├── run.sh                # Development server
├── test_api.sh           # API testing
└── wsgi.py               # Production server
```

## API Endpoints

### Chat System
- `POST /chat/stream` - **🔄 Streaming chat** (Server-Sent Events)
- `POST /chat/message` - Legacy non-streaming chat
- `GET /chat/history/{chat_id}` - Chat conversation history
- `GET /chat/user` - User's chat list
- `POST /chat/share/{message_id}` - Share/unshare messages
- `GET /chat/shared` - Browse shared messages

### Authentication
- `POST /auth/register` - Email registration
- `POST /auth/login` - Email login
- `GET /auth/google` - Google OAuth
- `GET /auth/microsoft` - Microsoft OAuth
- `GET /auth/apple` - Apple OAuth
- `GET /auth/profile` - User profile
- `POST /auth/logout` - Logout

### Other
- `GET /search` - Content search
- `GET /health` - Service health check
- `GET /` - Main chat interface

## Environment Configuration

Create `backend/.env` from `.env.example`:

```bash
# Server
PORT=8100
DEBUG=True

# Database
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=dxeechina

# Search Engine
MEILISEARCH_HOST=http://localhost:7701
MEILISEARCH_API_KEY=masterKey

# AI Service
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_API_URL=https://api.deepseek.com

# Authentication
AUTH_SECRET_KEY=your_auth_secret_key_here
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
MICROSOFT_CLIENT_ID=your_microsoft_client_id
MICROSOFT_CLIENT_SECRET=your_microsoft_client_secret
APPLE_CLIENT_ID=your_apple_client_id
APPLE_CLIENT_SECRET=your_apple_client_secret

# Logging
LOG_LEVEL=INFO
```

## 🔄 Streaming Chat Implementation

The application now features real-time streaming chat responses:

### How Streaming Works
1. **User sends message** → Immediate display in chat
2. **Server processes request** → AI service starts generating response
3. **Response streams back** → Text appears word-by-word in real-time
4. **Complete response** → Saved to MongoDB for history

### Technical Implementation
- **Frontend**: Fetch API with ReadableStream for SSE processing
- **Backend**: Server-Sent Events with async streaming queues
- **AI Service**: DeepSeek API with `stream=True` parameter
- **Database**: Complete messages stored after streaming completes

### Streaming vs Legacy
- **Streaming endpoint**: `/chat/stream` (default in UI)
- **Legacy endpoint**: `/chat/message` (fallback available)
- **Graceful degradation**: Falls back to legacy if streaming fails

## Dependencies

### Core Framework
- **tornado**: 6.4.2 - Async web framework
- **gunicorn**: WSGI server for production
- **motor**: Async MongoDB driver
- **pymongo**: MongoDB operations

### AI & Search
- **openai**: DeepSeek API client with streaming support
- **meilisearch-python-sdk**: Search integration

### Authentication
- **bcrypt**: Password hashing
- **PyJWT**: JWT tokens
- **google-auth**: Google OAuth
- **msal**: Microsoft OAuth

### Development
- **python-dotenv**: Environment variables
- **asyncio**: Async programming support

## Data Architecture

### Primary Storage (MongoDB)
- **Users collection**: Authentication and profile data
- **Chats collection**: Chat metadata and titles
- **Messages collection**: Complete chat history
- **Indexes**: Optimized for user queries and chat retrieval

### Search Index (Meilisearch)
- **Shared messages**: User-selected content for discovery
- **Fast search**: Full-text search across shared conversations
- **Privacy**: Only explicitly shared content indexed

### Streaming Queues
- **Input queue**: Messages waiting for AI processing
- **Stream queues**: Real-time response chunks per chat session
- **Memory-based**: Temporary queues for active streaming

## Development Commands

```bash
# Environment setup
./setup_venvs.sh
source activate_backend.sh

# Development server
cd backend && ./run.sh

# API testing
./backend/test_api.sh

# Test DeepSeek integration
python ./backend/test_deepseek_api.py

# Check service health
curl http://localhost:8100/health
```

## Production Deployment

The application is production-ready with:

- **Gunicorn**: WSGI server with Tornado workers
- **Nginx**: Reverse proxy configuration included
- **Environment isolation**: UV-based dependency management
- **Logging**: Structured logging to files and console
- **Error handling**: Graceful degradation and retry logic

### Production Start
```bash
cd backend
uv run gunicorn --bind 0.0.0.0:8100 --workers=1 --worker-class=tornado wsgi:application
```

## Performance & Scalability

- **Capacity**: ~2000 users with ~200 chats each
- **Async design**: Non-blocking I/O for concurrent requests
- **Streaming**: Reduced perceived latency with real-time responses
- **Local databases**: Privacy-focused, no external dependencies
- **Memory efficient**: Queue-based processing with cleanup

## Testing

### Automated Tests
- `test_api.sh` - HTTP endpoint testing
- `test_deepseek_api.py` - AI service integration testing

### Manual Testing
1. **Chat streaming**: Send message and verify real-time response
2. **Authentication**: Test OAuth providers and email login
3. **Search**: Share messages and search for content
4. **History**: Verify chat persistence and retrieval

## Troubleshooting

### Common Issues
1. **Virtual environment warnings**: Use `source activate_backend.sh`
2. **MongoDB connection**: Ensure MongoDB is running locally
3. **Meilisearch**: Start with `docker-compose up -d`
4. **DeepSeek API**: Verify API key in `.env` file
5. **Port conflicts**: Default port is 8100, configurable in `.env`

### Debug Logs
- **Backend logs**: `backend/backend.log`
- **Console output**: Real-time logging with `./run.sh`
- **Health endpoint**: `GET /health` for service status

## Contributing

1. Follow the async-first architecture
2. Maintain backward compatibility with legacy endpoints
3. Update tests when adding new features
4. Use UV for dependency management
5. Follow the existing code structure and naming conventions