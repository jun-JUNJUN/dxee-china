# Technology Stack

## Architecture Overview
Modern async-first Python web application with real-time streaming capabilities, built on Tornado framework with MongoDB storage and AI-powered research functionality.

## Core Technologies

### Backend Framework
- **Tornado**: 6.4.2 - Python async web server and framework
- **Python**: >=3.11 with modern async/await patterns
- **UV**: Modern Python package manager for dependency management
- **Gunicorn**: WSGI server for production deployment

### Database & Storage
- **MongoDB**: Primary data storage (users, chats, messages)
- **Motor**: 3.3.2 - Async MongoDB driver for Python
- **PyMongo**: 4.6.1 - MongoDB operations and indexing

### Search Engine
- **Meilisearch**: Docker-containerized search engine
- **meilisearch-python-sdk**: 4.6.0 - Python integration
- **Dual Storage**: Complete data in MongoDB + searchable subset in Meilisearch

### AI & Research
- **DeepSeek API**: AI chat completions with streaming support
- **OpenAI Client**: 1.82.0 - API client for DeepSeek integration
- **Advanced Research System**: Multi-source content extraction and reasoning
- **Content Processing**: BeautifulSoup4, newspaper3k, readability-lxml

### Authentication
- **Multi-Provider OAuth**: Google, Microsoft, Apple
- **JWT Tokens**: PyJWT 2.10.1 for session management
- **BCrypt**: 4.0.1 for password hashing
- **Secure Cookies**: Token-based authentication

## Development Environment

### System Requirements
- **Python**: >=3.11
- **MongoDB**: Local installation
- **Docker**: For Meilisearch container
- **UV Package Manager**: Modern Python dependency management

### Environment Variables
```bash
# Server Configuration
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
MICROSOFT_CLIENT_ID=your_microsoft_client_id
APPLE_CLIENT_ID=your_apple_client_id
```

## Key Development Commands

### Environment Setup
```bash
# Setup virtual environments
./setup_venvs.sh

# Activate backend environment
./activate_backend.sh
# or: source activate_backend.sh
```

### Development Server
```bash
# Start development server
cd backend && ./run.sh

# Start services
docker-compose up -d  # Meilisearch
# MongoDB: brew services start mongodb-community (macOS)
```

### Testing
```bash
# API endpoint testing
./backend/test_api.sh

# DeepSeek integration testing
python ./backend/test_deepseek_api.py

# Health check
curl http://localhost:8100/health
```

### Production Deployment
```bash
# Production server
cd backend
uv run gunicorn --bind 0.0.0.0:8100 --workers=1 --worker-class=tornado wsgi:application
```

## Port Configuration
- **Main Application**: 8100 (configurable via PORT env var)
- **MongoDB**: 27017 (default local installation)
- **Meilisearch**: 7701 (Docker container)

## Streaming Architecture

### Real-time Chat Implementation
- **Protocol**: Server-Sent Events (SSE) over HTTP
- **Endpoint**: `/chat/stream` (primary) + `/chat/message` (fallback)
- **Queue Management**: Per-chat stream queues with automatic cleanup
- **Error Handling**: Graceful degradation to non-streaming mode

### Technical Flow
1. **Frontend**: Fetch API with ReadableStream for SSE processing
2. **Backend**: ChatStreamHandler manages real-time streaming
3. **AI Service**: DeepSeek API called with `stream=True` parameter
4. **Processing**: Async stream queues handle real-time chunks
5. **Storage**: Complete messages saved to MongoDB after streaming

## Performance Characteristics

- **Capacity**: ~2000 users with ~200 chats each
- **Architecture**: Async-first for concurrent request handling
- **Memory**: Efficient streaming with automatic queue cleanup
- **Latency**: Real-time word-by-word responses like ChatGPT
- **Scalability**: Horizontal scaling via multiple Gunicorn workers

## Security & Privacy

- **Local Data**: MongoDB and Meilisearch run locally for privacy
- **Authentication**: Multi-provider OAuth with secure token management
- **API Security**: Environment-based configuration for sensitive keys
- **HTTPS**: Production deployment with TLS termination
- **Data Isolation**: User-specific data access controls

## Development Patterns

- **Async-first**: All I/O operations use async/await
- **Error Handling**: Comprehensive try/catch with graceful degradation
- **Logging**: Structured logging for debugging and monitoring
- **Testing**: Both automated (scripts) and manual testing workflows
- **Code Organization**: Handler → Service → Database layered architecture