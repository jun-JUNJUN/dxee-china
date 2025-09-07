# Technology Stack - dxee-china

## Architecture Overview

**dxee-china** implements a modern async-first architecture with real-time streaming capabilities, designed for high-performance AI-powered research and cross-cultural information exchange.

### High-Level Architecture
- **Backend**: Python async web service with Tornado framework
- **Database**: Local MongoDB for privacy-focused data persistence
- **Search Engine**: Dockerized Meilisearch for fast content discovery
- **AI Integration**: Multiple AI providers with streaming support
- **Authentication**: Multi-provider OAuth with JWT sessions
- **Deployment**: Production-ready with Gunicorn + Nginx

## Backend Technology Stack

### Core Framework
- **tornado**: 6.4.2 - Async web framework and HTTP server
- **gunicorn**: 23.0.0 - WSGI server for production deployment
- **motor**: 3.3.2 - Async MongoDB driver for non-blocking database operations
- **pymongo**: 4.6.1 - Synchronous MongoDB driver for compatibility

### AI and Machine Learning
- **openai**: 1.82.0 - Client for DeepSeek API with streaming support
- **pydantic**: 2.11.5 - Data validation and serialization for AI models
- **tiktoken**: >=0.9.0 - Token counting and text processing utilities

### Web Research and Content Processing
- **aiohttp**: 3.8.5 - Async HTTP client for web requests
- **httpx**: 0.28.1 - Modern HTTP client with HTTP/2 support
- **beautifulsoup4**: 4.12.3 - HTML parsing and content extraction
- **newspaper3k**: 0.2.8 - Article extraction and text processing
- **readability-lxml**: 0.8.1 - Content readability analysis
- **lxml**: 5.3.0 - Fast XML and HTML processing

### Search and Discovery
- **meilisearch-python-sdk**: 4.6.0 - Fast search engine integration

### Authentication and Security
- **bcrypt**: 4.0.1 - Password hashing and verification
- **PyJWT**: 2.10.1 - JWT token generation and validation

### Development and Configuration
- **python-dotenv**: 1.0.0 - Environment variable management
- **aiofiles**: 24.1.0 - Async file operations
- **psutil**: >=7.0.0 - System and process utilities

## Development Environment

### Python Requirements
- **Python Version**: >=3.11 (Modern async/await syntax and performance)
- **Package Manager**: UV for fast dependency resolution and virtual environment management
- **Project Configuration**: pyproject.toml for modern Python packaging

### Development Tools
```bash
# Environment setup
./setup_venvs.sh          # Initialize virtual environment
./activate_backend.sh     # Activate backend environment

# Development server
cd backend && ./run.sh    # Start development server

# Testing
./backend/test_api.sh     # API endpoint testing
python test_deepseek_api.py  # AI service testing
```

## Common Commands

### Development Workflow
```bash
# Backend development
./activate_backend.sh && cd backend/
uv run python script_name.py        # Run Python scripts
uv run gunicorn wsgi:application     # Production server

# Service management
docker-compose up -d                 # Start Meilisearch
mongod --config /usr/local/etc/mongod.conf  # Start MongoDB
```

### API Testing
```bash
# Health check
curl http://localhost:8100/health

# Chat streaming test
curl -X POST http://localhost:8100/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello", "chat_id":"test"}'
```

## Environment Variables

### Core Configuration
```bash
# Server
PORT=8100                    # Application port
DEBUG=True                   # Development mode

# Database
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=dxeechina

# Search Engine  
MEILISEARCH_HOST=http://localhost:7701
MEILISEARCH_API_KEY=masterKey
```

### AI Services Configuration
```bash
# DeepSeek API
DEEPSEEK_API_KEY=your_key_here
DEEPSEEK_API_URL=https://api.deepseek.com

# Research Configuration
DEEPSEEK_RESEARCH_TIMEOUT=600    # 10 minutes
CACHE_EXPIRY_DAYS=30            # MongoDB cache expiry
MAX_CONCURRENT_RESEARCH=3        # Concurrent sessions

# Qwen API (Alternative AI provider)
DASHSCOPE_API_KEY=your_key_here
DASHSCOPE_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### Web Search APIs
```bash
# Google Custom Search
GOOGLE_API_KEY=your_key_here
GOOGLE_CSE_ID=your_search_engine_id

# Bright Data (Content extraction)
BRIGHTDATA_API_KEY=your_key_here
BRIGHTDATA_API_URL=https://api.brightdata.com/datasets/v3/scrape

# Serper API (Deep-think search)
SERPER_API_KEY=your_key_here
SERPER_SEARCH_URL=https://google.serper.dev/search
SERPER_SCRAPE_URL=https://scrape.serper.dev/scrape
```

### Authentication Configuration
```bash
# JWT
AUTH_SECRET_KEY=your_secret_key_here

# OAuth Providers
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
MICROSOFT_CLIENT_ID=your_microsoft_client_id
MICROSOFT_CLIENT_SECRET=your_microsoft_client_secret
APPLE_CLIENT_ID=your_apple_client_id
APPLE_CLIENT_SECRET=your_apple_client_secret
```

## Port Configuration

### Standard Ports
- **8100**: Main application server (configurable via PORT env var)
- **27017**: MongoDB database (default MongoDB port)
- **7701**: Meilisearch service (configured in docker-compose.yml)

### Development vs Production
```bash
# Development
PORT=8100                    # Default development port

# Production
PORT=80                      # Standard HTTP port
# or configure with reverse proxy (Nginx) on port 80/443
```

## Service Architecture

### Async-First Design
- **Non-blocking I/O**: All database and API operations use async/await
- **Streaming Support**: Real-time response delivery via Server-Sent Events
- **Concurrent Processing**: Multiple research sessions and user requests
- **Resource Management**: Automatic cleanup of stream queues and connections

### Data Flow
```
User Request → Tornado Handler → Service Layer → AI/DB/Search → Async Response → Stream to Client
```

### Research Workflow
1. **Query Generation**: AI-powered multi-query generation
2. **Web Search**: Parallel search execution with multiple APIs
3. **Content Extraction**: High-quality content extraction and caching
4. **Relevance Evaluation**: AI scoring with threshold filtering
5. **Answer Synthesis**: Consolidation and statistical analysis
6. **Streaming Response**: Real-time progress updates via SSE

## Performance Characteristics

### Scalability Targets
- **User Capacity**: ~2000 concurrent users
- **Chat Storage**: ~200 chats per user with full history
- **Research Timeout**: 10 minutes per session with graceful handling
- **Cache Performance**: 30-day retention with intelligent invalidation

### Optimization Features
- **Memory Efficiency**: Queue-based processing with automatic cleanup
- **Database Optimization**: Indexed queries and aggregation pipelines
- **Caching Strategy**: Multi-level caching (MongoDB + in-memory)
- **Error Recovery**: Graceful degradation and retry mechanisms

## Development Guidelines

### Code Quality Standards
- **Async/Await**: Use async/await for all I/O operations
- **Type Hints**: Comprehensive typing with pydantic models
- **Error Handling**: Structured exception handling with logging
- **Testing**: Unit tests and integration tests for critical paths

### Security Best Practices
- **Environment Variables**: Never commit secrets to version control
- **Input Validation**: Pydantic models for request validation
- **Authentication**: Secure JWT implementation with proper expiration
- **Data Privacy**: Local database storage for sensitive information