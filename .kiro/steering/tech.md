# Technology Stack

## Architecture
**Backend-Focused Async Web Application** with sophisticated AI research capabilities and real-time streaming communication.

### High-Level Design
- **Async-First Architecture**: Built on Tornado web framework for high-concurrency handling
- **Microservice-Style Organization**: Handlers, services, and research modules with clear separation of concerns
- **Real-Time Communication**: Server-Sent Events (SSE) for streaming chat responses and research progress
- **Dual Storage Strategy**: MongoDB for complete data + Meilisearch for searchable subset
- **AI Integration**: DeepSeek API with streaming support and advanced research orchestration

## Backend Stack

### Core Framework
- **Tornado 6.4.2**: Primary async web framework for handling concurrent requests and streaming
- **Python >=3.11**: Modern Python with async/await support and improved performance
- **UV Package Manager**: Modern dependency management replacing pip/virtualenv
- **Gunicorn**: Production WSGI server with Tornado workers

### Database & Search
- **MongoDB (Local)**: Primary persistent storage for users, chats, messages, and research cache
  - **Motor 3.3.2**: Async MongoDB driver for non-blocking database operations
  - **PyMongo 4.6.1**: Sync MongoDB driver for administrative operations
- **Meilisearch (Docker)**: Fast full-text search engine for shared content discovery
  - **meilisearch-python-sdk 4.6.0**: Official Python integration
  - **Version**: v1.7 (containerized)

### AI & Research Services
- **OpenAI Client 1.82.0**: DeepSeek API integration with streaming support
- **DeepSeek API**: Primary AI service for chat completions and research analysis
- **Google Custom Search API**: Web search capabilities for research workflows
- **Bright Data API**: Professional content extraction and web scraping
- **Qwen/DashScope API**: Alternative AI service for specialized tasks

### Authentication & Security
- **Multi-Provider OAuth**: Google, Microsoft, Apple, GitHub integration
- **PyJWT 2.10.1**: JSON Web Token handling for session management
- **bcrypt 4.0.1**: Password hashing and verification
- **Python-dotenv 1.0.0**: Environment variable management

### Content Processing
- **BeautifulSoup4 4.12.3**: HTML parsing and content extraction
- **newspaper3k 0.2.8**: News article extraction and processing
- **readability-lxml 0.8.1**: Content readability analysis
- **lxml 5.3.0**: XML/HTML processing with security updates
- **tiktoken >=0.9.0**: Token counting for AI API optimization

### Network & HTTP
- **HTTPX 0.28.1**: Modern async HTTP client for API integrations
- **aiohttp 3.8.5**: Alternative async HTTP client and server components
- **requests 2.32.3**: Sync HTTP client for simple operations

## Development Environment

### Package Management
```bash
# UV-based virtual environment (replaces venv/pip)
uv sync                    # Install dependencies from pyproject.toml
uv run python script.py   # Run scripts in virtual environment
uv add package-name       # Add new dependencies
```

### Required Tools
- **Docker**: For Meilisearch container (`docker-compose up -d`)
- **MongoDB**: Local installation (macOS: `brew services start mongodb-community`)
- **Python >=3.11**: Required for modern async features
- **UV**: Modern Python package manager (replaces pip/virtualenv)

## Common Commands

### Development Setup
```bash
# Initial setup
./setup_venvs.sh           # Create virtual environments
./activate_backend.sh      # Activate backend environment

# Development server
cd backend && ./run.sh     # Start Tornado development server

# Testing
./backend/test_api.sh      # API endpoint testing
uv run python test_deepseek_api.py  # AI integration testing
```

### Service Management
```bash
# Start required services
brew services start mongodb-community  # MongoDB (macOS)
docker-compose up -d                   # Meilisearch container

# Health checks
curl http://localhost:8100/health      # Application health
curl http://localhost:7701/health      # Meilisearch health
```

### Production Deployment
```bash
# Production server
cd backend
uv run gunicorn --bind 0.0.0.0:8100 --workers=1 --worker-class=tornado wsgi:application
```

## Environment Variables

### Core Configuration
```bash
PORT=8100                  # Application port (default: 8100, .env.example: 8888)
DEBUG=True                 # Development mode
MONGODB_URI=mongodb://localhost:27017  # MongoDB connection
MONGODB_DB_NAME=dxeechina  # Database name
```

### Search & AI Services
```bash
MEILISEARCH_HOST=http://localhost:7701  # Search engine endpoint
MEILISEARCH_API_KEY=masterKey           # Search API key

DEEPSEEK_API_KEY=your_key_here          # Primary AI service
DEEPSEEK_API_URL=https://api.deepseek.com
GOOGLE_API_KEY=your_key_here            # Web search API
GOOGLE_CSE_ID=your_cse_id_here          # Custom search engine
BRIGHTDATA_API_KEY=your_key_here        # Content extraction API
```

### Research Configuration
```bash
DEEPSEEK_RESEARCH_TIMEOUT=600           # Research timeout (10 minutes)
CACHE_EXPIRY_DAYS=30                   # MongoDB cache expiry
MAX_CONCURRENT_RESEARCH=3              # Concurrent research sessions
```

### Authentication
```bash
AUTH_SECRET_KEY=your_secret_here        # JWT signing key
GOOGLE_CLIENT_ID=your_id_here          # OAuth credentials
GOOGLE_CLIENT_SECRET=your_secret_here
MICROSOFT_CLIENT_ID=your_id_here
MICROSOFT_CLIENT_SECRET=your_secret_here
APPLE_CLIENT_ID=your_id_here
APPLE_CLIENT_SECRET=your_secret_here
GITHUB_CLIENT_ID=your_id_here
GITHUB_CLIENT_SECRET=your_secret_here
```

## Port Configuration

### Standard Ports
- **8100**: Main application (production default)
- **8888**: Alternative application port (.env.example default)
- **27017**: MongoDB (standard MongoDB port)
- **7701**: Meilisearch (mapped from container port 7700)

### Service Dependencies
- **MongoDB**: Must be running locally on port 27017
- **Meilisearch**: Docker container mapped to host port 7701
- **External APIs**: DeepSeek, Google CSE, Bright Data (outbound HTTPS)

## Performance Characteristics

### Scalability
- **Concurrent Users**: ~2000 users with ~200 chats each
- **Async Processing**: Non-blocking I/O for database and API operations
- **Memory Efficient**: Queue-based streaming with automatic cleanup
- **Cache Strategy**: 30-day content caching with configurable expiry

### Monitoring
- **Health Endpoints**: `/health` for service status checks
- **Logging**: Structured logging with configurable levels (INFO default)
- **Error Handling**: Graceful degradation with retry logic
- **Metrics**: Cache hit rates, response times, research completion rates