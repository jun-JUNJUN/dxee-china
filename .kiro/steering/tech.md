# Technology Stack

## Core Framework
- **Python**: >=3.11 (required)
- **Tornado**: 6.4.2 - Async web framework for handling HTTP requests and WebSockets
- **Gunicorn**: WSGI server for production deployment with Tornado workers
- **UV**: Modern Python package manager (preferred over pip)

## Database & Search
- **MongoDB**: Local installation for user data, chat history, and messages
- **Motor**: Async MongoDB driver for Tornado integration
- **Meilisearch**: Docker-based search engine for content discovery
- **meilisearch-python-sdk**: Python client for search integration

## AI & External APIs
- **OpenAI SDK**: Used for DeepSeek API integration with streaming support
- **DeepSeek API**: Primary AI service for chat responses
- **Web Scraping**: BeautifulSoup4, newspaper3k, readability-lxml for content extraction

## Authentication & Security
- **bcrypt**: Password hashing
- **PyJWT**: JWT token handling
- **OAuth Libraries**: google-auth, msal for multi-provider authentication
- **Tornado Sessions**: Cookie-based session management

## Development Tools
- **python-dotenv**: Environment variable management
- **asyncio**: Async programming support
- **logging**: Structured logging to files and console

## Common Commands

### Environment Setup
```bash
# Setup virtual environments and dependencies
./setup_venvs.sh

# Activate backend environment
./activate_backend.sh
```

### Development
```bash
# Start development server
cd backend && ./run.sh

# Test API endpoints
./backend/test_api.sh

# Test DeepSeek integration
python ./backend/test_deepseek_api.py
```

### Services
```bash
# Start MongoDB (install locally first)
# macOS: brew services start mongodb-community
# Ubuntu: sudo systemctl start mongod

# Start Meilisearch (Docker)
docker-compose up -d

# Check service health
curl http://localhost:8100/health
```

### Production
```bash
# Production deployment
cd backend
uv run gunicorn --bind 0.0.0.0:8100 --workers=1 --worker-class=tornado wsgi:application
```

## Configuration
- Environment variables in `backend/.env` (copy from `.env.example`)
- Default ports: Backend (8100), Meilisearch (7701), MongoDB (27017)
- UV package manager for dependency management (pyproject.toml + requirements.txt)
