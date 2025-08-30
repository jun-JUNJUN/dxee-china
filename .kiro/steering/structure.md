# Project Structure & Organization

## Root Directory Structure

```
dxee-china/
├── backend/                 # Python backend application
├── .kiro/                   # Spec-driven development system
├── kiro/                    # Kiro command definitions
├── .claude/                 # Claude Code configuration
├── README.md                # Project documentation
├── CLAUDE.md                # Claude Code instructions
├── docker-compose.yml       # Meilisearch container setup
├── setup_venvs.sh           # Environment setup script
├── activate_backend.sh      # Environment activation
└── .gitignore               # Git ignore patterns
```

## Backend Application Structure

```
backend/
├── app/                     # Main application code
│   ├── __init__.py
│   ├── tornado_main.py      # Application entry point
│   ├── handler/             # Request handlers (MVC Controllers)
│   │   ├── __init__.py
│   │   ├── auth_handler.py      # Authentication (OAuth, login, register)
│   │   ├── chat_handler.py      # Real-time streaming chat
│   │   ├── search_handler.py    # Content search functionality
│   │   ├── health_handler.py    # Health checks and monitoring
│   │   ├── main_handler.py      # Main page and static content
│   │   ├── admin_handler.py     # Administrative functions
│   │   ├── deep_search_handler.py    # Advanced search features
│   │   └── dual_research_handler.py  # Research system endpoints
│   ├── service/             # Business logic layer
│   │   ├── __init__.py
│   │   ├── deepseek_service.py       # AI chat service with streaming
│   │   ├── enhanced_deepseek_service.py  # Enhanced AI capabilities
│   │   ├── mongodb_service.py        # Database operations
│   │   ├── search_service.py         # Meilisearch integration
│   │   ├── deep_search_service.py    # Advanced search logic
│   │   ├── dual_research_service.py  # Research orchestration
│   │   └── message_formatter.py     # Message processing utilities
│   └── research/            # Advanced research system
│       ├── __init__.py
│       ├── interfaces.py        # Type definitions and protocols
│       ├── config.py            # Research system configuration
│       ├── web_search.py        # Web search and crawling
│       ├── content_extractor.py # Content processing and extraction
│       ├── ai_reasoning.py      # AI-powered analysis and reasoning
│       ├── orchestrator.py      # Research workflow coordination
│       ├── cache.py             # Caching and optimization
│       ├── metrics.py           # Performance metrics and monitoring
│       └── migration_guide.py   # System migration utilities
├── templates/               # HTML templates
│   └── index.html           # Main chat interface with streaming
├── static/                  # Static assets (CSS, JS, images)
├── pyproject.toml           # Modern Python project configuration
├── requirements.txt         # Python dependencies
├── run.sh                   # Development server script
├── test_api.sh              # API testing script
├── test_deepseek_advanced_web_research*.py  # Research system evolution (v3.01-v4.01)
├── wsgi.py                  # Production WSGI configuration
├── .env.example             # Environment variables template
└── backend.log              # Application logs
```

## Development System Structure

```
.kiro/
├── steering/                # Project steering documents
│   ├── product.md           # Product overview and features
│   ├── tech.md              # Technology stack and architecture
│   └── structure.md         # This file - code organization
└── specs/                   # Feature specifications
    └── [feature-name]/
        ├── spec.json        # Metadata and approval status
        ├── requirements.md   # Feature requirements
        ├── design.md        # Technical design
        └── tasks.md         # Implementation tasks

.claude/
└── settings.local.json      # Claude Code local settings

kiro/                        # Kiro command definitions
├── spec-init.md
├── spec-requirements.md
├── spec-design.md
├── spec-tasks.md
├── spec-status.md
├── steering-init.md
├── steering-update.md
└── steering-custom.md
```

## Naming Conventions

### Python Code
- **Modules**: Snake_case (e.g., `deepseek_service.py`, `mongodb_service.py`)
- **Classes**: PascalCase (e.g., `ChatHandler`, `DeepSeekService`)
- **Functions/Methods**: Snake_case (e.g., `handle_stream_chat`, `get_user_chats`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `API_BASE_URL`, `DEFAULT_TIMEOUT`)
- **Private Methods**: Leading underscore (e.g., `_validate_token`, `_process_chunk`)

### File Organization
- **Handlers**: `*_handler.py` - HTTP request handlers
- **Services**: `*_service.py` - Business logic implementations
- **Models**: `*_model.py` - Data models and schemas (if used)
- **Utils**: `*_utils.py` - Utility functions and helpers
- **Tests**: `test_*.py` - Test files matching module names

### Database Collections (MongoDB)
- **Users**: `users` - User authentication and profile data
- **Chats**: `chats` - Chat metadata and titles
- **Messages**: `messages` - Individual chat messages
- **Shared**: `shared_messages` - User-shared content for search

### API Endpoints
- **REST Pattern**: `/resource` or `/resource/{id}`
- **Chat System**: `/chat/stream`, `/chat/message`, `/chat/history/{chat_id}`
- **Authentication**: `/auth/login`, `/auth/register`, `/auth/{provider}`
- **Utilities**: `/health`, `/search`

### Environment Variables
- **Pattern**: `CATEGORY_SPECIFIC_NAME`
- **Examples**: `MONGODB_URI`, `DEEPSEEK_API_KEY`, `AUTH_SECRET_KEY`
- **Booleans**: Use `True`/`False` strings

### Configuration Files
- **Python**: `pyproject.toml` (modern), `requirements.txt` (compatibility)
- **Docker**: `docker-compose.yml` for services
- **Environment**: `.env` (local), `.env.example` (template)
- **Scripts**: `.sh` extension for shell scripts

## Code Organization Patterns

### Request Handling Pattern (MVC-style)
```
Request → Handler → Service → Database/External API
       ←         ←         ←
```

1. **Handlers** (`app/handler/`): Receive HTTP requests, validate input, call services
2. **Services** (`app/service/`): Implement business logic, coordinate between data sources
3. **Data Layer**: MongoDB operations, external API calls, caching

### Async Pattern
- **All I/O operations**: Use `async`/`await`
- **Database calls**: Use Motor (async MongoDB driver)
- **HTTP requests**: Use aiohttp or similar async clients
- **Stream processing**: Async generators for real-time data

### Error Handling Pattern
```python
try:
    result = await some_async_operation()
    return result
except SpecificException as e:
    logger.error(f"Specific error: {e}")
    raise HTTPError(400, "User-friendly message")
except Exception as e:
    logger.exception("Unexpected error")
    raise HTTPError(500, "Internal server error")
```

### Streaming Implementation Pattern
```python
async def stream_response(self):
    async for chunk in ai_service.stream_chat(message):
        self.write(f"data: {json.dumps(chunk)}\n\n")
        await self.flush()
```

### Configuration Pattern
- **Environment-based**: Use `os.getenv()` with defaults
- **Centralized**: Configuration constants in service modules
- **Validation**: Check required environment variables on startup

## Architectural Principles

### 1. Async-First Design
- **Non-blocking I/O**: All database and external API calls use async/await
- **Concurrent Processing**: Multiple requests handled simultaneously
- **Stream Processing**: Real-time data streaming with async generators
- **Queue Management**: Async queues for stream processing and cleanup

### 2. Layered Architecture
- **Presentation Layer**: Tornado handlers manage HTTP requests/responses
- **Business Logic**: Service classes implement core functionality
- **Data Layer**: MongoDB for persistence, Meilisearch for search
- **External Integration**: DeepSeek API, OAuth providers

### 3. Modular Design
- **Handler Modules**: Each feature area has dedicated handler
- **Service Modules**: Business logic separated by domain
- **Research System**: Self-contained module for advanced research
- **Utility Modules**: Shared functionality across services

### 4. Streaming Architecture
- **Real-time Responses**: Server-Sent Events for live chat streaming
- **Queue-based Processing**: Per-session queues for stream management
- **Graceful Degradation**: Fallback to non-streaming mode on errors
- **Memory Management**: Automatic cleanup of stream resources

### 5. Research System Evolution
- **Algorithmic Versioning**: Iterative research algorithm development (v3.01-v4.01)
- **MCP Integration**: Model Context Protocol patterns for enhanced reasoning
- **Deep-Thinking Patterns**: Inspired by 'jan' project for multi-perspective analysis
- **API Evolution**: Progressive enhancement from Google CSE to Serper API
- **Testing-Driven**: Test files serve as algorithm implementation and validation

## Development Best Practices

### 1. Code Quality
- **Type Hints**: Use Python type annotations where beneficial
- **Docstrings**: Document complex functions and classes
- **Error Handling**: Comprehensive exception handling with logging
- **Logging**: Structured logging for debugging and monitoring

### 2. Testing Strategy
- **API Testing**: Automated scripts for endpoint validation
- **Integration Testing**: Test external service connections
- **Algorithm Evolution**: Versioned test files for research system development
- **Manual Testing**: User workflow validation for streaming features
- **Health Checks**: Built-in service monitoring endpoints

### 3. Performance Optimization
- **Database Indexing**: Optimize MongoDB queries with proper indexes
- **Caching**: Strategic caching for frequently accessed data
- **Connection Pooling**: Efficient database connection management
- **Memory Management**: Proper cleanup of streaming resources

### 4. Security Practices
- **Environment Variables**: Sensitive configuration kept in .env files
- **Token Validation**: Secure JWT token handling and validation
- **Input Sanitization**: Validate and sanitize all user inputs
- **CORS Configuration**: Proper cross-origin request handling

### 5. Deployment Readiness
- **Production Server**: Gunicorn with Tornado workers
- **Process Management**: Proper signal handling and graceful shutdown
- **Logging Configuration**: File-based logging for production
- **Health Monitoring**: Built-in health check endpoints for load balancers