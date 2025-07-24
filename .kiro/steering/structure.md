# Project Structure

## Root Directory
```
├── backend/                 # Python backend application
├── web/                     # Nginx configuration
├── scripts/                 # Utility scripts
├── .kiro/                   # Kiro IDE configuration
├── docker-compose.yml       # Meilisearch service
└── setup_venvs.sh          # Environment setup script
```

## Backend Architecture (`backend/`)

### Application Structure
```
backend/
├── app/                     # Main application package
│   ├── handler/            # HTTP request handlers (controllers)
│   ├── service/            # Business logic services
│   ├── research/           # AI research and web search modules
│   └── tornado_main.py     # Application entry point
├── templates/              # HTML templates
├── .env                    # Environment configuration
├── pyproject.toml          # Modern Python project config
├── requirements.txt        # Dependencies
├── run.sh                  # Development server script
└── wsgi.py                 # Production WSGI entry point
```

### Handler Layer (`app/handler/`)
- **Purpose**: HTTP request handling and response formatting
- **Pattern**: One handler per major feature area
- **Key Files**:
  - `chat_handler.py` - Chat messaging (streaming + legacy)
  - `auth_handler.py` - Authentication and OAuth
  - `search_handler.py` - Content search
  - `deep_search_handler.py` - Advanced AI research
  - `admin_handler.py` - Admin dashboard
  - `main_handler.py` - Main page and 404 handling

### Service Layer (`app/service/`)
- **Purpose**: Business logic and external API integration
- **Pattern**: Async-first design with proper error handling
- **Key Files**:
  - `deepseek_service.py` - AI API integration with streaming
  - `mongodb_service.py` - Database operations
  - `search_service.py` - Meilisearch integration
  - `message_formatter.py` - Response formatting utilities

### Research Module (`app/research/`)
- **Purpose**: Advanced AI research and web search capabilities
- **Pattern**: Modular components with clear interfaces
- **Key Files**:
  - `orchestrator.py` - Research workflow coordination
  - `web_search.py` - Web search integration
  - `content_extractor.py` - Web content processing
  - `ai_reasoning.py` - AI reasoning logic

## Code Organization Patterns

### Handler Pattern
```python
class ExampleHandler(tornado.web.RequestHandler):
    async def post(self):
        # 1. Parse and validate request
        # 2. Call service layer
        # 3. Format and return response
```

### Service Pattern
```python
class ExampleService:
    def __init__(self):
        # Initialize with dependencies
    
    async def process_request(self, data):
        # Async business logic
        # Error handling with logging
```

### Configuration
- Environment variables in `.env` file
- Logging configured in `tornado_main.py`
- MongoDB indexes created automatically
- Stream queues managed at application level

## File Naming Conventions
- **Handlers**: `*_handler.py` (e.g., `chat_handler.py`)
- **Services**: `*_service.py` (e.g., `deepseek_service.py`)
- **Tests**: `test_*.py` (e.g., `test_deepseek_api.py`)
- **Scripts**: Descriptive names with `.sh` or `.py` extension
- **Templates**: `.html` files in `templates/` directory

## Import Patterns
- Relative imports within app package: `from app.service.mongodb_service import MongoDBService`
- External libraries imported at top of file
- Environment variables loaded via `python-dotenv`
- Logging configured per module: `logger = logging.getLogger(__name__)`
