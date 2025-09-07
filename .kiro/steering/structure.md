# Project Structure - dxee-china

## Root Directory Organization

```
dxee-china/
├── backend/                 # Python backend application
├── docs/                    # Project documentation
├── .kiro/                   # Kiro spec-driven development files
├── docker-compose.yml       # Meilisearch container configuration
├── README.md               # Main project documentation
├── CLAUDE.md               # Claude Code project instructions
└── setup_venvs.sh          # Environment setup script
```

### Key Root Files
- **docker-compose.yml**: Meilisearch service configuration for content search
- **setup_venvs.sh**: Automated virtual environment setup for development
- **activate_backend.sh**: Environment activation script for backend development
- **CLAUDE.md**: Comprehensive project instructions and development guidelines

## Backend Directory Structure (`/backend/`)

```
backend/
├── app/                     # Main application package
│   ├── handler/            # Request handlers (Tornado handlers)
│   ├── service/            # Business logic services
│   ├── research/           # Advanced research system components
│   └── tornado_main.py     # Application entry point
├── templates/              # HTML templates
├── static/                 # Static assets (CSS, JS, images)
├── tests/                  # Test files (numerous test_*.py files)
├── pyproject.toml          # Modern Python project configuration
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── run.sh                 # Development server startup script
├── test_api.sh            # API testing script
└── wsgi.py                # Production server configuration
```

## Application Package Structure (`/backend/app/`)

### Handler Layer (`/handler/`)
Request handlers implementing Tornado RequestHandler patterns:

- **auth_handler.py**: Authentication (OAuth, login, registration, password reset)
- **chat_handler.py**: Main chat functionality with streaming support
- **deepthink_handler.py**: Deep-thinking AI research mode
- **dual_research_handler.py**: Dual research capabilities
- **deep_search_handler.py**: Enhanced web search functionality
- **search_handler.py**: Content search using Meilisearch
- **health_handler.py**: Service health monitoring
- **main_handler.py**: Main page and static content
- **admin_handler.py**: Administrative functionality

### Service Layer (`/service/`)
Business logic services implementing core functionality:

- **deepseek_service.py**: DeepSeek AI integration with streaming
- **mongodb_service.py**: Database operations and queries
- **search_service.py**: Meilisearch integration
- **enhanced_deepseek_research_service.py**: Advanced research capabilities
- **deepthink_orchestrator.py**: Deep-thinking workflow management
- **jan_reasoning_engine.py**: Jan AI reasoning integration
- **query_generation_engine.py**: Multi-query generation for research
- **answer_synthesizer.py**: Research result synthesis
- **serper_api_client.py**: Serper API integration for web search
- **error_recovery_system.py**: Error handling and recovery
- **message_formatter.py**: Message formatting utilities

### Research System (`/research/`)
Modular research system with clean interfaces:

- **interfaces.py**: Abstract base classes and data structures
- **orchestrator.py**: Main research workflow coordination
- **web_search.py**: Web search functionality
- **content_extractor.py**: Content extraction from web pages
- **ai_reasoning.py**: AI-powered reasoning and analysis
- **cache.py**: Research result caching system
- **metrics.py**: Performance and quality metrics
- **config.py**: Research system configuration
- **migration_guide.py**: Migration utilities for system updates

## Code Organization Patterns

### Async-First Architecture
```python
# Handler pattern
class ChatStreamHandler(BaseHandler):
    async def post(self):
        # Async request handling
        
# Service pattern  
class DeepSeekService:
    async def generate_response(self):
        # Async AI API calls
        
# Database pattern
async def save_message(self, message_data):
    # Async MongoDB operations
```

### Request Flow Pattern
```
HTTP Request → Handler → Service → Database/API → Response Stream
```

### Research System Pattern
```
Query → Orchestrator → Search → Extract → Reason → Synthesize → Stream
```

## File Naming Conventions

### Python Files
- **Handlers**: `*_handler.py` - Tornado request handlers
- **Services**: `*_service.py` - Business logic services  
- **Models**: `*_models.py` - Data models and structures
- **Tests**: `test_*.py` - Test files with descriptive names
- **Configuration**: `config.py`, `settings.py` - Configuration modules

### Research System Files
- **Interfaces**: `interfaces.py` - Abstract base classes
- **Core Components**: `orchestrator.py`, `cache.py`, `metrics.py`
- **Functional Modules**: `web_search.py`, `content_extractor.py`, `ai_reasoning.py`

### Development Scripts
- **Setup**: `setup_*.sh`, `activate_*.sh` - Environment management
- **Testing**: `test_*.sh`, `run_*.sh` - Development utilities
- **Configuration**: `*.example` - Template files

## Import Organization

### Handler Imports
```python
# Standard library
import os, asyncio, logging

# Third-party
import tornado.web
from tornado.concurrent import Future

# Local services
from app.service.deepseek_service import DeepSeekService
from app.service.mongodb_service import MongoDBService
```

### Service Imports  
```python
# Standard library
import asyncio, json, logging
from typing import List, Dict, Any, Optional

# Third-party
from openai import AsyncOpenAI
from motor.motor_asyncio import AsyncIOMotorClient

# Local modules
from .message_formatter import MessageFormatter
```

### Research System Imports
```python
# Standard library
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncGenerator

# Local interfaces
from .interfaces import ResearchQuery, SearchResult
```

## Key Architectural Principles

### Separation of Concerns
- **Handlers**: HTTP request/response and user interaction
- **Services**: Business logic and external API integration
- **Research**: Modular research system with clean interfaces
- **Database**: Data persistence and retrieval operations

### Async/Await Throughout
- All I/O operations use async/await pattern
- Non-blocking database operations with Motor (async MongoDB driver)
- Streaming responses with async generators
- Concurrent processing for research workflows

### Modular Research System
- **Interface-based design**: Abstract base classes for all components
- **Pluggable components**: Easy to swap implementations
- **Configuration-driven**: Centralized configuration management
- **Metrics and monitoring**: Built-in performance tracking

### Error Handling Strategy
```python
# Service level error handling
try:
    result = await external_api_call()
except SpecificAPIError as e:
    logger.error(f"API error: {e}")
    return fallback_response()
except Exception as e:
    logger.exception("Unexpected error")
    raise ServiceError("Internal service error")
```

### Configuration Management
- **Environment variables**: All secrets and configuration in .env
- **Layered configuration**: Default → Environment → Runtime overrides
- **Validation**: Pydantic models for configuration validation
- **Documentation**: Comprehensive .env.example with comments

## Testing Structure

### Test File Organization
```
backend/
├── test_api.sh                    # Integration test script
├── test_deepseek_*.py            # DeepSeek API tests
├── test_unit_*.py                # Unit tests
├── test_task*_*.py               # Feature-specific tests
└── test_*_integration.py         # Integration tests
```

### Test Patterns
- **Unit tests**: Individual component testing
- **Integration tests**: Service interaction testing
- **API tests**: HTTP endpoint testing
- **Performance tests**: Load and benchmark testing

## Documentation Organization

### Documentation Structure
```
docs/
├── deepseek-*.md                 # AI service documentation
├── environment-configuration.md  # Setup guides
└── *.md                         # Feature-specific documentation
```

### Documentation Principles
- **Feature-focused**: Each major feature has dedicated documentation
- **Setup guides**: Clear environment and deployment instructions
- **API documentation**: Endpoint specifications and examples
- **Troubleshooting**: Common issues and solutions

## Development Workflow Structure

### Environment Management
```bash
# Setup (one-time)
./setup_venvs.sh

# Development session
./activate_backend.sh
cd backend && ./run.sh

# Testing
./backend/test_api.sh
python test_specific_feature.py
```

### Code Quality Tools
- **Type hints**: Comprehensive typing throughout codebase
- **Logging**: Structured logging with multiple levels
- **Error tracking**: Comprehensive exception handling
- **Performance monitoring**: Built-in metrics collection

This structure supports rapid development while maintaining clean separation of concerns and enabling easy testing and deployment.