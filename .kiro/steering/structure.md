# Project Structure

## Root Directory Organization

```
dxee-china/
├── backend/                    # Primary application directory
├── docs/                       # Project documentation
├── scripts/                    # Utility scripts
├── web/                        # Nginx configuration
├── .kiro/                      # Kiro spec-driven development
├── docker-compose.yml          # Meilisearch container setup
├── setup_venvs.sh              # Environment setup script
├── activate_backend.sh         # Virtual environment activation
└── README.md                   # Project overview
```

### Key Root Files
- **CLAUDE.md**: Project instructions and development guidelines
- **docker-compose.yml**: Meilisearch v1.7 container configuration
- **setup_venvs.sh**: Automated virtual environment and dependency setup
- **activate_backend.sh**: Backend environment activation helper

## Backend Directory Structure

```
backend/
├── app/
│   ├── handler/                # HTTP request handlers (Tornado)
│   ├── service/                # Business logic services
│   ├── research/               # Advanced web research system
│   └── tornado_main.py         # Application entry point
├── templates/                  # HTML templates
├── pyproject.toml              # Modern Python project configuration
├── requirements.txt            # Python dependencies (legacy)
├── uv.lock                     # UV package manager lock file
├── run.sh                      # Development server script
├── test_api.sh                 # API testing script
├── wsgi.py                     # Production WSGI configuration
└── extensive test files        # Comprehensive testing suite
```

## Subdirectory Structures

### Handler Layer (`backend/app/handler/`)
Request handlers following Tornado patterns:

- **auth_handler.py**: Multi-provider OAuth and email/password authentication
- **chat_handler.py**: Real-time streaming chat with Server-Sent Events
- **search_handler.py**: Content search functionality
- **health_handler.py**: Service health monitoring
- **main_handler.py**: Main application interface
- **admin_handler.py**: Administrative functions
- **deep_search_handler.py**: Advanced search capabilities
- **dual_research_handler.py**: Dual-mode research operations

### Service Layer (`backend/app/service/`)
Business logic services with clear separation:

- **deepseek_service.py**: Core AI chat service with streaming support
- **enhanced_deepseek_service.py**: Advanced AI service with research capabilities
- **enhanced_deepseek_research_service.py**: Sophisticated research orchestration
- **mongodb_service.py**: Database operations and connection management
- **search_service.py**: Meilisearch integration
- **message_formatter.py**: Response formatting and processing
- **dual_research_service.py**: Multi-mode research coordination
- **deep_search_service.py**: Advanced search algorithms

### Research System (`backend/app/research/`)
Comprehensive web research and analysis framework:

- **orchestrator.py**: Main research workflow coordination
- **web_search.py**: Google Custom Search API integration
- **content_extractor.py**: Bright Data API for content extraction
- **ai_reasoning.py**: AI-powered relevance evaluation and analysis
- **cache.py**: MongoDB-based content caching system
- **metrics.py**: Research performance and quality metrics
- **config.py**: Research system configuration
- **interfaces.py**: Type definitions and contracts
- **migration_guide.py**: System upgrade and migration procedures

## Code Organization Patterns

### Handler-Service Pattern
- **Handlers**: HTTP request/response, input validation, error handling
- **Services**: Business logic, external API integration, data processing
- **Clear Separation**: Handlers delegate to services, services handle business logic

### Async-First Design
```python
# All I/O operations use async/await
async def handler_method(self):
    result = await service.async_operation()
    return self.write_json(result)
```

### Error Handling Strategy
- **Graceful Degradation**: Fallback to legacy endpoints if streaming fails
- **Comprehensive Logging**: Structured logging throughout the application
- **User-Friendly Responses**: Clear error messages with appropriate HTTP status codes

## File Naming Conventions

### Python Files
- **snake_case**: All Python files use snake_case naming
- **Service Suffix**: Service files end with `_service.py`
- **Handler Suffix**: Request handlers end with `_handler.py`
- **Test Prefix**: Test files start with `test_`

### Configuration Files
- **pyproject.toml**: Modern Python project configuration (preferred)
- **requirements.txt**: Legacy dependency specification (maintained for compatibility)
- **uv.lock**: UV package manager lock file
- **.env.example**: Environment variable template

### Test Files
Extensive testing infrastructure with descriptive naming:
- **test_deepseek_advanced_web_research3_07.py**: Specific algorithm version testing
- **test_task[N]_[description].py**: Task-specific testing modules
- **test_[component]_[functionality].py**: Component-focused tests

## Import Organization

### Standard Import Order
```python
# Standard library imports
import asyncio
import json
from typing import Dict, List, Optional

# Third-party imports
import tornado.web
from motor.motor_asyncio import AsyncIOMotorClient
from openai import AsyncOpenAI

# Local application imports
from app.service.mongodb_service import MongoDBService
from app.research.orchestrator import ResearchOrchestrator
```

### Service Dependencies
- **Handlers import Services**: Clear dependency direction
- **Services import Research**: Modular research system integration
- **Configuration Centralized**: Environment variables managed centrally

## Key Architectural Principles

### Separation of Concerns
- **Presentation Layer**: Tornado handlers manage HTTP concerns
- **Business Logic Layer**: Services contain domain logic
- **Data Layer**: MongoDB and Meilisearch operations
- **Research Layer**: Specialized web research and AI analysis

### Async-First Architecture
- **Non-blocking I/O**: All database and API operations use async/await
- **Concurrent Processing**: Multiple research queries processed simultaneously
- **Stream Processing**: Real-time response streaming with queues

### Configuration Management
- **Environment-Based**: All configuration via environment variables
- **Development vs Production**: Clear separation of development and production settings
- **Security-Conscious**: Sensitive data (API keys, secrets) never committed to repository

### Testing Strategy
- **Unit Tests**: Individual component testing
- **Integration Tests**: Service interaction testing
- **Performance Tests**: Research workflow performance validation
- **API Tests**: HTTP endpoint testing with `test_api.sh`

### Documentation Organization
- **README Files**: Component-specific documentation in subdirectories
- **Inline Documentation**: Comprehensive docstrings for complex functions
- **Configuration Documentation**: Clear environment variable documentation
- **API Documentation**: Endpoint specifications and examples