# Claude Code Spec-Driven Development

Kiro-style Spec Driven Development implementation using claude code slash commands, hooks and agents.

## Project Context
# dxee-china Project

A bidirectional information bridge application providing official China data to Asia-Europe users and verified global information to mainland China users.

## Architecture

- **Backend**: Tornado (Python async web framework)
- **Database**: MongoDB (local installation)
- **Search**: Meilisearch (Docker container)
- **AI**: DeepSeek API for chat completions with streaming support
- **Auth**: Multi-provider OAuth + email/password
- **Python**: >=3.11 with UV package manager

### Paths
- Steering: `.kiro/steering/`
- Specs: `.kiro/specs/`
- Commands: `.claude/commands/`

### Steering vs Specification

**Steering** (`.kiro/steering/`) - Guide AI with project-wide rules and context  
**Specs** (`.kiro/specs/`) - Formalize development process for individual features

### Active Specifications
- **web-research-relevance-enhancement**: Web検索による事実収集と関連性評価(70%以上)に基づく回答集計・要約機能
- **comprehensive-analysis-fix**: Fix comprehensive analysis generation and result summarization in research system v3.07
- **deepseek-button-integration**: Integrate test_deepseek_advanced_web_research3_07.py algorithm as DeepSeek button functionality
- **serper-deep-think-integration**: Replicate test_deepseek_advanced_web_search4_01.py algorithm with serper-mcp API and Jan deep-thinking logic as "deep-think" button functionality
- **deepthink-streamlining-and-caching**: Frontend button cleanup (remove Google Deep/DeepSeek buttons) and Deep Think enhancement with MongoDB HTML caching and session resilience
- Check `.kiro/specs/` for active specifications
- Use `/kiro:spec-status [feature-name]` to check progress
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

# DeepSeek Research Feature
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_custom_search_engine_id
BRIGHTDATA_API_KEY=your_bright_data_api_key
DEEPSEEK_RESEARCH_TIMEOUT=600  # 10 minutes
CACHE_EXPIRY_DAYS=30          # MongoDB cache expiry in days
```

## API Endpoints

### Chat System
- `/chat/stream` - **Real-time streaming chat** (Server-Sent Events) with DeepSeek research support
- `/chat/message` - Legacy non-streaming chat
- `/chat/history/{chat_id}` - Chat conversation history
- `/chat/user` - User's chat list
- `/chat/share/{message_id}` - Share/unshare messages
- `/chat/shared` - Browse shared messages

#### DeepSeek Research Mode
The chat system supports enhanced research mode via `search_mode: "deepseek"` parameter:

```json
POST /chat/stream
{
    "message": "user question",
    "chat_id": "uuid",
    "search_mode": "deepseek",
    "chat_history": []
}
```

**Response**: Server-Sent Events stream with research progress:
```javascript
data: {"type": "research_step", "step": 1, "description": "Initializing MongoDB cache", "progress": 10}
data: {"type": "research_step", "step": 2, "description": "Generating search queries", "progress": 20}
data: {"type": "research_step", "step": 3, "description": "Performing web search", "progress": 40}
data: {"type": "research_step", "step": 4, "description": "Extracting content", "progress": 60}
data: {"type": "research_step", "step": 5, "description": "Evaluating relevance", "progress": 80}
data: {"type": "complete", "content": "formatted results", "search_results": [...]}
```

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

# Run Python scripts (ALWAYS use this pattern)
./activate_backend.sh && cd ./backend/ && uv run python script_name.py

# Test DeepSeek Research
./activate_backend.sh && cd ./backend/ && uv run python test_deepseek_performance_benchmark.py
./activate_backend.sh && cd ./backend/ && uv run python test_deepseek_timeout.py
./activate_backend.sh && cd ./backend/ && uv run python test_deepseek_concurrent.py
./activate_backend.sh && cd ./backend/ && uv run python test_deepseek_result_validation.py
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

### Enhanced DeepSeek Research

#### Core Features
- **Advanced web research**: Multi-query generation and comprehensive content analysis
- **Relevance evaluation**: AI-powered scoring system (0-10 scale) with 70% threshold filtering
- **Answer aggregation**: High-relevance content consolidation and deduplication
- **MongoDB caching**: Configurable content caching (default 30 days) for improved performance
- **Real-time progress**: Live streaming updates during 10-step research process
- **Statistical analysis**: Comprehensive summaries with numerical data extraction
- **Source attribution**: Complete provenance tracking with confidence metrics
- **Time management**: 10-minute research sessions with graceful timeout handling
- **Resource optimization**: Token-aware content summarization and parallel processing

#### Research Workflow
1. **Query Generation**: AI-powered generation of 3-4 search queries for comprehensive coverage
2. **Web Search**: Multi-query Google Custom Search API execution with result deduplication
3. **Content Extraction**: Bright Data API for high-quality content extraction with caching
4. **Relevance Evaluation**: AI scoring of content relevance (0-10 scale) with threshold filtering
5. **Answer Aggregation**: Consolidation and ranking of high-relevance answers (≥7.0 score)
6. **Statistical Analysis**: Numerical data extraction and comprehensive summary generation
7. **Result Formatting**: Structured markdown output with confidence metrics and source attribution
8. **Streaming Updates**: Real-time progress via Server-Sent Events throughout the process
9. **Cache Management**: Intelligent content caching with configurable expiry and statistics
10. **Performance Tracking**: Comprehensive metrics collection and timeout management

#### Configuration Options
```bash
# Research timeout (default: 600 seconds)
DEEPSEEK_RESEARCH_TIMEOUT=600

# Cache expiry in days (default: 30 days)
CACHE_EXPIRY_DAYS=30

# Maximum concurrent research sessions (default: 3)
MAX_CONCURRENT_RESEARCH=3
```

#### UI Integration
- **DeepSeek Button**: Toggle button in chat interface for research mode
- **Progress Indicators**: Real-time step-by-step progress display
- **Results Display**: Enhanced formatting with relevance scores and source attribution
- **Cache Metrics**: Cache hit/miss statistics and performance indicators
- **Error Handling**: Graceful degradation with informative error messages

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

- Think in English, generate responses in English
- Write the requirement/design/task document in both English and Japanese translation.

## Environment Setup Notes

- **UV package manager**: Modern Python dependency management
- **Virtual environment**: Backend uses `.venv` directory
- **Database**: Local MongoDB for privacy and control
- **Search**: Dockerized Meilisearch for easy deployment
- **Configuration**: Environment variables in `.env` file



## Workflow

### Phase 0: Steering (Optional)
`/kiro:steering` - Create/update steering documents  
`/kiro:steering-custom` - Create custom steering for specialized contexts

Note: Optional for new features or small additions. You can proceed directly to spec-init.

### Phase 1: Specification Creation
1. `/kiro:spec-init [detailed description]` - Initialize spec with detailed project description
2. `/kiro:spec-requirements [feature]` - Generate requirements document
3. `/kiro:spec-design [feature]` - Interactive: "Have you reviewed requirements.md? [y/N]"
4. `/kiro:spec-tasks [feature]` - Interactive: Confirms both requirements and design review

### Phase 2: Progress Tracking
`/kiro:spec-status [feature]` - Check current progress and phases

## Development Rules
1. **Consider steering**: Run `/kiro:steering` before major development (optional for new features)
2. **Follow 3-phase approval workflow**: Requirements → Design → Tasks → Implementation
3. **Approval required**: Each phase requires human review (interactive prompt or manual)
4. **No skipping phases**: Design requires approved requirements; Tasks require approved design
5. **Update task status**: Mark tasks as completed when working on them
6. **Keep steering current**: Run `/kiro:steering` after significant changes
7. **Check spec compliance**: Use `/kiro:spec-status` to verify alignment

## Steering Configuration

### Current Steering Files
Managed by `/kiro:steering` command. Updates here reflect command changes.

### Active Steering Files
- `product.md`: Always included - Product context and business objectives
- `tech.md`: Always included - Technology stack and architectural decisions
- `structure.md`: Always included - File organization and code patterns

### Custom Steering Files
<!-- Added by /kiro:steering-custom command -->
<!-- Format: 
- `filename.md`: Mode - Pattern(s) - Description
  Mode: Always|Conditional|Manual
  Pattern: File patterns for Conditional mode
-->

### Inclusion Modes
- **Always**: Loaded in every interaction (default)
- **Conditional**: Loaded for specific file patterns (e.g., "*.test.js")
- **Manual**: Reference with `@filename.md` syntax
