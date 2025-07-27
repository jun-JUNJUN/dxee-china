# Claude Code Spec-Driven Development

This project implements Kiro-style Spec-Driven Development for Claude Code using hooks and slash commands.

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

### Project Steering
- Product overview: `.kiro/steering/product.md`
- Technology stack: `.kiro/steering/tech.md`
- Project structure: `.kiro/steering/structure.md`
- Custom steering docs for specialized contexts

### Active Specifications
- **web-research-relevance-enhancement**: Web検索による事実収集と関連性評価(70%以上)に基づく回答集計・要約機能
- **comprehensive-analysis-fix**: Fix comprehensive analysis generation and result summarization in research system v3.07
- Current spec: Check `.kiro/specs/` for active specifications
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
```

## API Endpoints

### Chat System
- `/chat/stream` - **Real-time streaming chat** (Server-Sent Events)
- `/chat/message` - Legacy non-streaming chat
- `/chat/history/{chat_id}` - Chat conversation history
- `/chat/user` - User's chat list
- `/chat/share/{message_id}` - Share/unshare messages
- `/chat/shared` - Browse shared messages

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

## Environment Setup Notes

- **UV package manager**: Modern Python dependency management
- **Virtual environment**: Backend uses `.venv` directory
- **Database**: Local MongoDB for privacy and control
- **Search**: Dockerized Meilisearch for easy deployment
- **Configuration**: Environment variables in `.env` file

## Development Guidelines
- Think in English, generate responses in Japanese

## Spec-Driven Development Workflow

### Phase 0: Steering Generation (Recommended)

#### Kiro Steering (`.kiro/steering/`)
```
/kiro:steering-init          # Generate initial steering documents
/kiro:steering-update        # Update steering after changes
/kiro:steering-custom        # Create custom steering for specialized contexts
```

**Note**: For new features or empty projects, steering is recommended but not required. You can proceed directly to spec-requirements if needed.

### Phase 1: Specification Creation
```
/kiro:spec-init [feature-name]           # Initialize spec structure only
/kiro:spec-requirements [feature-name]   # Generate requirements → Review → Edit if needed
/kiro:spec-design [feature-name]         # Generate technical design → Review → Edit if needed
/kiro:spec-tasks [feature-name]          # Generate implementation tasks → Review → Edit if needed
```

### Phase 2: Progress Tracking
```
/kiro:spec-status [feature-name]         # Check current progress and phases
```

## Spec-Driven Development Workflow

Kiro's spec-driven development follows a strict **3-phase approval workflow**:

### Phase 1: Requirements Generation & Approval
1. **Generate**: `/kiro:spec-requirements [feature-name]` - Generate requirements document
2. **Review**: Human reviews `requirements.md` and edits if needed
3. **Approve**: Manually update `spec.json` to set `"requirements": true`

### Phase 2: Design Generation & Approval
1. **Generate**: `/kiro:spec-design [feature-name]` - Generate technical design (requires requirements approval)
2. **Review**: Human reviews `design.md` and edits if needed
3. **Approve**: Manually update `spec.json` to set `"design": true`

### Phase 3: Tasks Generation & Approval
1. **Generate**: `/kiro:spec-tasks [feature-name]` - Generate implementation tasks (requires design approval)
2. **Review**: Human reviews `tasks.md` and edits if needed
3. **Approve**: Manually update `spec.json` to set `"tasks": true`

### Implementation
Only after all three phases are approved can implementation begin.

**Key Principle**: Each phase requires explicit human approval before proceeding to the next phase, ensuring quality and accuracy throughout the development process.

## Development Rules

1. **Consider steering**: Run `/kiro:steering-init` before major development (optional for new features)
2. **Follow the 3-phase approval workflow**: Requirements → Design → Tasks → Implementation
3. **Manual approval required**: Each phase must be explicitly approved by human review
4. **No skipping phases**: Design requires approved requirements; Tasks require approved design
5. **Update task status**: Mark tasks as completed when working on them
6. **Keep steering current**: Run `/kiro:steering-update` after significant changes
7. **Check spec compliance**: Use `/kiro:spec-status` to verify alignment

## Automation

This project uses Claude Code hooks to:
- Automatically track task progress in tasks.md
- Check spec compliance
- Preserve context during compaction
- Detect steering drift

### Task Progress Tracking

When working on implementation:
1. **Manual tracking**: Update tasks.md checkboxes manually as you complete tasks
2. **Progress monitoring**: Use `/kiro:spec-status` to view current completion status
3. **TodoWrite integration**: Use TodoWrite tool to track active work items
4. **Status visibility**: Checkbox parsing shows completion percentage

## Getting Started

1. Initialize steering documents: `/kiro:steering-init`
2. Create your first spec: `/kiro:spec-init [your-feature-name]`
3. Follow the workflow through requirements, design, and tasks

## Kiro Steering Details

Kiro-style steering provides persistent project knowledge through markdown files:

### Core Steering Documents
- **product.md**: Product overview, features, use cases, value proposition
- **tech.md**: Architecture, tech stack, dev environment, commands, ports
- **structure.md**: Directory organization, code patterns, naming conventions

### Custom Steering
Create specialized steering documents for:
- API standards
- Testing approaches
- Code style guidelines
- Security policies
- Database conventions
- Performance standards
- Deployment workflows

### Inclusion Modes
- **Always Included**: Loaded in every interaction (default)
- **Conditional**: Loaded for specific file patterns (e.g., `"*.test.js"`)
- **Manual**: Loaded on-demand with `#filename` reference
