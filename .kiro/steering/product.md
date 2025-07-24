# Product Overview

**dxee-china** is a bidirectional information bridge application that provides:

- Official China data to Asia-Europe users
- Verified global information to mainland China users
- Real-time streaming AI chat with word-by-word responses (like ChatGPT)
- Fast content search and discovery
- Multi-provider authentication (Google, Microsoft, Apple, email/password)
- Persistent chat history and user data storage

## Core Features

- **Streaming Chat**: Server-Sent Events (SSE) for real-time AI responses
- **Search Engine**: Meilisearch-powered content discovery
- **Authentication**: Multi-provider OAuth + traditional email/password
- **Data Persistence**: MongoDB for chat history and user profiles
- **Content Sharing**: Users can share/unshare messages for public discovery

## Target Architecture

- Privacy-focused with local database storage
- Async-first design for high concurrency
- Graceful degradation (streaming falls back to legacy endpoints)
- Production-ready with proper logging and error handling
