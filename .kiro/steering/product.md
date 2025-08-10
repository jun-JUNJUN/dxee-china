# Product Overview

## Product Name
dxee-china (DeepSchina)

## Product Description
A bidirectional information bridge application providing official China data to Asia-Europe users and verified global information to mainland China users. Features real-time streaming AI chat, advanced web research capabilities, search functionality, and multi-provider authentication.

## Core Features

1. **🔄 Real-time Streaming Chat**
   - AI responses stream word-by-word like ChatGPT
   - Server-Sent Events (SSE) implementation with async queues
   - DeepSeek API integration with streaming support
   - Graceful fallback to legacy non-streaming mode

2. **🔍 Advanced Web Research System**
   - Enhanced web research with relevance scoring (70%+ threshold)
   - Multi-source content extraction and aggregation
   - AI-powered reasoning and comprehensive analysis
   - Statistical summary generation and result optimization

3. **🔐 Multi-Provider Authentication**
   - OAuth integration: Google, Microsoft, Apple
   - Email/password registration and login
   - JWT token-based sessions with secure cookie management
   - User profile management

4. **💾 Persistent Data Management**
   - MongoDB for chat history and user data
   - Meilisearch for fast content search
   - Dual storage architecture (complete data + searchable subset)
   - Message sharing and discovery functionality

5. **🌐 Bidirectional Information Bridge**
   - Official China data delivery to Asia-Europe users
   - Verified global information provision to mainland China users
   - Cultural and linguistic content adaptation
   - Cross-regional information validation

6. **⚡ Modern Architecture**
   - Tornado async web framework
   - UV package manager for modern Python development
   - Docker containerization for services
   - Production-ready with Gunicorn deployment

## Target Use Cases

- **Cross-Border Business Intelligence**: Companies needing reliable China market data
- **Academic Research**: Researchers requiring verified information from both regions
- **News and Media**: Journalists seeking authentic cross-regional perspectives
- **Cultural Exchange**: Organizations facilitating Asia-Europe cultural understanding
- **Investment Analysis**: Financial analysts needing comprehensive regional insights

## Key Value Proposition

- **🔗 Bidirectional Information Flow**: Unique positioning as a bridge between East and West
- **✨ Real-time Experience**: ChatGPT-like streaming responses for immediate engagement
- **🎯 High Relevance**: Advanced research system with 70%+ relevance threshold
- **🔒 Trust & Security**: Multi-provider auth with secure data handling
- **📊 Comprehensive Analysis**: AI-powered reasoning with statistical summaries
- **🚀 Performance**: Async-first architecture supporting ~2000 users with ~200 chats each

## Target Users

- **Business Professionals**: Executives and analysts working across Asia-Europe markets
- **Researchers & Academics**: Scholars studying cross-regional topics
- **Media & Journalists**: Content creators needing verified cross-regional information
- **Government & NGOs**: Organizations facilitating international cooperation
- **General Public**: Individuals seeking authentic perspectives from both regions