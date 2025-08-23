# Serper MCP Integration Requirements Document
## DeepSeek Advanced Web Research with Serper API v4.01

### Document Version
- Version: 1.0
- Date: 2025-01-23
- Author: System Architecture Team

---

## 1. Executive Summary

This document outlines the requirements for `test_deepseek_advanced_web_research4_01.py`, which integrates:
- **DeepSeek LLM** via OpenAI API interface for advanced reasoning and query generation
- **Serper API** for professional web search and content scraping
- **Deep-thinking algorithms** inspired by 'jan' project for intelligent query formulation
- **MCP-server-serper** patterns for structured API interaction

The goal is to create an intelligent research system that leverages DeepSeek's reasoning capabilities to generate thoughtful search queries, execute them via Serper API, and synthesize comprehensive answers.

---

## 2. System Architecture

### 2.1 Core Components

```
┌─────────────────────────────────────────────────────────┐
│                   DeepSeek LLM Engine                    │
│  (OpenAI API Interface - deepseek-chat/deepseek-reasoner)│
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Query Generation & Reasoning                 │
│         (Deep-thinking algorithms from 'jan')            │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                  Serper API Client                       │
│        (Web Search & Content Scraping Service)           │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Result Processing & Synthesis               │
│            (Answer Generation & Refinement)              │
└──────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

1. **User Input** → Research question/topic
2. **DeepSeek Analysis** → Understanding intent and context
3. **Query Generation** → Multi-angle search queries using deep-thinking
4. **Serper API Calls** → Execute searches and scrape content
5. **Result Processing** → Extract and analyze information
6. **Answer Synthesis** → Generate comprehensive response
7. **Iterative Refinement** → Fill knowledge gaps with follow-up queries

---

## 3. Functional Requirements

### 3.1 DeepSeek LLM Integration

#### 3.1.1 API Configuration
- **Endpoint**: Use OpenAI-compatible API endpoint for DeepSeek
- **Models**: Support both `deepseek-chat` and `deepseek-reasoner`
- **Authentication**: API key via environment variable `DEEPSEEK_API_KEY`
- **Parameters**: Temperature, max_tokens, streaming support

#### 3.1.2 Reasoning Capabilities
- **Query Understanding**: Analyze user intent and research scope
- **Context Building**: Maintain conversation context across queries
- **Deep Thinking**: Implement multi-step reasoning for query generation
- **Answer Synthesis**: Generate coherent responses from multiple sources

### 3.2 Serper API Integration

#### 3.2.1 Search Functionality
- **Basic Search**: Standard web search with query string
- **Advanced Operators**: Support for site:, filetype:, intitle:, etc.
- **Regional Settings**: Country (gl) and language (hl) parameters
- **Time Filters**: Recent results (qdr:h, qdr:d, qdr:w, qdr:m, qdr:y)
- **Pagination**: Support for multiple pages of results

#### 3.2.2 Content Scraping
- **URL Scraping**: Extract text and markdown from web pages
- **Metadata Extraction**: JSON-LD, head tags, structured data
- **Error Handling**: Graceful handling of failed scrapes

#### 3.2.3 API Configuration
- **Authentication**: API key via environment variable `SERPER_API_KEY`
- **Rate Limiting**: Respect API rate limits
- **Error Recovery**: Retry logic for transient failures

### 3.3 Deep-Thinking Query Generation (from 'jan')

#### 3.3.1 Multi-Angle Analysis
- **Perspective Exploration**: Generate queries from different viewpoints
- **Temporal Analysis**: Historical, current, and future perspectives
- **Geographic Scope**: Local, regional, global considerations
- **Domain Expertise**: Technical, business, social angles

#### 3.3.2 Query Refinement Process
```python
1. Initial Understanding Phase
   - Parse user question
   - Identify key concepts
   - Determine research scope

2. Query Expansion Phase
   - Generate synonyms and related terms
   - Identify domain-specific terminology
   - Create query variations

3. Strategic Query Building
   - Combine terms strategically
   - Apply search operators
   - Prioritize query order

4. Iterative Refinement
   - Analyze initial results
   - Identify knowledge gaps
   - Generate follow-up queries
```

### 3.4 Result Processing & Synthesis

#### 3.4.1 Content Analysis
- **Relevance Scoring**: Evaluate result relevance to research question
- **Source Credibility**: Assess domain authority and content quality
- **Information Extraction**: Extract key facts, statistics, quotes
- **Deduplication**: Remove redundant information

#### 3.4.2 Progressive Answer Building
- **Initial Hypothesis**: Form preliminary answer from first results
- **Evidence Gathering**: Collect supporting/contradicting evidence
- **Gap Analysis**: Identify missing information
- **Answer Refinement**: Update answer with new findings

---

## 4. Technical Implementation

### 4.1 Class Structure

```python
class DeepSeekResearcher:
    """Main orchestrator for research workflow"""
    - __init__(deepseek_api_key, serper_api_key)
    - async research(question: str) -> ResearchResult
    - async generate_queries(question: str) -> List[str]
    - async synthesize_answer(sources: List[Source]) -> str

class SerperClient:
    """Serper API client for search and scraping"""
    - __init__(api_key: str)
    - async search(query: str, params: dict) -> SearchResult
    - async scrape(url: str) -> ScrapeResult
    - build_advanced_query(params: dict) -> str

class DeepThinkingEngine:
    """Query generation using deep-thinking algorithms"""
    - __init__(llm_client: AsyncOpenAI)
    - async analyze_question(question: str) -> QuestionAnalysis
    - async generate_search_queries(analysis: QuestionAnalysis) -> List[Query]
    - async refine_queries(queries: List[Query], results: List[Result]) -> List[Query]

class AnswerSynthesizer:
    """Progressive answer building and refinement"""
    - __init__(llm_client: AsyncOpenAI)
    - async create_initial_answer(sources: List[Source]) -> Answer
    - async update_answer(current: Answer, new_sources: List[Source]) -> Answer
    - async generate_final_summary(answer: Answer) -> str
```

### 4.2 Key Methods

#### 4.2.1 Deep-Thinking Query Generation
```python
async def generate_thoughtful_queries(question: str) -> List[str]:
    """
    Generate search queries using deep-thinking approach
    
    Steps:
    1. Decompose question into core concepts
    2. Identify implicit assumptions and context
    3. Generate queries for different aspects:
       - Factual/definitional queries
       - Comparative/analytical queries
       - Temporal/trend queries
       - Causal/relationship queries
    4. Apply search operators strategically
    5. Prioritize queries by expected value
    """
```

#### 4.2.2 Serper API Integration
```python
async def execute_serper_search(query: str, **params) -> dict:
    """
    Execute search via Serper API with advanced parameters
    
    Parameters:
    - query: Search query with operators
    - gl: Country code (e.g., 'us')
    - hl: Language code (e.g., 'en')
    - num: Number of results
    - tbs: Time filter
    - Advanced operators via query string
    """
```

#### 4.2.3 Progressive Answer Building
```python
async def build_answer_progressively(question: str, sources: List[Source]) -> Answer:
    """
    Build answer iteratively as new information arrives
    
    Process:
    1. Create initial hypothesis
    2. Extract supporting evidence
    3. Identify contradictions
    4. Synthesize coherent narrative
    5. Add citations and confidence scores
    """
```

---

## 5. Data Models

### 5.1 Core Data Structures

```python
@dataclass
class ResearchQuery:
    text: str
    priority: int
    search_type: str  # 'general', 'academic', 'news', 'technical'
    operators: Dict[str, str]  # site, filetype, etc.
    
@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str
    content: Optional[str]
    domain: str
    relevance_score: float
    
@dataclass
class ResearchSession:
    session_id: str
    question: str
    queries: List[ResearchQuery]
    results: List[SearchResult]
    answer_versions: List[Answer]
    metadata: Dict[str, Any]
    
@dataclass
class Answer:
    version: int
    content: str
    confidence_score: float
    sources: List[str]
    timestamp: datetime
    gaps: List[str]  # Identified knowledge gaps
```

---

## 6. Configuration & Environment

### 6.1 Required Environment Variables
```bash
# DeepSeek API Configuration
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_API_URL=https://api.deepseek.com  # Optional, defaults to this

# Serper API Configuration
SERPER_API_KEY=your_serper_api_key
SERPER_BASE_URL=https://google.serper.dev  # Optional

# Optional MongoDB for caching (from v3.07)
MONGODB_URI=mongodb://localhost:27017  # Optional
```

### 6.2 Configuration Parameters
```python
# Research Configuration
MAX_QUERIES_PER_RESEARCH = 10
MAX_RESULTS_PER_QUERY = 10
MAX_SCRAPE_ATTEMPTS = 3
RESEARCH_TIMEOUT = 600  # 10 minutes

# DeepSeek Configuration
DEEPSEEK_MODEL = "deepseek-chat"  # or "deepseek-reasoner"
DEEPSEEK_TEMPERATURE = 0.7
DEEPSEEK_MAX_TOKENS = 4000

# Serper Configuration
SERPER_DEFAULT_REGION = "us"
SERPER_DEFAULT_LANGUAGE = "en"
```

---

## 7. Error Handling & Logging

### 7.1 Error Handling Strategy
- **API Failures**: Implement exponential backoff retry
- **Rate Limiting**: Queue management and throttling
- **Content Extraction Failures**: Fallback to snippet/cache
- **Token Limits**: Content summarization and chunking

### 7.2 Logging Requirements
- **Debug Level**: Query generation, API calls
- **Info Level**: Research progress, major milestones
- **Warning Level**: Retries, fallbacks, degraded performance
- **Error Level**: Failures, exceptions

---

## 8. Performance Optimization

### 8.1 Caching Strategy
- **Query Cache**: Cache search results by query hash
- **Content Cache**: Store scraped content with TTL
- **Answer Cache**: Save intermediate answer versions

### 8.2 Concurrent Processing
- **Parallel Searches**: Execute multiple queries concurrently
- **Async Scraping**: Batch URL scraping operations
- **Stream Processing**: Process results as they arrive

### 8.3 Resource Management
- **Token Optimization**: Minimize token usage through summarization
- **API Call Efficiency**: Batch operations where possible
- **Memory Management**: Stream large content, cleanup after processing

---

## 9. Testing Requirements

### 9.1 Unit Tests
- Query generation logic
- Serper API client methods
- Answer synthesis algorithms
- Error handling paths

### 9.2 Integration Tests
- End-to-end research workflow
- API interaction scenarios
- Cache functionality
- Timeout and rate limit handling

### 9.3 Test Scenarios
```python
# Example test cases
test_cases = [
    "What are the latest advances in quantum computing?",
    "Compare renewable energy adoption in Europe vs Asia",
    "Explain the impact of AI on healthcare in 2024",
    "What are the top 5 emerging cybersecurity threats?",
]
```

---

## 10. Success Metrics

### 10.1 Quality Metrics
- **Answer Completeness**: Coverage of question aspects
- **Source Diversity**: Variety of credible sources
- **Factual Accuracy**: Correctness of information
- **Coherence**: Logical flow and readability

### 10.2 Performance Metrics
- **Research Duration**: Time to complete research
- **API Efficiency**: Calls per research session
- **Token Usage**: Average tokens per research
- **Cache Hit Rate**: Percentage of cached results used

---

## 11. Future Enhancements

### 11.1 Phase 2 Features
- Real-time streaming of research progress
- Multi-modal search (images, videos)
- Academic paper integration
- Social media sentiment analysis

### 11.2 Phase 3 Features
- Custom knowledge base integration
- Collaborative research sessions
- Research report generation
- API endpoint for external services

---

## 12. Dependencies

### 12.1 Python Packages
```python
# Core Dependencies
openai>=1.0.0  # For DeepSeek API via OpenAI interface
aiohttp>=3.8.0  # Async HTTP client
asyncio  # Async programming
requests>=2.28.0  # HTTP requests

# Data Processing
pydantic>=2.0.0  # Data validation
tiktoken>=0.5.0  # Token counting

# Optional
pymongo>=4.0.0  # MongoDB caching
motor>=3.0.0  # Async MongoDB

# Development
pytest>=7.0.0  # Testing
pytest-asyncio>=0.21.0  # Async testing
python-dotenv>=0.19.0  # Environment management
```

### 12.2 External Services
- DeepSeek API (via OpenAI-compatible endpoint)
- Serper API for web search and scraping
- Optional: MongoDB for caching

---

## 13. Implementation Timeline

### Week 1: Foundation
- Set up project structure
- Implement Serper API client
- Create DeepSeek integration

### Week 2: Core Logic
- Develop deep-thinking query generation
- Implement search execution pipeline
- Build answer synthesis engine

### Week 3: Optimization
- Add caching layer
- Implement progressive answer building
- Optimize performance

### Week 4: Testing & Refinement
- Comprehensive testing
- Performance tuning
- Documentation

---

## Appendix A: Example Usage

```python
# Example usage of the new research system
async def main():
    # Initialize researcher
    researcher = DeepSeekResearcher(
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
        serper_api_key=os.getenv("SERPER_API_KEY")
    )
    
    # Conduct research
    question = "What are the latest breakthroughs in renewable energy storage?"
    result = await researcher.research(question)
    
    # Display results
    print(f"Question: {result.question}")
    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence_score:.2f}")
    print(f"Sources: {len(result.sources)}")
    print(f"Research Duration: {result.duration:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Appendix B: Serper API Advanced Search Operators

| Operator | Description | Example |
|----------|-------------|---------|
| site: | Limit to specific domain | site:github.com |
| filetype: | Specific file types | filetype:pdf |
| intitle: | Word in page title | intitle:tutorial |
| inurl: | Word in URL | inurl:documentation |
| related: | Similar websites | related:stackoverflow.com |
| cache: | Cached version | cache:example.com |
| before: | Date before | before:2024-01-01 |
| after: | Date after | after:2023-01-01 |
| "phrase" | Exact phrase | "machine learning" |
| -word | Exclude term | python -snake |
| OR | Alternative terms | tutorial OR guide |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-01-23 | System | Initial requirements document |

---

*End of Requirements Document*
