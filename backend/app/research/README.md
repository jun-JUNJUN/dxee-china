# Modular Research System

A comprehensive, pluggable research system with MongoDB caching, enhanced web search, and AI reasoning capabilities. This system is designed to be easily updatable and maintainable, with clear separation of concerns and modular architecture.

## 🏗️ Architecture Overview

The research system is built with a modular, plugin-based architecture that separates different concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Research Orchestrator                    │
│                  (Coordinates all components)               │
└─────────────────────┬───────────────────────────────────────┘
                      │
    ┌─────────────────┼─────────────────┐
    │                 │                 │
┌───▼────┐    ┌──────▼──────┐    ┌─────▼─────┐
│ Cache  │    │ Web Search  │    │    AI     │
│Service │    │   Service   │    │Reasoning  │
└────────┘    └─────────────┘    └───────────┘
    │                 │                 │
┌───▼────┐    ┌──────▼──────┐    ┌─────▼─────┐
│MongoDB │    │   Google    │    │ DeepSeek  │
│        │    │   Search    │    │   API     │
└────────┘    └─────────────┘    └───────────┘
```

### Core Components

1. **Interfaces** (`interfaces.py`) - Abstract base classes defining contracts
2. **Configuration** (`config.py`) - Centralized configuration management
3. **Metrics** (`metrics.py`) - Performance and usage tracking
4. **Cache Service** (`cache.py`) - MongoDB-based content caching
5. **Web Search** (`web_search.py`) - Enhanced Google search with filtering
6. **Content Extractor** (`content_extractor.py`) - Multi-method web content extraction
7. **AI Reasoning** (`ai_reasoning.py`) - DeepSeek integration with intelligent model selection
8. **Orchestrator** (`orchestrator.py`) - Main coordination service

## 🚀 Quick Start

### Basic Usage

```python
from backend.app.research import create_research_system, ResearchQuery
from datetime import datetime

# Create the research system
orchestrator, config, metrics = create_research_system()

# Create a research query
query = ResearchQuery(
    question="Find the top CRM software companies in Japan by revenue",
    query_id="research_001",
    timestamp=datetime.utcnow(),
    search_mode="enhanced",
    target_relevance=7,
    max_iterations=3
)

# Conduct research
result = await orchestrator.conduct_research(query)

# Check results
if result.success:
    print(f"Research completed: {result.analysis.analysis_content}")
    print(f"Relevance score: {result.metrics['final_relevance_score']}/10")
else:
    print(f"Research failed: {result.error}")

# Cleanup
await orchestrator.cleanup()
```

### Integration with Existing Backend

```python
from backend.app.service.enhanced_deepseek_service import EnhancedDeepSeekService

# Replace the old DeepSeekService with the enhanced version
input_queue = []
output_queue = []

# Create enhanced service
enhanced_service = EnhancedDeepSeekService(input_queue, output_queue)

# Process messages as before
message_data = {
    'message': 'Your research question here',
    'chat_id': 'chat_001',
    'message_id': 'msg_001',
    'search_mode': 'enhanced'  # or 'standard', 'deep', 'googleweb'
}

result = await enhanced_service.process_message(message_data)
```

## 🔧 Configuration

### Environment Variables

```bash
# DeepSeek API (Required)
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_API_URL=https://api.deepseek.com

# Google Search API (Optional but recommended)
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_CSE_ID=your_custom_search_engine_id_here

# MongoDB (Optional - enables caching)
MONGODB_URI=mongodb://localhost:27017

# Research Settings (Optional)
DEFAULT_SEARCH_MODE=enhanced
DEFAULT_TARGET_RELEVANCE=7
DEFAULT_MAX_ITERATIONS=3
CACHE_ENABLED=true
CACHE_EXPIRY_DAYS=7

# Model Settings (Optional)
PRIMARY_CHAT_MODEL=deepseek-chat
PRIMARY_REASONING_MODEL=deepseek-reasoner
MODEL_SELECTION_STRATEGY=mode_based
```

### Configuration Validation

```python
from backend.app.research import validate_system_setup

validation = validate_system_setup()
if validation['valid']:
    print("✅ System properly configured")
else:
    print(f"❌ Configuration issues: {validation['issues']}")
    print(f"⚠️ Warnings: {validation['warnings']}")
```

## 📊 Features

### Enhanced Research Capabilities

- **Iterative Research**: Automatically refines search queries based on relevance scores
- **Multi-angle Search**: Generates multiple search queries from different perspectives
- **Content Caching**: MongoDB-based caching to avoid duplicate scraping
- **Quality Assessment**: Domain quality scoring and source diversification
- **Progress Tracking**: Real-time progress updates and comprehensive metrics

### Intelligent Model Selection

- **Mode-based Selection**: Automatically chooses the best model for the task
- **Fallback Strategy**: Graceful degradation when primary models fail
- **Retry Logic**: Exponential backoff and model switching on errors

### Performance Monitoring

```python
from backend.app.research import get_metrics_collector

metrics = get_metrics_collector()

# Get performance report
report = metrics.get_performance_report()
print(f"Cache hit rate: {report['summary']['cache_performance']['cache_hit_rate']:.1f}%")
print(f"Average extraction time: {report['summary']['timing_summaries']['content_extraction']['avg_time']:.2f}s")
```

## 🔄 Migration from Old System

### Comparison Testing

```python
from backend.app.research.migration_guide import ResearchSystemMigrator

migrator = ResearchSystemMigrator()

# Compare old vs new system
comparison = await migrator.compare_research_methods(
    "What are the latest AI trends?", 
    search_mode="googleweb"
)

# Get migration report
report = migrator.get_migration_report()
print(f"Recommendation: {report['migration_recommendation']}")
```

### Gradual Migration Strategy

1. **Phase 1**: Deploy new system alongside old system
2. **Phase 2**: Route specific search modes to new system
3. **Phase 3**: Compare performance and reliability
4. **Phase 4**: Gradually increase traffic to new system
5. **Phase 5**: Complete migration and remove old system

## 🎯 Search Modes

### Standard Mode
- Basic AI reasoning without web search for simple questions
- Fast response for knowledge-based queries

### Enhanced Mode
- Full iterative research with MongoDB caching
- Multi-angle search queries
- Quality-based source filtering
- Relevance scoring and gap analysis

### Deep Mode
- Maximum iterations and comprehensive analysis
- Detailed reasoning traces
- Extensive source diversification

### GoogleWeb Mode (Legacy)
- Compatible with existing googleweb mode
- Enhanced with new caching and quality assessment

## 📈 Performance Optimization

### Caching Strategy

```python
# Cache performance metrics
cache_stats = await cache_service.get_cache_stats()
print(f"Total entries: {cache_stats['total_entries']}")
print(f"Fresh entries: {cache_stats['fresh_entries']}")
print(f"Cache hit rate: {cache_stats['cache_hit_rate']:.1f}%")

# Manual cache cleanup
deleted_count = await cache_service.cleanup_expired_content()
print(f"Cleaned up {deleted_count} expired entries")
```

### Content Extraction Optimization

- **Domain Quality Assessment**: Prioritizes high-quality sources
- **Method Selection**: Chooses extraction method based on domain type
- **Batch Processing**: Efficient handling of multiple URLs
- **Rate Limiting**: Respectful scraping with configurable delays

## 🔌 Extensibility

### Adding New Search Services

```python
from backend.app.research.interfaces import IWebSearchService, SearchResult

class CustomSearchService(IWebSearchService):
    async def search(self, query: str, num_results: int = 5, **kwargs) -> List[SearchResult]:
        # Implement your custom search logic
        pass
    
    async def search_with_filters(self, query: str, num_results: int = 5, 
                                 exclude_domains: List[str] = None,
                                 prefer_domains: List[str] = None) -> List[SearchResult]:
        # Implement filtered search
        pass
```

### Adding New Cache Services

```python
from backend.app.research.interfaces import ICacheService, ExtractedContent

class CustomCacheService(ICacheService):
    async def get_cached_content(self, url: str) -> Optional[ExtractedContent]:
        # Implement your caching logic
        pass
    
    async def save_content(self, content: ExtractedContent, keywords: List[str]) -> bool:
        # Implement content saving
        pass
```

### Custom Progress Callbacks

```python
from backend.app.research.interfaces import IProgressCallback

class CustomProgressCallback(IProgressCallback):
    async def on_progress(self, step: str, data: Dict[str, Any]):
        # Send progress to your UI/API
        print(f"Progress: {step} - {data}")
    
    async def on_error(self, step: str, error: str, data: Dict[str, Any] = None):
        # Handle errors
        print(f"Error in {step}: {error}")
    
    async def on_complete(self, result: ResearchResult):
        # Handle completion
        print(f"Research completed: {result.success}")
```

## 🐛 Troubleshooting

### Common Issues

1. **MongoDB Connection Failed**
   ```bash
   # Check MongoDB is running
   sudo systemctl status mongod
   
   # Check connection string
   echo $MONGODB_URI
   ```

2. **DeepSeek API Errors**
   ```bash
   # Verify API key
   echo $DEEPSEEK_API_KEY
   
   # Test API connectivity
   curl -H "Authorization: Bearer $DEEPSEEK_API_KEY" https://api.deepseek.com/v1/models
   ```

3. **Google Search Not Working**
   ```bash
   # Check API credentials
   echo $GOOGLE_API_KEY
   echo $GOOGLE_CSE_ID
   ```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging for research components
logger = logging.getLogger('backend.app.research')
logger.setLevel(logging.DEBUG)
```

### Performance Issues

```python
# Get detailed performance metrics
metrics = get_metrics_collector()
performance_report = metrics.get_performance_report()

# Check for slow operations
for operation, stats in performance_report['summary']['timing_summaries'].items():
    if stats['avg_time'] > 5.0:
        print(f"Slow operation: {operation} - {stats['avg_time']:.2f}s average")
```

## 📚 API Reference

### Core Classes

- [`ResearchQuery`](interfaces.py#L15) - Research query specification
- [`ResearchResult`](interfaces.py#L65) - Complete research results
- [`ExtractedContent`](interfaces.py#L35) - Web content extraction results
- [`AnalysisResult`](interfaces.py#L50) - AI analysis results

### Services

- [`IWebSearchService`](interfaces.py#L80) - Web search interface
- [`IContentExtractor`](interfaces.py#L90) - Content extraction interface
- [`ICacheService`](interfaces.py#L100) - Caching interface
- [`IAIReasoningService`](interfaces.py#L115) - AI reasoning interface
- [`IResearchOrchestrator`](interfaces.py#L135) - Main orchestrator interface

### Factory Functions

- `create_research_system()` - Create complete research system
- `create_cache_service()` - Create cache service
- `create_web_search_service()` - Create web search service
- `create_content_extractor()` - Create content extractor
- `create_ai_reasoning_service()` - Create AI reasoning service

## 🤝 Contributing

### Adding New Features

1. Define interfaces in `interfaces.py`
2. Implement concrete classes
3. Add factory functions
4. Update configuration as needed
5. Add tests and documentation

### Testing

```python
# Run migration comparison tests
python -m backend.app.research.migration_guide

# Test individual components
from backend.app.research import validate_system_setup
validation = validate_system_setup()
```

## 📄 License

This research system is part of the DeepSeek China backend project.
