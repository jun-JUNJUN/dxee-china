# DeepSeek Advanced Web Research with Multi-Step Process

This implementation provides a sophisticated 4-step research workflow that combines DeepSeek's reasoning capabilities with comprehensive web content analysis.

## ? Multi-Step Research Process

### Step 1: Query Optimization
- Uses DeepSeek API to analyze the original question
- Generates optimal search queries for web search
- Returns formatted query: `Query="optimized_search_terms"`

### Step 2: Web Search
- Performs Google Custom Search with optimized query
- Retrieves relevant web pages with titles, URLs, and snippets
- Configurable number of results (default: 5)

### Step 3: Content Extraction
- Extracts full article content from each search result page
- Uses multiple extraction methods for reliability:
  - **Newspaper3k**: Best for news articles and blog posts
  - **Readability + BeautifulSoup**: For general web content
  - **BeautifulSoup**: Fallback method for any HTML content
- Handles various content types and website structures

### Step 4: Relevance Analysis
- Uses DeepSeek Reasoning model to analyze extracted content
- Evaluates relevance of each source to the original question
- Provides relevance scores and detailed analysis
- Identifies discrepancies and synthesizes information

## ?? Setup Requirements

### 1. Install Dependencies

```bash
pip install beautifulsoup4 lxml newspaper3k readability-lxml
```

Or install from the updated requirements.txt:

```bash
pip install -r requirements.txt
```

### 2. API Configuration

Add to your `.env` file:

```bash
# DeepSeek API
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_API_URL=https://api.deepseek.com

# Google Custom Search API
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_CSE_ID=your_custom_search_engine_id_here
```

## ? Usage

### Running the Test

```bash
cd backend
python test_deepseek_advanced_web_research.py
```

### Code Integration

```python
from test_deepseek_advanced_web_research import DeepSeekAdvancedResearchService

# Initialize the service
service = DeepSeekAdvancedResearchService()

# Conduct complete research
results = await service.conduct_advanced_research(
    "Find the CRM/SFA software available in Japan and make the rank by their revenues"
)

# Access individual steps
step1_query = results['steps']['step1']['search_query']
step2_results = results['steps']['step2']['search_results']
step3_content = results['steps']['step3']['extracted_contents']
step4_analysis = results['steps']['step4']['analysis']
```

## ? Response Format

### Complete Research Results

```python
{
    'original_question': 'Your research question',
    'timestamp': '2024-01-01T00:00:00.000000',
    'success': True,
    'steps': {
        'step1': {
            'description': 'Generate optimal search query',
            'search_query': 'optimized search terms',
            'success': True
        },
        'step2': {
            'description': 'Perform web search',
            'search_results': [
                {
                    'title': 'Page title',
                    'link': 'URL',
                    'snippet': 'Description',
                    'displayLink': 'Domain'
                }
            ],
            'results_count': 5,
            'success': True
        },
        'step3': {
            'description': 'Extract content from web pages',
            'extracted_contents': [
                {
                    'url': 'URL',
                    'title': 'Article title',
                    'content': 'Full article text',
                    'method': 'newspaper3k',
                    'word_count': 1500,
                    'success': True,
                    'search_result': {...}
                }
            ],
            'successful_extractions': 4,
            'total_extractions': 5,
            'success': True
        },
        'step4': {
            'description': 'Analyze content relevance',
            'analysis': {
                'original_question': 'Your question',
                'analysis_content': 'Detailed analysis',
                'reasoning_content': 'AI reasoning process',
                'sources_analyzed': 5,
                'successful_extractions': 4,
                'model': 'deepseek-reasoner'
            },
            'success': True
        }
    }
}
```

## ? Content Extraction Methods

### Method 1: Newspaper3k
- **Best for**: News articles, blog posts, structured content
- **Advantages**: Excellent at identifying main article content
- **Limitations**: May not work well with complex layouts

### Method 2: Readability + BeautifulSoup
- **Best for**: General web content, articles with ads/sidebars
- **Advantages**: Good at extracting main content from cluttered pages
- **Limitations**: May miss some content in unusual layouts

### Method 3: BeautifulSoup Fallback
- **Best for**: Any HTML content when other methods fail
- **Advantages**: Most reliable fallback, works with any HTML
- **Limitations**: May include unwanted content (ads, navigation)

## ? Key Features

### Intelligent Query Generation
- Analyzes research questions to create optimal search terms
- Considers industry-specific terminology and location modifiers
- Balances specificity with comprehensiveness

### Robust Content Extraction
- Multiple extraction methods ensure high success rate
- Handles various website structures and content types
- Respects website rate limits with built-in delays

### Advanced Relevance Analysis
- Uses DeepSeek Reasoning model for sophisticated analysis
- Provides relevance scores for each source
- Identifies and analyzes discrepancies between sources
- Synthesizes information across multiple sources

### Comprehensive Error Handling
- Graceful degradation when extraction fails
- Detailed error reporting and logging
- Continues processing even if some sources fail

## ? Performance Considerations

### Processing Time
- **Step 1**: ~2-5 seconds (DeepSeek API call)
- **Step 2**: ~1-3 seconds (Google Search API)
- **Step 3**: ~5-15 seconds (content extraction with delays)
- **Step 4**: ~10-30 seconds (DeepSeek Reasoning analysis)
- **Total**: ~18-53 seconds depending on content complexity

### Rate Limiting
- Built-in 1-second delay between content extractions
- Respects Google Search API rate limits
- DeepSeek API timeout handling

### Resource Usage
- Memory usage scales with content size
- Network bandwidth for downloading web pages
- Consider implementing caching for repeated queries

## ?? Error Handling

### Common Scenarios
1. **Content Extraction Failures**: Uses multiple methods, continues with available content
2. **Network Timeouts**: Configurable timeouts with graceful fallbacks
3. **API Rate Limits**: Built-in delays and retry mechanisms
4. **Invalid URLs**: Skips problematic URLs, continues with others

### Logging
- Comprehensive logging to `deepseek_advanced_research.log`
- Console output for real-time monitoring
- Error details for debugging

## ? Troubleshooting

### Content Extraction Issues
```python
# Check extraction success rate
step3 = results['steps']['step3']
success_rate = step3['successful_extractions'] / step3['total_extractions']
print(f"Extraction success rate: {success_rate:.2%}")
```

### Search Result Quality
```python
# Analyze search results
step2 = results['steps']['step2']
for result in step2['search_results']:
    print(f"Title: {result['title']}")
    print(f"Relevance: {result['snippet']}")
```

### API Issues
- Verify API keys are correctly set in `.env`
- Check API quotas and rate limits
- Monitor log files for detailed error messages

## ? Customization

### Modify Search Parameters
```python
# Custom number of search results
search_results = await service.step2_web_search(query, num_results=10)
```

### Custom Content Extraction
```python
# Add custom extraction logic
class CustomWebContentExtractor(WebContentExtractor):
    async def extract_article_content(self, url: str) -> Dict[str, Any]:
        # Your custom extraction logic here
        pass
```

### Analysis Customization
```python
# Modify the analysis prompt in step4_analyze_relevance method
system_message = """Your custom analysis instructions here"""
```

## ? Dependencies

### Core Libraries
- `openai`: DeepSeek API integration
- `requests`: HTTP requests and web scraping
- `python-dotenv`: Environment variable management

### Content Extraction
- `beautifulsoup4`: HTML parsing
- `lxml`: XML/HTML parser
- `newspaper3k`: Article extraction
- `readability-lxml`: Content extraction

### Optional Enhancements
- `aiohttp`: For async HTTP requests (future enhancement)
- `selenium`: For JavaScript-heavy sites (future enhancement)
- `langchain`: For advanced RAG capabilities (future enhancement)

## ? Future Enhancements

1. **Caching System**: Cache search results and extracted content
2. **Parallel Processing**: Extract content from multiple URLs simultaneously
3. **JavaScript Support**: Use Selenium for dynamic content
4. **Content Summarization**: Summarize long articles before analysis
5. **Source Credibility**: Evaluate source reliability and bias
6. **Multi-language Support**: Handle non-English content
7. **RAG Integration**: Use vector databases for semantic search

## ? License

This implementation follows the same license as the parent project.
