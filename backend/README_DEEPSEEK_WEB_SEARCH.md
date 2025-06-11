# DeepSeek Reasoning with Web Search Integration

This implementation demonstrates how to integrate DeepSeek's Reasoning model with Google Web Search API to provide AI-powered research capabilities with real-time web information.

## Features

- **DeepSeek Reasoner Model**: Uses the `deepseek-reasoner` model for enhanced reasoning capabilities
- **Google Custom Search Integration**: Fetches real-time web search results
- **Streaming Support**: Both synchronous and streaming response modes
- **Comprehensive Logging**: Detailed logging for debugging and monitoring
- **Error Handling**: Robust error handling with fallback mechanisms

## Setup Requirements

### 1. DeepSeek API Configuration

You need a DeepSeek API key to use the reasoning model:

```bash
# Add to your .env file
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_API_URL=https://api.deepseek.com  # Optional, defaults to this
```

### 2. Google Custom Search API Setup

To enable web search functionality, you need to set up Google Custom Search API:

#### Step 1: Enable Google Custom Search API
1. Go to [Google Cloud Console](https://console.developers.google.com/)
2. Create a new project or select an existing one
3. Enable the "Custom Search API"
4. Create credentials (API Key)

#### Step 2: Create Custom Search Engine
1. Go to [Google Custom Search Engine](https://cse.google.com/cse/)
2. Click "Add" to create a new search engine
3. Enter `*` in "Sites to search" to search the entire web
4. Create the search engine and note the Search Engine ID

#### Step 3: Configure Environment Variables
```bash
# Add to your .env file
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_CSE_ID=your_custom_search_engine_id_here
```

## Usage

### Running the Test

```bash
cd backend
python test_deepseek_reasoning_with_web_search.py
```

### Example Query

The test includes a specific query about CRM/SFA software in Japan:

```
"Find the CRM/SFA software available in Japan and make the rank by their revenues"
```

### Code Integration

You can integrate this functionality into your own applications:

```python
from test_deepseek_reasoning_with_web_search import DeepSeekReasoningWithWebSearch

# Initialize the service
service = DeepSeekReasoningWithWebSearch()

# Basic usage
result = await service.reasoning_with_web_search(
    query="Your research question here"
)

# Streaming usage
async for chunk in service.streaming_reasoning_with_web_search(
    query="Your research question here"
):
    if chunk['type'] == 'search_results':
        print(f"Found {chunk['search_results_count']} search results")
    elif chunk['type'] == 'reasoning_chunk':
        print("Reasoning:", chunk['reasoning_content'])
    elif chunk['type'] == 'answer_chunk':
        print("Answer:", chunk['content'])
```

## How It Works

### 1. Web Search Phase
- Takes the user query and performs a Google Custom Search
- Retrieves up to 8 relevant web results
- Formats the results with titles, URLs, and snippets

### 2. Reasoning Phase
- Sends the original query along with formatted search results to DeepSeek Reasoner
- Uses an enhanced system prompt that instructs the AI to:
  - Analyze web search results carefully
  - Synthesize information from multiple sources
  - Provide accurate, up-to-date information
  - Cite sources when possible
  - Use thorough reasoning process

### 3. Response Generation
- DeepSeek Reasoner processes the information and generates:
  - **Reasoning Content**: Internal thought process (if available)
  - **Answer Content**: Final comprehensive answer
- Both synchronous and streaming modes are supported

## Response Format

### Synchronous Response
```python
{
    'query': 'Original user query',
    'search_query': 'Query used for web search',
    'search_results': [
        {
            'title': 'Result title',
            'link': 'URL',
            'snippet': 'Description',
            'displayLink': 'Domain'
        }
    ],
    'reasoning_content': 'AI reasoning process',
    'answer_content': 'Final answer',
    'model': 'deepseek-reasoner',
    'timestamp': '2024-01-01T00:00:00.000000',
    'search_results_count': 8
}
```

### Streaming Response Chunks
```python
# Search results chunk
{
    'type': 'search_results',
    'search_results': [...],
    'search_results_count': 8
}

# Reasoning chunk
{
    'type': 'reasoning_chunk',
    'reasoning_content': 'Partial reasoning content'
}

# Answer chunk
{
    'type': 'answer_chunk',
    'content': 'Partial answer content'
}

# Completion chunk
{
    'type': 'complete',
    'reasoning_content': 'Complete reasoning',
    'answer_content': 'Complete answer',
    'search_results': [...],
    'reasoning_length': 1500,
    'answer_length': 800
}
```

## Error Handling

The implementation includes comprehensive error handling:

- **Missing API Keys**: Clear error messages with setup instructions
- **Network Issues**: Timeout handling and retry mechanisms
- **API Errors**: Graceful degradation with informative error messages
- **Search Failures**: Continues with reasoning even if search fails

## Logging

Detailed logging is provided for:
- API requests and responses
- Search operations
- Error conditions
- Performance metrics

Logs are written to `deepseek_web_search_test.log` and also displayed in the console.

## Customization

### Search Parameters
You can customize the search behavior:

```python
# Custom search with specific parameters
search_results = await service.web_search.search(
    query="your search query",
    num_results=5  # Limit results
)
```

### System Prompts
Modify the system message in the `reasoning_with_web_search` method to customize AI behavior:

```python
system_message = """Your custom system prompt here"""
```

### Model Selection
The implementation uses `deepseek-reasoner` by default, but you can modify it to use other models:

```python
response = await self.client.chat.completions.create(
    model="deepseek-chat",  # Alternative model
    # ... other parameters
)
```

## Performance Considerations

- **Search Latency**: Web search adds 1-3 seconds to response time
- **Token Usage**: Including search results increases token consumption
- **Rate Limits**: Both Google Search API and DeepSeek API have rate limits
- **Caching**: Consider implementing search result caching for repeated queries

## Troubleshooting

### Common Issues

1. **"GOOGLE_API_KEY not set"**
   - Ensure you've added the API key to your `.env` file
   - Verify the API key is valid and has Custom Search API enabled

2. **"No search results found"**
   - Check your Custom Search Engine configuration
   - Verify the CSE ID is correct
   - Ensure your search query is not too restrictive

3. **"DeepSeek API timeout"**
   - The reasoning model may take longer with large search results
   - Consider reducing the number of search results
   - Check your network connection

4. **"Authentication error"**
   - Verify your DeepSeek API key is valid
   - Check if your account has access to the reasoner model

### Debug Mode

Enable debug logging by setting the log level:

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## License

This implementation follows the same license as the parent project.
