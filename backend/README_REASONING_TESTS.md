# DeepSeek Reasoning & Answer Content Tests

This directory contains test scripts that demonstrate how to capture both **reasoning content** (thinking process) and **answer content** (final response) from DeepSeek's reasoning models.

## ? Purpose

These tests specifically demonstrate the query:
> "How many languages do you say 'hello'? Please show me 'hello' in each language you can."

The tests show how DeepSeek's `deepseek-reasoner` model provides:
1. **Reasoning Content**: The AI's internal thinking process
2. **Answer Content**: The final formatted response to the user

## ? Test Files

### 1. `test_reasoning_and_answer.py`
**Direct API Test** - Uses OpenAI client directly
- Tests streaming responses with `deepseek-reasoner` model
- Captures reasoning and answer content separately
- Includes comparison with `deepseek-chat` model
- Saves results to timestamped files

### 2. `test_service_reasoning.py`
**Service Class Test** - Uses the existing `DeepSeekService` class
- Tests the service's streaming functionality
- Demonstrates how the service handles reasoning content
- Shows integration with the existing codebase
- Compares streaming vs non-streaming methods

### 3. `run_reasoning_tests.py`
**Test Runner** - Convenient way to run all tests
- Runs both test scripts with proper environment checking
- Provides summary of results
- Supports running individual or all tests

## ? Quick Start

### Prerequisites
1. Set up your environment variables in `.env`:
   ```bash
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   DEEPSEEK_API_URL=https://api.deepseek.com
   ```

2. Install dependencies:
   ```bash
   pip install openai python-dotenv requests
   ```

### Running Tests

#### Option 1: Run All Tests (Recommended)
```bash
python run_reasoning_tests.py
```

#### Option 2: Run Individual Tests
```bash
# Direct API test
python test_reasoning_and_answer.py

# Service class test
python test_service_reasoning.py
```

#### Option 3: Run Specific Test Type
```bash
# Only direct API test
python run_reasoning_tests.py --test-type direct

# Only service test
python run_reasoning_tests.py --test-type service
```

## ? Expected Output

### Reasoning Content Example
```
The user is asking about greetings in different languages. Let me think about this systematically:

1. First, I should consider major language families
2. Include both formal and informal greetings
3. Provide accurate translations
4. Count the total number of languages

I can provide greetings from:
- Indo-European languages (English, Spanish, French, German, etc.)
- Sino-Tibetan languages (Mandarin, Cantonese)
- Afroasiatic languages (Arabic, Hebrew)
- And many others...
```

### Answer Content Example
```
I can say "hello" in approximately 25-30 languages. Here are greetings in various languages:

1. English: Hello
2. Spanish: Hola
3. French: Bonjour
4. German: Guten Tag
5. Italian: Ciao
6. Japanese: Ç±ÇÒÇ…ÇøÇÕ (Konnichiwa)
7. Mandarin: ?çD (N? h?o)
8. Arabic: ????? (Marhaba)
...
```

## ? Generated Files

After running tests, you'll find these files:

### Reasoning Content Files
- `reasoning_content_YYYYMMDD_HHMMSS.txt` - Raw reasoning content
- `service_reasoning_YYYYMMDD_HHMMSS.txt` - Service test reasoning

### Answer Content Files
- `answer_content_YYYYMMDD_HHMMSS.txt` - Raw answer content
- `service_answer_YYYYMMDD_HHMMSS.txt` - Service test answer

### Structured Data
- `reasoning_and_answer_YYYYMMDD_HHMMSS.json` - Complete structured data
- `service_test_results_YYYYMMDD_HHMMSS.json` - Service test results

### Comparison Files
- `model_comparison_YYYYMMDD_HHMMSS.txt` - Model comparison results
- `non_streaming_comparison_YYYYMMDD_HHMMSS.txt` - Streaming vs non-streaming

### Log Files
- `reasoning_test.log` - Direct API test logs
- `service_reasoning_test.log` - Service test logs

## ? Key Differences

### Reasoning vs Answer Content

| Aspect | Reasoning Content | Answer Content |
|--------|------------------|----------------|
| **Purpose** | Shows AI's thinking process | Final response to user |
| **Format** | Stream of consciousness | Structured, formatted |
| **Visibility** | Internal process | User-facing content |
| **Model** | Only `deepseek-reasoner` | All models |

### Streaming vs Non-Streaming

| Method | Reasoning Content | Real-time Updates | Use Case |
|--------|------------------|-------------------|----------|
| **Streaming** | ? Available | ? Yes | Interactive chat |
| **Non-Streaming** | ? Not available | ? No | Batch processing |

## ?? Technical Details

### Model Selection
- **`deepseek-reasoner`**: Provides both reasoning and answer content
- **`deepseek-chat`**: Provides only answer content
- **Search Mode**: Use `"deep"` mode to prefer reasoning model

### Streaming Implementation
```python
async for chunk in stream:
    if hasattr(chunk.choices[0].delta, 'reasoning_content'):
        # Handle reasoning content
        reasoning_content += chunk.choices[0].delta.reasoning_content
    elif chunk.choices[0].delta.content:
        # Handle answer content
        answer_content += chunk.choices[0].delta.content
```

### Service Integration
```python
# Use the DeepSeekService with streaming
message_data = {
    'message': query,
    'search_mode': 'deep',  # Prefer deepseek-reasoner
    'streaming': True
}

await service.process_message_stream(message_data, stream_queue)
```

## ? Troubleshooting

### Common Issues

1. **No Reasoning Content**
   - Ensure you're using `deepseek-reasoner` model
   - Check that streaming is enabled
   - Verify API key has access to reasoning models

2. **API Errors**
   - Check `DEEPSEEK_API_KEY` is valid
   - Verify network connectivity
   - Try different models if one fails

3. **Import Errors**
   - Ensure you're in the `backend` directory
   - Check Python path includes the app directory
   - Install required dependencies

### Debug Mode
Add more logging by setting:
```python
logging.getLogger().setLevel(logging.DEBUG)
```

## ? Analysis Tips

1. **Compare Reasoning vs Answer**: Look for how the AI's thinking process differs from the final answer
2. **Language Detection**: Check how many languages are actually provided vs mentioned
3. **Model Differences**: Compare `deepseek-reasoner` vs `deepseek-chat` responses
4. **Streaming Behavior**: Observe how content arrives in chunks

## ? Related Files

- [`deepseek_service.py`](app/service/deepseek_service.py) - Main service implementation
- [`test_deepseek_api.py`](test_deepseek_api.py) - Basic API connectivity tests
- [`.env.example`](.env.example) - Environment variable template

## ? Notes

- Tests use a 60-second timeout for streaming requests
- All content is saved with UTF-8 encoding to handle international characters
- Timestamps are included in all generated files for easy tracking
- Both raw and formatted content are preserved for analysis
