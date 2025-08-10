# DeepSeek Research API Documentation

## Overview

The DeepSeek Research API provides enhanced web research capabilities through the existing chat streaming infrastructure. This API enables comprehensive research with relevance evaluation, answer aggregation, and real-time progress updates.

## Authentication

All API endpoints require authentication using JWT tokens or session cookies, following the existing authentication system.

## Endpoints

### Enhanced Chat Stream with Research

**Endpoint**: `POST /chat/stream`

**Description**: Primary endpoint for DeepSeek research functionality, providing real-time streaming results via Server-Sent Events.

#### Request Format

```json
{
    "message": "Your research question",
    "chat_id": "unique-chat-identifier",
    "search_mode": "deepseek",
    "chat_history": [
        {
            "role": "user|assistant", 
            "content": "previous message content"
        }
    ]
}
```

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `message` | string | Yes | The research question to investigate |
| `chat_id` | string | Yes | Unique identifier for the chat session |
| `search_mode` | string | Yes | Must be "deepseek" for research mode |
| `chat_history` | array | No | Previous conversation context |

#### Response Format

The endpoint returns Server-Sent Events (SSE) stream with the following event types:

##### Research Step Events

```javascript
data: {
    "type": "research_step",
    "step": 1,
    "description": "Initializing MongoDB cache",
    "progress": 10,
    "timestamp": "2025-01-15T10:30:00Z"
}
```

##### Progress Update Events

```javascript
data: {
    "type": "research_progress", 
    "step": 2,
    "description": "Generating search queries",
    "progress": 20,
    "metrics": {
        "queries_generated": 4,
        "cache_stats": {
            "total_entries": 1250,
            "fresh_entries": 890
        }
    }
}
```

##### Content Extraction Events

```javascript
data: {
    "type": "content_extraction",
    "step": 4,
    "description": "Extracting content from sources",
    "progress": 60,
    "metrics": {
        "urls_processed": 12,
        "successful_extractions": 10,
        "cache_hits": 5,
        "cache_misses": 7
    }
}
```

##### Relevance Evaluation Events

```javascript
data: {
    "type": "relevance_evaluation",
    "step": 5, 
    "description": "Evaluating content relevance",
    "progress": 80,
    "metrics": {
        "contents_evaluated": 15,
        "high_relevance_count": 8,
        "average_relevance_score": 7.4
    }
}
```

##### Final Results Event

```javascript
data: {
    "type": "complete",
    "content": "# Research Results: Your Question\n\n## Analysis\n[Comprehensive analysis content]\n\n## Key Findings\n- Finding 1\n- Finding 2\n\n## Sources\n1. https://source1.com\n2. https://source2.com",
    "metadata": {
        "research_type": "enhanced_deepseek_research",
        "relevance_score": 8.7,
        "confidence": 0.85,
        "sources_used": 8,
        "cache_performance": {
            "hits": 12,
            "misses": 8,
            "hit_rate": 0.6
        },
        "timing_metrics": {
            "total_duration": 45.2,
            "query_generation_duration": 3.1,
            "web_search_duration": 12.4,
            "content_extraction_duration": 18.7,
            "relevance_evaluation_duration": 8.9,
            "analysis_generation_duration": 2.1
        }
    }
}
```

##### Error Events

```javascript
data: {
    "type": "error",
    "error_code": "API_TIMEOUT",
    "message": "Research partially completed due to timeout",
    "partial_results": {
        "steps_completed": 4,
        "content": "Partial analysis based on available data..."
    }
}
```

### Research Configuration Endpoint

**Endpoint**: `GET /api/deepseek/config`

**Description**: Returns current DeepSeek research configuration and status.

#### Response Format

```json
{
    "enabled": true,
    "apis_configured": {
        "google_search": true,
        "bright_data": true,
        "deepseek": true,
        "mongodb": true
    },
    "configuration": {
        "research_timeout": 600,
        "cache_expiry_days": 30,
        "max_concurrent_research": 3,
        "relevance_threshold": 7.0
    },
    "cache_stats": {
        "total_entries": 1250,
        "fresh_entries": 890,
        "successful_entries": 1100,
        "hit_rate": 0.67
    },
    "performance_metrics": {
        "avg_research_duration": 42.3,
        "avg_sources_per_research": 12.4,
        "avg_relevance_score": 7.8
    }
}
```

### Cache Management Endpoint

**Endpoint**: `POST /api/deepseek/cache/clear`

**Description**: Clear expired cache entries or perform cache maintenance.

#### Request Format

```json
{
    "operation": "clear_expired",
    "older_than_days": 30
}
```

#### Response Format

```json
{
    "success": true,
    "message": "Cache cleared successfully",
    "entries_removed": 150,
    "entries_remaining": 1100
}
```

## Error Handling

### HTTP Status Codes

- `200 OK`: Request successful
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Insufficient permissions
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: External APIs unavailable

### Error Response Format

```json
{
    "error": {
        "code": "DEEPSEEK_API_UNAVAILABLE",
        "message": "DeepSeek API is currently unavailable",
        "details": "Connection timeout after 30 seconds",
        "timestamp": "2025-01-15T10:30:00Z",
        "request_id": "req_abc123"
    },
    "fallback": {
        "available": true,
        "mode": "regular_chat",
        "message": "Falling back to regular chat mode"
    }
}
```

## Rate Limits

- **Research Requests**: 10 per hour per user
- **Configuration Requests**: 100 per hour per user  
- **Cache Operations**: 20 per hour per user
- **Concurrent Research Sessions**: 3 per user

## Webhooks (Optional)

### Research Completion Webhook

**URL**: Configurable webhook URL
**Method**: POST
**Trigger**: When research completes or fails

#### Webhook Payload

```json
{
    "event": "research_completed",
    "timestamp": "2025-01-15T10:30:00Z",
    "chat_id": "unique-chat-identifier", 
    "user_id": "user-123",
    "research": {
        "question": "Original research question",
        "success": true,
        "duration": 45.2,
        "sources_analyzed": 12,
        "relevance_score": 8.7,
        "confidence": 0.85
    }
}
```

## Code Examples

### JavaScript/TypeScript

```typescript
// Initialize EventSource for DeepSeek research
const startDeepSeekResearch = (question: string, chatId: string) => {
    const eventSource = new EventSource('/chat/stream', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${authToken}`
        },
        body: JSON.stringify({
            message: question,
            chat_id: chatId,
            search_mode: 'deepseek',
            chat_history: []
        })
    });

    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        switch (data.type) {
            case 'research_step':
                updateProgress(data.progress, data.description);
                break;
                
            case 'research_progress':
                displayMetrics(data.metrics);
                break;
                
            case 'complete':
                displayResults(data.content, data.metadata);
                eventSource.close();
                break;
                
            case 'error':
                handleError(data.error_code, data.message);
                eventSource.close();
                break;
        }
    };

    eventSource.onerror = (error) => {
        console.error('DeepSeek research stream error:', error);
        eventSource.close();
    };

    return eventSource;
};
```

### Python

```python
import asyncio
import aiohttp
import json

async def conduct_deepseek_research(question: str, chat_id: str, auth_token: str):
    """Conduct DeepSeek research using the streaming API"""
    
    payload = {
        'message': question,
        'chat_id': chat_id, 
        'search_mode': 'deepseek',
        'chat_history': []
    }
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {auth_token}'
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            '/chat/stream',
            json=payload,
            headers=headers
        ) as response:
            
            async for line in response.content:
                if line.startswith(b'data: '):
                    data = json.loads(line[6:])
                    
                    if data['type'] == 'research_step':
                        print(f"Step {data['step']}: {data['description']} ({data['progress']}%)")
                    
                    elif data['type'] == 'complete':
                        print("Research completed!")
                        print(data['content'])
                        return data
                    
                    elif data['type'] == 'error':
                        print(f"Research failed: {data['message']}")
                        return None
```

### cURL

```bash
# Start DeepSeek research
curl -X POST http://localhost:8100/chat/stream \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "message": "What are the benefits of renewable energy?",
    "chat_id": "research-session-123",
    "search_mode": "deepseek",
    "chat_history": []
  }'

# Get research configuration
curl -X GET http://localhost:8100/api/deepseek/config \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"

# Clear expired cache
curl -X POST http://localhost:8100/api/deepseek/cache/clear \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "operation": "clear_expired",
    "older_than_days": 30
  }'
```

## Performance Considerations

### Optimization Tips

1. **Cache Utilization**: Recent queries benefit from cache hits, reducing response time by up to 80%
2. **Concurrent Limits**: Respect the 3 concurrent research sessions limit to maintain performance
3. **Question Quality**: Well-formulated questions yield better relevance scores and results
4. **Timeout Management**: Research automatically times out after 10 minutes with partial results

### Expected Response Times

- **Cache Hit Scenarios**: 5-15 seconds
- **Fresh Research**: 30-90 seconds  
- **Complex Queries**: 60-300 seconds (up to 10 minutes)
- **Timeout Cases**: Exactly 600 seconds with partial results

## Monitoring and Logging

### Key Metrics

- Research completion rate
- Average response time  
- Cache hit/miss ratio
- API error rates
- User engagement metrics

### Log Events

- Research session start/completion
- API failures and fallbacks
- Cache operations and performance
- Timeout and resource limit events
- User interaction patterns

## Security Considerations

### Data Privacy

- Research queries are logged for performance analysis only
- Cache entries automatically expire based on configuration
- No sensitive information is stored in plain text
- User data is protected by existing authentication system

### API Security

- All requests require valid authentication
- Rate limiting prevents abuse
- Input validation on all parameters
- Secure handling of external API keys
- CORS policies restrict client access

## Support and Troubleshooting

### Common Issues

1. **"Research timeout"**: Query too complex, try breaking into smaller questions
2. **"API unavailable"**: External services down, retry later or use regular chat
3. **"Low relevance results"**: Rephrase question for better search terms
4. **"Cache errors"**: MongoDB connection issues, check database connectivity

### Debug Information

Include the following in support requests:
- Request timestamp
- Chat ID
- Error codes returned
- Browser/client information
- Network connectivity details