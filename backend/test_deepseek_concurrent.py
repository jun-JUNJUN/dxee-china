#!/usr/bin/env python3
"""
Concurrent Sessions Testing for DeepSeek Integration

This module tests concurrent user research sessions, resource management,
and system stability under concurrent load.
"""

import asyncio
import time
import pytest
import logging
import random
from typing import List, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch
from concurrent.futures import ThreadPoolExecutor

# Test imports
from app.service.enhanced_deepseek_research_service import EnhancedDeepSeekResearchService
from app.service.mongodb_service import MongoDBService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
CONCURRENT_USER_COUNT = 5
MAX_CONCURRENT_RESEARCH = 3
TEST_QUESTIONS = [
    "What are the top CRM software companies in Japan?",
    "How is AI transforming healthcare?", 
    "What are the latest renewable energy trends?",
    "Which programming languages are most popular?",
    "What are the benefits of microservices?",
    "How does blockchain technology work?",
    "What is the future of electric vehicles?",
    "How to implement DevOps best practices?",
    "What are the security challenges in cloud computing?",
    "How does machine learning improve business processes?"
]

class ConcurrentTestManager:
    """Manages concurrent testing scenarios"""
    
    def __init__(self):
        self.active_sessions = {}
        self.completed_sessions = []
        self.failed_sessions = []
        self.start_time = None
        self.end_time = None
        
    def start_session(self, session_id: str):
        """Start tracking a session"""
        self.active_sessions[session_id] = {
            'start_time': time.time(),
            'status': 'running'
        }
        
    def complete_session(self, session_id: str, result: Dict[str, Any]):
        """Mark session as completed"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session['end_time'] = time.time()
            session['duration'] = session['end_time'] - session['start_time']
            session['result'] = result
            session['status'] = 'completed'
            
            if result.get('success', False):
                self.completed_sessions.append(session)
            else:
                self.failed_sessions.append(session)
            
            del self.active_sessions[session_id]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get test statistics"""
        total_duration = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        return {
            'total_sessions': len(self.completed_sessions) + len(self.failed_sessions),
            'successful_sessions': len(self.completed_sessions),
            'failed_sessions': len(self.failed_sessions),
            'active_sessions': len(self.active_sessions),
            'success_rate': len(self.completed_sessions) / (len(self.completed_sessions) + len(self.failed_sessions)) if (len(self.completed_sessions) + len(self.failed_sessions)) > 0 else 0,
            'total_duration': total_duration,
            'avg_session_duration': sum(s['duration'] for s in self.completed_sessions) / len(self.completed_sessions) if self.completed_sessions else 0
        }

@pytest.fixture
async def mock_research_service():
    """Create a mock research service for concurrent testing"""
    
    # Mock MongoDB service
    mock_mongodb = AsyncMock(spec=MongoDBService)
    mock_mongodb.get_cached_content = AsyncMock(return_value=None)
    mock_mongodb.cache_content = AsyncMock(return_value=True)
    mock_mongodb.create_research_indexes = AsyncMock(return_value=True)
    mock_mongodb.get_cache_stats = AsyncMock(return_value={'total_entries': 0, 'successful_entries': 0})
    
    # Create service
    service = EnhancedDeepSeekResearchService(mongodb_service=mock_mongodb)
    
    # Mock external APIs with realistic delays
    async def mock_search(query, num_results=10):
        # Add random delay to simulate real API
        await asyncio.sleep(random.uniform(0.5, 2.0))
        return [
            {
                'title': f'Result {i} for {query}',
                'url': f'https://example.com/{query.replace(" ", "-")}-{i}',
                'snippet': f'Test snippet for {query} result {i}',
                'display_link': 'example.com'
            } for i in range(min(num_results, 5))
        ]
    
    async def mock_extract(url):
        # Add random delay to simulate content extraction
        await asyncio.sleep(random.uniform(1.0, 3.0))
        return {
            'url': url,
            'title': f'Content for {url}',
            'content': f'Extracted content from {url}' * 10,  # Some realistic content
            'success': True,
            'method': 'mock'
        }
    
    async def mock_ai_call(**kwargs):
        # Add delay to simulate AI processing
        await asyncio.sleep(random.uniform(2.0, 5.0))
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Mock AI response with comprehensive analysis."
        mock_response.usage.total_tokens = random.randint(100, 500)
        return mock_response
    
    service.web_search.search = mock_search
    service.content_extractor.extract_content = mock_extract
    service.client.chat.completions.create = mock_ai_call
    
    return service

@pytest.mark.asyncio
async def test_basic_concurrent_sessions(mock_research_service):
    """Test basic concurrent session handling"""
    
    test_manager = ConcurrentTestManager()
    test_manager.start_time = time.time()
    
    # Create concurrent sessions
    tasks = []
    for i in range(3):  # Start with 3 concurrent sessions
        session_id = f"session-{i}"
        question = TEST_QUESTIONS[i % len(TEST_QUESTIONS)]
        
        async def run_session(sid, q):
            test_manager.start_session(sid)
            try:
                result = await mock_research_service.conduct_deepseek_research(q, sid)
                test_manager.complete_session(sid, result)
                return result
            except Exception as e:
                test_manager.complete_session(sid, {'success': False, 'error': str(e)})
                return {'success': False, 'error': str(e)}
        
        tasks.append(run_session(session_id, question))
    
    # Run concurrent sessions
    results = await asyncio.gather(*tasks, return_exceptions=True)
    test_manager.end_time = time.time()
    
    # Verify results
    stats = test_manager.get_stats()
    
    assert stats['total_sessions'] == 3
    assert stats['success_rate'] >= 0.5  # At least 50% should succeed
    assert stats['total_duration'] > 0
    
    # Log results
    logger.info(f"Basic Concurrent Sessions Test:")
    logger.info(f"  Total Sessions: {stats['total_sessions']}")
    logger.info(f"  Success Rate: {stats['success_rate']:.1%}")
    logger.info(f"  Total Duration: {stats['total_duration']:.2f}s")
    logger.info(f"  Avg Session Duration: {stats['avg_session_duration']:.2f}s")
    
    return stats

@pytest.mark.asyncio
async def test_high_concurrency_load(mock_research_service):
    """Test system behavior under high concurrent load"""
    
    test_manager = ConcurrentTestManager()
    test_manager.start_time = time.time()
    
    # Create many concurrent sessions
    concurrent_count = 10
    tasks = []
    
    for i in range(concurrent_count):
        session_id = f"load-session-{i}"
        question = TEST_QUESTIONS[i % len(TEST_QUESTIONS)]
        
        async def run_load_session(sid, q, delay=0):
            # Stagger session starts to simulate real-world usage
            if delay > 0:
                await asyncio.sleep(delay)
            
            test_manager.start_session(sid)
            try:
                result = await mock_research_service.conduct_deepseek_research(q, sid)
                test_manager.complete_session(sid, result)
                return result
            except Exception as e:
                test_manager.complete_session(sid, {'success': False, 'error': str(e)})
                return {'success': False, 'error': str(e)}
        
        # Stagger starts over 5 seconds
        start_delay = (i * 0.5) % 5
        tasks.append(run_load_session(session_id, question, start_delay))
    
    # Run all sessions
    results = await asyncio.gather(*tasks, return_exceptions=True)
    test_manager.end_time = time.time()
    
    # Analyze results
    stats = test_manager.get_stats()
    
    # Check that system handled the load reasonably
    assert stats['total_sessions'] == concurrent_count
    assert stats['success_rate'] >= 0.3  # At least 30% should succeed under load
    
    # Log results
    logger.info(f"High Concurrency Load Test:")
    logger.info(f"  Concurrent Sessions: {concurrent_count}")
    logger.info(f"  Total Sessions: {stats['total_sessions']}")
    logger.info(f"  Success Rate: {stats['success_rate']:.1%}")
    logger.info(f"  Total Duration: {stats['total_duration']:.2f}s")
    logger.info(f"  Avg Session Duration: {stats['avg_session_duration']:.2f}s")
    
    return stats

@pytest.mark.asyncio
async def test_resource_exhaustion_handling():
    """Test behavior when system resources are exhausted"""
    
    # Create multiple service instances to simulate resource exhaustion
    services = []
    
    for i in range(5):
        mock_mongodb = AsyncMock(spec=MongoDBService)
        mock_mongodb.get_cached_content = AsyncMock(return_value=None)
        mock_mongodb.cache_content = AsyncMock(return_value=True)
        mock_mongodb.create_research_indexes = AsyncMock(return_value=True)
        mock_mongodb.get_cache_stats = AsyncMock(return_value={'total_entries': 0})
        
        service = EnhancedDeepSeekResearchService(mongodb_service=mock_mongodb)
        
        # Mock resource-intensive operations
        async def resource_heavy_op(*args, **kwargs):
            # Simulate heavy resource usage
            await asyncio.sleep(random.uniform(3, 8))
            return []
        
        service._generate_search_queries = AsyncMock(side_effect=resource_heavy_op)
        services.append(service)
    
    # Create tasks that would exhaust resources
    tasks = []
    for i, service in enumerate(services):
        for j in range(3):  # Multiple requests per service
            session_id = f"resource-session-{i}-{j}"
            question = TEST_QUESTIONS[(i + j) % len(TEST_QUESTIONS)]
            
            task = service.conduct_deepseek_research(question, session_id)
            tasks.append(task)
    
    # Run with timeout to prevent hanging
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=30  # 30 second timeout
        )
    except asyncio.TimeoutError:
        results = ['timeout'] * len(tasks)
    
    # Analyze results
    successful = sum(1 for r in results if isinstance(r, dict) and r.get('success', False))
    failed = len(results) - successful
    
    # System should handle resource exhaustion gracefully
    assert failed >= successful, "Some requests should fail under resource exhaustion"
    
    logger.info(f"Resource Exhaustion Test:")
    logger.info(f"  Total Requests: {len(results)}")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Failure Rate: {failed/len(results):.1%}")
    
    return {'total': len(results), 'successful': successful, 'failed': failed}

@pytest.mark.asyncio
async def test_session_isolation():
    """Test that concurrent sessions don't interfere with each other"""
    
    # Create service with shared MongoDB mock
    mock_mongodb = AsyncMock(spec=MongoDBService)
    shared_cache = {}  # Shared cache to test isolation
    
    async def get_cached(url):
        return shared_cache.get(url)
    
    async def set_cache(url, content, keywords):
        shared_cache[url] = content
        return True
    
    mock_mongodb.get_cached_content = get_cached
    mock_mongodb.cache_content = set_cache
    mock_mongodb.create_research_indexes = AsyncMock(return_value=True)
    mock_mongodb.get_cache_stats = AsyncMock(return_value={'total_entries': 0})
    
    service = EnhancedDeepSeekResearchService(mongodb_service=mock_mongodb)
    
    # Mock consistent responses for testing
    service.web_search.search = AsyncMock(return_value=[
        {'title': 'Test', 'url': 'https://test.com', 'snippet': 'Test snippet'}
    ])
    
    service.content_extractor.extract_content = AsyncMock(return_value={
        'url': 'https://test.com',
        'title': 'Test Content',
        'content': 'Test content data',
        'success': True
    })
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.usage.total_tokens = 100
    service.client.chat.completions.create = AsyncMock(return_value=mock_response)
    
    # Run sessions with different questions simultaneously
    session_questions = [
        ("session-1", "What is machine learning?"),
        ("session-2", "How does blockchain work?"),
        ("session-3", "What are microservices?")
    ]
    
    # Track responses per session
    session_responses = {}
    
    async def isolated_session(session_id, question):
        # Set unique response for this session
        mock_response.choices[0].message.content = f"Response for {session_id}: {question}"
        
        result = await service.conduct_deepseek_research(question, session_id)
        session_responses[session_id] = result
        return result
    
    # Run sessions concurrently
    tasks = [isolated_session(sid, q) for sid, q in session_questions]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Verify session isolation
    for i, (session_id, question) in enumerate(session_questions):
        result = session_responses.get(session_id)
        assert result is not None, f"Session {session_id} should have a result"
        assert result['chat_id'] == session_id, f"Session {session_id} result should have correct chat_id"
        assert result['original_question'] == question, f"Session {session_id} should have correct question"
    
    logger.info(f"Session Isolation Test:")
    logger.info(f"  Sessions Run: {len(session_questions)}")
    logger.info(f"  All Sessions Isolated: ✅")
    
    return session_responses

@pytest.mark.asyncio
async def test_cache_contention():
    """Test cache behavior under concurrent access"""
    
    # Shared cache state
    cache_data = {}
    cache_access_count = {'reads': 0, 'writes': 0}
    
    async def concurrent_cache_get(url):
        cache_access_count['reads'] += 1
        await asyncio.sleep(0.01)  # Simulate cache access delay
        return cache_data.get(url)
    
    async def concurrent_cache_set(url, content, keywords):
        cache_access_count['writes'] += 1
        await asyncio.sleep(0.01)  # Simulate cache write delay
        cache_data[url] = content
        return True
    
    # Create multiple services sharing the same cache
    services = []
    for i in range(3):
        mock_mongodb = AsyncMock(spec=MongoDBService)
        mock_mongodb.get_cached_content = concurrent_cache_get
        mock_mongodb.cache_content = concurrent_cache_set
        mock_mongodb.create_research_indexes = AsyncMock(return_value=True)
        mock_mongodb.get_cache_stats = AsyncMock(return_value={'total_entries': 0})
        
        service = EnhancedDeepSeekResearchService(mongodb_service=mock_mongodb)
        services.append(service)
    
    # Mock content extractor for all services
    for service in services:
        service.web_search.search = AsyncMock(return_value=[
            {'title': 'Shared Test', 'url': 'https://shared.com', 'snippet': 'Shared content'}
        ])
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Shared analysis"
        mock_response.usage.total_tokens = 100
        service.client.chat.completions.create = AsyncMock(return_value=mock_response)
    
    # Run concurrent research that will access the same URLs
    tasks = []
    for i, service in enumerate(services):
        task = service.conduct_deepseek_research(
            "What is the shared information?", 
            f"cache-session-{i}"
        )
        tasks.append(task)
    
    start_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    duration = time.time() - start_time
    
    # Verify cache was accessed concurrently
    assert cache_access_count['reads'] > 0, "Cache should have been read"
    assert cache_access_count['writes'] > 0, "Cache should have been written"
    
    # Should handle concurrent access gracefully
    successful_results = [r for r in results if isinstance(r, dict) and r.get('success', False)]
    assert len(successful_results) > 0, "At least some sessions should succeed despite cache contention"
    
    logger.info(f"Cache Contention Test:")
    logger.info(f"  Concurrent Services: {len(services)}")
    logger.info(f"  Cache Reads: {cache_access_count['reads']}")
    logger.info(f"  Cache Writes: {cache_access_count['writes']}")
    logger.info(f"  Successful Sessions: {len(successful_results)}/{len(services)}")
    logger.info(f"  Duration: {duration:.2f}s")
    
    return cache_access_count

if __name__ == "__main__":
    """Run concurrent tests when executed directly"""
    
    async def run_concurrent_tests():
        logger.info("Starting DeepSeek Concurrent Sessions Tests...")
        
        # Create mock service
        mock_mongodb = AsyncMock(spec=MongoDBService)
        mock_mongodb.get_cached_content = AsyncMock(return_value=None)
        mock_mongodb.cache_content = AsyncMock(return_value=True)
        mock_mongodb.create_research_indexes = AsyncMock(return_value=True)
        mock_mongodb.get_cache_stats = AsyncMock(return_value={'total_entries': 0})
        
        service = EnhancedDeepSeekResearchService(mongodb_service=mock_mongodb)
        
        # Add mock implementations
        service.web_search.search = AsyncMock(return_value=[])
        service.content_extractor.extract_content = AsyncMock(return_value={
            'success': True, 'content': 'Test', 'title': 'Test'
        })
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage.total_tokens = 100
        service.client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Run tests
        logger.info("1. Testing basic concurrent sessions...")
        await test_basic_concurrent_sessions(service)
        
        logger.info("2. Testing high concurrency load...")
        await test_high_concurrency_load(service)
        
        logger.info("3. Testing resource exhaustion handling...")
        await test_resource_exhaustion_handling()
        
        logger.info("4. Testing session isolation...")
        await test_session_isolation()
        
        logger.info("5. Testing cache contention...")
        await test_cache_contention()
        
        logger.info("All concurrent session tests completed successfully!")
    
    # Run tests
    asyncio.run(run_concurrent_tests())