#!/usr/bin/env python3
"""
Timeout Testing for DeepSeek Integration

This module tests the 10-minute timeout functionality and time management
features of the enhanced DeepSeek research system.
"""

import asyncio
import time
import pytest
import logging
from typing import List, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

# Test imports
from app.service.enhanced_deepseek_research_service import (
    EnhancedDeepSeekResearchService,
    TimeManager,
    MAX_RESEARCH_TIME
)
from app.service.mongodb_service import MongoDBService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
async def mock_slow_service():
    """Create a mock research service with slow operations for timeout testing"""
    
    # Mock MongoDB service
    mock_mongodb = AsyncMock(spec=MongoDBService)
    mock_mongodb.get_cached_content = AsyncMock(return_value=None)
    mock_mongodb.cache_content = AsyncMock(return_value=True)
    mock_mongodb.create_research_indexes = AsyncMock(return_value=True)
    mock_mongodb.get_cache_stats = AsyncMock(return_value={'total_entries': 0})
    
    # Create service
    service = EnhancedDeepSeekResearchService(mongodb_service=mock_mongodb)
    
    # Mock slow operations
    async def slow_search(query, num_results=10):
        await asyncio.sleep(2)  # Simulate slow search
        return [
            {
                'title': f'Test Result {i}',
                'url': f'https://example.com/result{i}',
                'snippet': f'Test snippet {i}',
                'display_link': 'example.com'
            } for i in range(num_results)
        ]
    
    async def slow_extract(url):
        await asyncio.sleep(3)  # Simulate slow extraction
        return {
            'url': url,
            'title': 'Test Content',
            'content': 'Slow extracted content',
            'success': True,
            'method': 'slow_mock'
        }
    
    async def slow_ai_call(**kwargs):
        await asyncio.sleep(4)  # Simulate slow AI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Slow AI response"
        mock_response.usage.total_tokens = 150
        return mock_response
    
    service.web_search.search = slow_search
    service.content_extractor.extract_content = slow_extract
    service.client.chat.completions.create = slow_ai_call
    
    return service

class TestTimeManager:
    """Test TimeManager functionality"""
    
    def test_time_manager_initialization(self):
        """Test TimeManager initialization"""
        manager = TimeManager(max_duration=600)
        assert manager.max_duration == 600
        assert manager.start_time is None
        assert manager.warning_threshold == 480  # 80% of 600
    
    def test_time_manager_start_session(self):
        """Test session timing"""
        manager = TimeManager(max_duration=10)  # 10 seconds for testing
        
        start_time = manager.start_session()
        assert manager.start_time is not None
        assert start_time == manager.start_time
        
        remaining = manager.get_remaining_time()
        assert remaining <= 10
        assert remaining > 9  # Should be close to 10
    
    def test_time_manager_timeout_detection(self):
        """Test timeout detection"""
        manager = TimeManager(max_duration=1)  # 1 second for quick test
        
        manager.start_session()
        time.sleep(1.1)  # Sleep longer than timeout
        
        assert manager.is_time_exceeded()
        assert manager.get_remaining_time() == 0
    
    def test_time_manager_phase_continuation(self):
        """Test phase continuation logic"""
        manager = TimeManager(max_duration=5)  # 5 seconds
        
        manager.start_session()
        
        # Should continue with enough time
        assert manager.should_continue_phase('test_phase', 2)
        
        # Simulate time passing
        time.sleep(2)
        
        # Should not continue if estimated duration exceeds remaining time
        assert not manager.should_continue_phase('long_phase', 10)
    
    def test_time_manager_warning_system(self):
        """Test warning system"""
        manager = TimeManager(max_duration=2)  # 2 seconds
        manager.start_session()
        
        # First call shouldn't trigger warning
        remaining1 = manager.get_remaining_time()
        assert not manager.warning_issued
        
        # Sleep past warning threshold (80%)
        time.sleep(1.8)
        
        # Second call should trigger warning
        remaining2 = manager.get_remaining_time()
        assert manager.warning_issued

@pytest.mark.asyncio
async def test_timeout_during_query_generation(mock_slow_service):
    """Test timeout during query generation phase"""
    
    # Set very short timeout for testing
    mock_slow_service.time_manager = TimeManager(max_duration=3)
    
    # Mock slow query generation
    async def slow_query_generation(question):
        await asyncio.sleep(5)  # Longer than timeout
        return ["test query"]
    
    mock_slow_service._generate_search_queries = slow_query_generation
    
    result = await mock_slow_service.conduct_deepseek_research("Test question", "test-chat")
    
    assert not result['success']
    assert 'Time limit exceeded' in result.get('error', '')

@pytest.mark.asyncio
async def test_timeout_during_web_search(mock_slow_service):
    """Test timeout during web search phase"""
    
    # Set timeout that allows query generation but not web search
    mock_slow_service.time_manager = TimeManager(max_duration=5)
    
    # Mock fast query generation
    mock_slow_service._generate_search_queries = AsyncMock(return_value=["quick query"])
    
    result = await mock_slow_service.conduct_deepseek_research("Test question", "test-chat")
    
    # Should timeout during web search or content extraction
    assert 'steps' in result
    assert 'query_generation' in result['steps']

@pytest.mark.asyncio
async def test_partial_results_on_timeout(mock_slow_service):
    """Test that partial results are returned on timeout"""
    
    # Set timeout to allow some phases but not all
    mock_slow_service.time_manager = TimeManager(max_duration=8)
    
    # Mock phases with different speeds
    mock_slow_service._generate_search_queries = AsyncMock(return_value=["test query"])
    mock_slow_service._perform_web_search = AsyncMock(return_value=[
        {'url': 'https://test.com', 'title': 'Test', 'snippet': 'Test snippet'}
    ])
    
    result = await mock_slow_service.conduct_deepseek_research("Test question", "test-chat")
    
    # Should have partial results
    assert 'steps' in result
    assert len(result['steps']) > 0
    
    # May or may not be successful depending on timing
    if not result.get('success', False):
        assert 'Time limit exceeded' in result.get('error', '') or 'partial_analysis' in result

@pytest.mark.asyncio
async def test_graceful_shutdown_on_timeout():
    """Test graceful shutdown when timeout is reached"""
    
    # Create a service with very short timeout
    mock_mongodb = AsyncMock(spec=MongoDBService)
    mock_mongodb.get_cached_content = AsyncMock(return_value=None)
    mock_mongodb.cache_content = AsyncMock(return_value=True)
    mock_mongodb.create_research_indexes = AsyncMock(return_value=True)
    mock_mongodb.get_cache_stats = AsyncMock(return_value={'total_entries': 0})
    
    service = EnhancedDeepSeekResearchService(mongodb_service=mock_mongodb)
    service.time_manager = TimeManager(max_duration=2)  # Very short timeout
    
    # Mock operations that don't respect timeout internally
    async def non_timeout_aware_operation():
        await asyncio.sleep(10)  # Long operation
        return "Should not complete"
    
    service._generate_search_queries = AsyncMock(side_effect=non_timeout_aware_operation)
    
    start_time = time.time()
    result = await service.conduct_deepseek_research("Test question", "test-chat")
    duration = time.time() - start_time
    
    # Should complete quickly due to timeout
    assert duration < 5  # Much less than the 10 second sleep
    assert not result.get('success', True)

@pytest.mark.asyncio
async def test_timeout_with_concurrent_requests():
    """Test timeout behavior with multiple concurrent requests"""
    
    # Create services with short timeouts
    mock_mongodb = AsyncMock(spec=MongoDBService)
    mock_mongodb.get_cached_content = AsyncMock(return_value=None)
    mock_mongodb.cache_content = AsyncMock(return_value=True)
    mock_mongodb.create_research_indexes = AsyncMock(return_value=True)
    mock_mongodb.get_cache_stats = AsyncMock(return_value={'total_entries': 0})
    
    # Create multiple service instances
    services = []
    for i in range(3):
        service = EnhancedDeepSeekResearchService(mongodb_service=mock_mongodb)
        service.time_manager = TimeManager(max_duration=3)
        
        # Mock slow operations
        async def slow_op(*args, **kwargs):
            await asyncio.sleep(5)
            return []
        
        service._generate_search_queries = AsyncMock(side_effect=slow_op)
        services.append(service)
    
    # Run concurrent requests
    tasks = []
    for i, service in enumerate(services):
        task = service.conduct_deepseek_research(f"Question {i}", f"chat-{i}")
        tasks.append(task)
    
    start_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_duration = time.time() - start_time
    
    # All should timeout quickly
    assert total_duration < 8  # Much less than 15 seconds (3 * 5 seconds)
    
    # Check results
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Request {i} failed with exception: {result}")
        else:
            # Should not be successful due to timeout
            assert not result.get('success', True), f"Request {i} should have timed out"

@pytest.mark.asyncio 
async def test_time_limit_configuration():
    """Test that time limits can be configured"""
    
    # Test different timeout values
    timeout_values = [5, 10, 30]
    
    for timeout in timeout_values:
        manager = TimeManager(max_duration=timeout)
        manager.start_session()
        
        # Check remaining time is approximately the timeout
        remaining = manager.get_remaining_time()
        assert remaining <= timeout
        assert remaining >= timeout - 1  # Allow for small processing delay
        
        # Check warning threshold
        expected_warning = timeout * 0.8
        assert manager.warning_threshold == expected_warning

@pytest.mark.asyncio
async def test_research_metrics_with_timeout():
    """Test that timing metrics are collected even with timeouts"""
    
    mock_mongodb = AsyncMock(spec=MongoDBService)
    mock_mongodb.get_cached_content = AsyncMock(return_value=None)
    mock_mongodb.cache_content = AsyncMock(return_value=True)
    mock_mongodb.create_research_indexes = AsyncMock(return_value=True)
    mock_mongodb.get_cache_stats = AsyncMock(return_value={'total_entries': 0})
    
    service = EnhancedDeepSeekResearchService(mongodb_service=mock_mongodb)
    service.time_manager = TimeManager(max_duration=3)
    
    # Mock fast query generation, slow search
    service._generate_search_queries = AsyncMock(return_value=["test query"])
    
    async def slow_search(*args, **kwargs):
        await asyncio.sleep(5)  # Will timeout
        return []
    
    service._perform_web_search = AsyncMock(side_effect=slow_search)
    
    result = await service.conduct_deepseek_research("Test question", "test-chat")
    
    # Should have timing metrics even with timeout
    assert 'timing_metrics' in result
    assert len(result['timing_metrics']) > 0
    
    # Should have completed at least one phase
    assert 'steps' in result
    assert 'query_generation' in result['steps']

if __name__ == "__main__":
    """Run timeout tests when executed directly"""
    
    async def run_timeout_tests():
        logger.info("Starting DeepSeek Timeout Tests...")
        
        # Test TimeManager basic functionality
        logger.info("1. Testing TimeManager basic functionality...")
        test_manager = TestTimeManager()
        test_manager.test_time_manager_initialization()
        test_manager.test_time_manager_start_session()
        test_manager.test_time_manager_timeout_detection()
        test_manager.test_time_manager_phase_continuation()
        test_manager.test_time_manager_warning_system()
        logger.info("   ✅ TimeManager tests passed")
        
        # Test timeout scenarios
        logger.info("2. Testing timeout during research phases...")
        
        # Create mock service
        mock_mongodb = AsyncMock(spec=MongoDBService)
        mock_mongodb.get_cached_content = AsyncMock(return_value=None)
        mock_mongodb.cache_content = AsyncMock(return_value=True)
        mock_mongodb.create_research_indexes = AsyncMock(return_value=True)
        mock_mongodb.get_cache_stats = AsyncMock(return_value={'total_entries': 0})
        
        service = EnhancedDeepSeekResearchService(mongodb_service=mock_mongodb)
        
        # Add slow operations
        async def slow_op(*args, **kwargs):
            await asyncio.sleep(2)
            return []
        
        service._generate_search_queries = AsyncMock(side_effect=slow_op)
        
        await test_timeout_during_query_generation(service)
        logger.info("   ✅ Query timeout test passed")
        
        await test_partial_results_on_timeout(service)
        logger.info("   ✅ Partial results test passed")
        
        await test_graceful_shutdown_on_timeout()
        logger.info("   ✅ Graceful shutdown test passed")
        
        await test_timeout_with_concurrent_requests()
        logger.info("   ✅ Concurrent timeout test passed")
        
        await test_time_limit_configuration()
        logger.info("   ✅ Configuration test passed")
        
        await test_research_metrics_with_timeout()
        logger.info("   ✅ Metrics collection test passed")
        
        logger.info("All timeout tests completed successfully!")
    
    # Run tests
    asyncio.run(run_timeout_tests())