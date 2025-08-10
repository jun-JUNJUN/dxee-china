#!/usr/bin/env python3
"""
Performance Benchmarking Tests for DeepSeek Integration

This module provides comprehensive performance benchmarking for the
enhanced DeepSeek research functionality, measuring timing, resource usage,
and throughput under various conditions.
"""

import asyncio
import time
import pytest
import logging
from typing import List, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch
import psutil
import os
from datetime import datetime

# Test imports
from app.service.enhanced_deepseek_research_service import (
    EnhancedDeepSeekResearchService,
    TimeManager,
    TokenOptimizer,
    BrightDataContentExtractor,
    RelevanceEvaluator,
    AnswerAggregator,
    SummaryGenerator,
    ResultFormatter
)
from app.service.mongodb_service import MongoDBService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
BENCHMARK_QUESTIONS = [
    "What are the top 5 CRM software companies in Japan?",
    "How has artificial intelligence impacted the healthcare industry?",
    "What are the latest trends in renewable energy technology?",
    "Which programming languages are most popular in 2025?",
    "What are the benefits of microservices architecture?"
]

class PerformanceBenchmark:
    """Performance benchmarking utilities"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.memory_usage = []
        self.cpu_usage = []
        
    def start_monitoring(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        self.memory_usage = []
        self.cpu_usage = []
        
    def record_metrics(self):
        """Record current system metrics"""
        process = psutil.Process()
        self.memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB
        self.cpu_usage.append(process.cpu_percent())
        
    def get_results(self) -> Dict[str, Any]:
        """Get benchmark results"""
        self.end_time = time.time()
        
        return {
            'duration_seconds': self.end_time - self.start_time if self.start_time else 0,
            'peak_memory_mb': max(self.memory_usage) if self.memory_usage else 0,
            'avg_memory_mb': sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0,
            'peak_cpu_percent': max(self.cpu_usage) if self.cpu_usage else 0,
            'avg_cpu_percent': sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0,
            'memory_samples': len(self.memory_usage),
            'cpu_samples': len(self.cpu_usage)
        }

@pytest.fixture
async def mock_research_service():
    """Create a mock research service for testing"""
    
    # Mock MongoDB service
    mock_mongodb = AsyncMock(spec=MongoDBService)
    mock_mongodb.get_cached_content = AsyncMock(return_value=None)
    mock_mongodb.cache_content = AsyncMock(return_value=True)
    mock_mongodb.create_research_indexes = AsyncMock(return_value=True)
    mock_mongodb.get_cache_stats = AsyncMock(return_value={'total_entries': 0, 'successful_entries': 0})
    
    # Create service with mocked dependencies
    service = EnhancedDeepSeekResearchService(mongodb_service=mock_mongodb)
    
    # Mock external APIs
    service.web_search.search = AsyncMock(return_value=[
        {
            'title': f'Test Result {i}',
            'url': f'https://example.com/result{i}',
            'snippet': f'Test content snippet {i}',
            'display_link': 'example.com'
        } for i in range(1, 6)
    ])
    
    service.content_extractor.extract_content = AsyncMock(return_value={
        'url': 'https://example.com/test',
        'title': 'Test Content',
        'content': 'This is test content for benchmarking purposes.',
        'success': True,
        'method': 'mock'
    })
    
    # Mock DeepSeek API calls
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test analysis content"
    mock_response.usage.total_tokens = 100
    service.client.chat.completions.create = AsyncMock(return_value=mock_response)
    
    return service

@pytest.mark.asyncio
async def test_single_research_performance(mock_research_service):
    """Benchmark single research request performance"""
    
    benchmark = PerformanceBenchmark()
    
    # Test question
    question = BENCHMARK_QUESTIONS[0]
    
    # Start monitoring
    benchmark.start_monitoring()
    
    # Monitor during execution
    async def monitor_task():
        while benchmark.start_time and not benchmark.end_time:
            benchmark.record_metrics()
            await asyncio.sleep(0.1)
    
    # Run research and monitoring concurrently
    monitor_task_handle = asyncio.create_task(monitor_task())
    
    try:
        result = await mock_research_service.conduct_deepseek_research(question, "test-chat-id")
    finally:
        monitor_task_handle.cancel()
    
    # Get performance results
    perf_results = benchmark.get_results()
    
    # Assertions
    assert result is not None
    assert perf_results['duration_seconds'] > 0
    assert perf_results['duration_seconds'] < 30  # Should complete quickly with mocks
    assert perf_results['peak_memory_mb'] > 0
    
    # Log results
    logger.info(f"Single Research Performance:")
    logger.info(f"  Duration: {perf_results['duration_seconds']:.2f}s")
    logger.info(f"  Peak Memory: {perf_results['peak_memory_mb']:.2f} MB")
    logger.info(f"  Avg CPU: {perf_results['avg_cpu_percent']:.1f}%")
    
    return perf_results

@pytest.mark.asyncio
async def test_batch_research_performance(mock_research_service):
    """Benchmark batch research performance"""
    
    benchmark = PerformanceBenchmark()
    
    # Start monitoring
    benchmark.start_monitoring()
    
    # Monitor during execution
    async def monitor_task():
        while benchmark.start_time and not benchmark.end_time:
            benchmark.record_metrics()
            await asyncio.sleep(0.1)
    
    monitor_task_handle = asyncio.create_task(monitor_task())
    
    try:
        # Run multiple research requests
        tasks = []
        for i, question in enumerate(BENCHMARK_QUESTIONS):
            task = mock_research_service.conduct_deepseek_research(question, f"test-chat-{i}")
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
    finally:
        monitor_task_handle.cancel()
    
    # Get performance results
    perf_results = benchmark.get_results()
    
    # Count successful results
    successful_results = [r for r in results if not isinstance(r, Exception)]
    
    # Assertions
    assert len(successful_results) == len(BENCHMARK_QUESTIONS)
    assert perf_results['duration_seconds'] > 0
    assert perf_results['peak_memory_mb'] > 0
    
    # Calculate throughput
    throughput = len(successful_results) / perf_results['duration_seconds']
    
    # Log results
    logger.info(f"Batch Research Performance:")
    logger.info(f"  Questions Processed: {len(successful_results)}")
    logger.info(f"  Total Duration: {perf_results['duration_seconds']:.2f}s")
    logger.info(f"  Throughput: {throughput:.2f} requests/second")
    logger.info(f"  Peak Memory: {perf_results['peak_memory_mb']:.2f} MB")
    logger.info(f"  Avg CPU: {perf_results['avg_cpu_percent']:.1f}%")
    
    return perf_results, throughput

@pytest.mark.asyncio
async def test_cache_performance():
    """Benchmark cache hit/miss performance"""
    
    benchmark = PerformanceBenchmark()
    
    # Mock MongoDB service with cache simulation
    mock_mongodb = AsyncMock(spec=MongoDBService)
    
    # Simulate cache behavior
    cache_data = {}
    
    async def mock_get_cached(url):
        return cache_data.get(url)
    
    async def mock_set_cache(url, content, keywords):
        cache_data[url] = content
        return True
    
    mock_mongodb.get_cached_content = mock_get_cached
    mock_mongodb.cache_content = mock_set_cache
    mock_mongodb.create_research_indexes = AsyncMock(return_value=True)
    mock_mongodb.get_cache_stats = AsyncMock(return_value={'total_entries': 0})
    
    # Create content extractor
    extractor = BrightDataContentExtractor(mock_mongodb)
    
    # Test URLs
    test_urls = [f"https://example.com/page{i}" for i in range(10)]
    
    benchmark.start_monitoring()
    
    # First round - all cache misses
    logger.info("Testing cache misses...")
    miss_times = []
    for url in test_urls:
        start_time = time.time()
        
        with patch.object(extractor, 'api_key', 'mock-key'):
            # Mock the API call
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=[{
                'title': 'Test Title',
                'content': 'Test content',
                'meta_description': 'Test description'
            }])
            mock_session.post = AsyncMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            
            with patch.object(extractor, 'get_session', return_value=mock_session):
                result = await extractor.extract_content(url)
        
        miss_time = time.time() - start_time
        miss_times.append(miss_time)
        benchmark.record_metrics()
    
    # Second round - all cache hits
    logger.info("Testing cache hits...")
    hit_times = []
    for url in test_urls:
        start_time = time.time()
        result = await extractor.extract_content(url)
        hit_time = time.time() - start_time
        hit_times.append(hit_time)
        benchmark.record_metrics()
    
    # Get performance results
    perf_results = benchmark.get_results()
    
    # Calculate cache performance
    avg_miss_time = sum(miss_times) / len(miss_times)
    avg_hit_time = sum(hit_times) / len(hit_times)
    speedup = avg_miss_time / avg_hit_time if avg_hit_time > 0 else 0
    
    # Assertions
    assert avg_hit_time < avg_miss_time, "Cache hits should be faster than misses"
    assert speedup > 2, f"Cache should provide significant speedup, got {speedup:.2f}x"
    
    # Log results
    logger.info(f"Cache Performance:")
    logger.info(f"  Avg Cache Miss Time: {avg_miss_time:.4f}s")
    logger.info(f"  Avg Cache Hit Time: {avg_hit_time:.4f}s")
    logger.info(f"  Cache Speedup: {speedup:.2f}x")
    logger.info(f"  Total Duration: {perf_results['duration_seconds']:.2f}s")
    
    return {
        'avg_miss_time': avg_miss_time,
        'avg_hit_time': avg_hit_time,
        'speedup': speedup,
        'performance': perf_results
    }

@pytest.mark.asyncio
async def test_token_optimization_performance():
    """Benchmark token optimization performance"""
    
    # Create test content with varying sizes
    test_contents = []
    for i in range(20):
        content_size = 1000 + (i * 500)  # Increasing content sizes
        test_contents.append({
            'title': f'Test Content {i}',
            'url': f'https://example.com/content{i}',
            'content': 'Test word ' * (content_size // 10),
            'success': True
        })
    
    optimizer = TokenOptimizer(max_tokens=50000)
    
    benchmark = PerformanceBenchmark()
    benchmark.start_monitoring()
    
    # Test optimization with different source limits
    results = {}
    
    for max_sources in [5, 8, 10, 15]:
        start_time = time.time()
        
        prepared_content, token_count = optimizer.prepare_content_for_analysis(
            test_contents, max_sources
        )
        
        optimization_time = time.time() - start_time
        benchmark.record_metrics()
        
        results[max_sources] = {
            'optimization_time': optimization_time,
            'token_count': token_count,
            'content_length': len(prepared_content),
            'sources_processed': min(max_sources, len(test_contents))
        }
    
    perf_results = benchmark.get_results()
    
    # Assertions
    assert all(r['token_count'] <= 45000 for r in results.values()), "Token count should stay within limits"
    assert results[5]['optimization_time'] < results[15]['optimization_time'], "More sources should take more time"
    
    # Log results
    logger.info("Token Optimization Performance:")
    for max_sources, result in results.items():
        logger.info(f"  Max Sources {max_sources}: {result['optimization_time']:.4f}s, "
                   f"{result['token_count']} tokens, {result['sources_processed']} sources")
    
    return results

@pytest.mark.asyncio
async def test_relevance_evaluation_performance():
    """Benchmark relevance evaluation performance"""
    
    # Mock OpenAI client
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "8.5"
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    
    evaluator = RelevanceEvaluator(mock_client, threshold=7.0)
    
    # Test content
    test_contents = [
        {
            'title': f'Test Content {i}',
            'url': f'https://example.com/test{i}',
            'content': f'Test content number {i} with some relevant information.'
        }
        for i in range(20)
    ]
    
    question = "What are the top CRM software companies?"
    
    benchmark = PerformanceBenchmark()
    benchmark.start_monitoring()
    
    # Single evaluation performance
    start_time = time.time()
    single_result = await evaluator.evaluate_relevance(question, test_contents[0])
    single_time = time.time() - start_time
    
    benchmark.record_metrics()
    
    # Batch evaluation performance
    start_time = time.time()
    batch_results = await evaluator.batch_evaluate(question, test_contents)
    batch_time = time.time() - start_time
    
    benchmark.record_metrics()
    
    perf_results = benchmark.get_results()
    
    # Calculate throughput
    single_throughput = 1 / single_time if single_time > 0 else 0
    batch_throughput = len(test_contents) / batch_time if batch_time > 0 else 0
    
    # Assertions
    assert single_result is not None
    assert len(batch_results) == len(test_contents)
    assert batch_throughput > single_throughput, "Batch processing should be more efficient"
    
    # Log results
    logger.info("Relevance Evaluation Performance:")
    logger.info(f"  Single Evaluation: {single_time:.4f}s ({single_throughput:.1f} evals/sec)")
    logger.info(f"  Batch Evaluation: {batch_time:.4f}s ({batch_throughput:.1f} evals/sec)")
    logger.info(f"  Batch Efficiency: {batch_throughput/single_throughput:.1f}x improvement")
    
    return {
        'single_time': single_time,
        'batch_time': batch_time,
        'single_throughput': single_throughput,
        'batch_throughput': batch_throughput
    }

@pytest.mark.asyncio
async def test_memory_usage_scaling():
    """Test memory usage with increasing workload"""
    
    benchmark = PerformanceBenchmark()
    benchmark.start_monitoring()
    
    # Create test workloads of increasing size
    workload_sizes = [10, 50, 100, 200]
    memory_usage_by_size = {}
    
    for size in workload_sizes:
        # Create test data
        test_data = [f"Test data item {i}" for i in range(size)]
        
        # Measure memory before processing
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024
        
        # Simulate processing
        processed_data = []
        for item in test_data:
            processed_data.append(item * 10)  # Simulate processing
        
        # Measure memory after processing
        memory_after = process.memory_info().rss / 1024 / 1024
        memory_usage = memory_after - memory_before
        
        memory_usage_by_size[size] = {
            'memory_before': memory_before,
            'memory_after': memory_after,
            'memory_usage': memory_usage,
            'memory_per_item': memory_usage / size if size > 0 else 0
        }
        
        benchmark.record_metrics()
        
        # Clear processed data to free memory
        del processed_data
        del test_data
    
    perf_results = benchmark.get_results()
    
    # Log results
    logger.info("Memory Usage Scaling:")
    for size, usage in memory_usage_by_size.items():
        logger.info(f"  Size {size}: {usage['memory_usage']:.2f} MB "
                   f"({usage['memory_per_item']:.4f} MB/item)")
    
    return memory_usage_by_size

if __name__ == "__main__":
    """Run performance benchmarks when executed directly"""
    
    async def run_all_benchmarks():
        logger.info("Starting DeepSeek Performance Benchmarks...")
        
        # Create mock service
        mock_mongodb = AsyncMock(spec=MongoDBService)
        mock_mongodb.get_cached_content = AsyncMock(return_value=None)
        mock_mongodb.cache_content = AsyncMock(return_value=True)
        mock_mongodb.create_research_indexes = AsyncMock(return_value=True)
        mock_mongodb.get_cache_stats = AsyncMock(return_value={'total_entries': 0})
        
        service = EnhancedDeepSeekResearchService(mongodb_service=mock_mongodb)
        
        # Mock external dependencies
        service.web_search.search = AsyncMock(return_value=[])
        service.content_extractor.extract_content = AsyncMock(return_value={
            'success': True, 'content': 'Test', 'title': 'Test'
        })
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage.total_tokens = 100
        service.client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Run benchmarks
        logger.info("1. Single Research Performance...")
        await test_single_research_performance(service)
        
        logger.info("2. Batch Research Performance...")  
        await test_batch_research_performance(service)
        
        logger.info("3. Cache Performance...")
        await test_cache_performance()
        
        logger.info("4. Token Optimization Performance...")
        await test_token_optimization_performance()
        
        logger.info("5. Relevance Evaluation Performance...")
        await test_relevance_evaluation_performance()
        
        logger.info("6. Memory Usage Scaling...")
        await test_memory_usage_scaling()
        
        logger.info("All benchmarks completed!")
    
    # Run benchmarks
    asyncio.run(run_all_benchmarks())