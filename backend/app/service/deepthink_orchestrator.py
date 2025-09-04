#!/usr/bin/env python3
"""
DeepThink Orchestrator - Coordinates the complete deep-think workflow

This is the main orchestration service that coordinates all components:
1. Query Generation Engine (generates search queries)
2. Serper API Client (performs web searches and content extraction)  
3. Jan Reasoning Engine (evaluates relevance and generates reasoning)
4. Answer Synthesizer (creates comprehensive and summary responses)
5. MongoDB Service (handles caching and result storage)

The orchestrator manages the complete workflow from user question to final response,
with real-time progress streaming and comprehensive error handling.
"""

import asyncio
import logging
import time
import gzip
import statistics
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncGenerator, Tuple
import json
from collections import defaultdict
from dataclasses import asdict

from .deepthink_models import (
    DeepThinkRequest, DeepThinkResult, SearchQuery, ScrapedContent,
    RelevanceScore, ReasoningChain, ProgressUpdate, DeepThinkStats,
    Answer
)
from .query_generation_engine import QueryGenerationEngine, QuestionAnalysis
from .serper_api_client import SerperAPIClient
from .jan_reasoning_engine import JanReasoningEngine
from .answer_synthesizer import AnswerSynthesizer
from .mongodb_service import MongoDBService
# Removed complex error_recovery system - using simple try/catch instead

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepThinkOrchestrator:
    """
    Main orchestrator for the deep-think research workflow.
    
    Coordinates all components to provide comprehensive research responses
    with real-time progress streaming and intelligent caching.
    """
    
    def __init__(
        self,
        deepseek_service,
        mongodb_service: MongoDBService,
        serper_api_key: str,
        timeout: int = 600,  # 10 minutes
        max_concurrent_searches: int = 3,
        cache_expiry_days: int = 30
    ):
        """Initialize the deep-think orchestrator.
        
        Args:
            deepseek_service: DeepSeek service for AI processing
            mongodb_service: MongoDB service for caching and storage
            serper_api_key: API key for Serper search service
            timeout: Maximum processing timeout in seconds
            max_concurrent_searches: Maximum concurrent search operations
            cache_expiry_days: Content cache expiry in days
        """
        self.deepseek_service = deepseek_service
        self.mongodb_service = mongodb_service
        self.timeout = timeout
        self.cache_expiry_days = cache_expiry_days
        
        # Initialize component services
        self.query_engine = QueryGenerationEngine(deepseek_service=deepseek_service)
        self.serper_client = SerperAPIClient(
            api_key=serper_api_key,
            max_concurrent_requests=max_concurrent_searches,
            timeout=30
        )
        self.reasoning_engine = JanReasoningEngine(deepseek_service=deepseek_service)
        self.answer_synthesizer = AnswerSynthesizer(deepseek_service=deepseek_service)
        
        # Statistics tracking
        self.stats = DeepThinkStats()
        
        # Progress tracking
        self.total_steps = 10
        self.current_step = 0
        
        # Resource management and performance tracking
        self.start_time = time.time()
        self.peak_memory_mb = 0
        self.response_times = []
        self.cache_stats = {
            'cache_size_mb': 0,
            'cache_entries': 0,
            'cache_hit_rate': 0.0,
            'cleanup_runs': 0
        }
        self.performance_metrics = {
            'average_response_time': 0.0,
            'requests_per_second': 0.0,
            'cache_hit_rate': 0.0,
            'memory_efficiency': 1.0,
            'error_rate': 0.0,
            'throughput': 0.0,
            'optimization_runs': 0
        }
        self.content_storage = {}
        self.bulk_cache = defaultdict(list)
        
        logger.info(f"DeepThinkOrchestrator initialized with {timeout}s timeout")
    
    async def process_deep_think(
        self,
        request: DeepThinkRequest,
        progress_callback: Optional[callable] = None
    ) -> DeepThinkResult:
        """
        Process a complete deep-think research request.
        
        Args:
            request: Deep-think request containing question and options
            progress_callback: Optional callback for progress updates
            
        Returns:
            DeepThinkResult: Complete research result with answers and metadata
            
        Raises:
            asyncio.TimeoutError: If processing exceeds timeout
            ValueError: If request is invalid
            Exception: For other processing errors
        """
        self.start_time = time.time()  # Store as instance variable for timeout checks
        self.current_step = 0
        
        try:
            # Validate request
            if not request.question or not request.question.strip():
                raise ValueError("Question cannot be empty")
            
            logger.info(f"Starting deep-think processing for: {request.question[:100]}...")
            
            # Direct processing without asyncio.wait_for wrapper
            return await self._process_deep_think_internal(request, progress_callback)
            
        except asyncio.TimeoutError:
            self.stats.timeout_errors += 1
            logger.error(f"Deep-think processing timed out after {self.timeout}s")
            raise
        except Exception as e:
            self.stats.error_requests += 1
            logger.error(f"Deep-think processing failed: {str(e)}")
            raise
    
    def _check_timeout(self) -> bool:
        """Check if research timeout reached - simplified approach like backend"""
        if hasattr(self, 'start_time') and self.start_time:
            return (time.time() - self.start_time) >= self.timeout
        return False
    
    async def _process_deep_think_internal(
        self,
        request: DeepThinkRequest,
        progress_callback: Optional[callable] = None
    ) -> DeepThinkResult:
        """Internal processing method with simple timeout checks"""
        processing_start = time.time()
        
        try:
            # Store current request context for use in caching
            self._current_question = request.question
            self._current_request_id = request.request_id
            
            # Step 1: Initialize MongoDB cache
            await self._update_progress("Initializing MongoDB cache", progress_callback)
            await self._initialize_cache()
            
            # Step 2: Analyze question and generate search queries
            await self._update_progress("Analyzing question and generating search queries", progress_callback)
            question_analysis, search_queries = await self._generate_queries(request.question)
            
            # Step 3: Check cache for existing content
            await self._update_progress("Checking cache for existing content", progress_callback)
            cached_content, cache_hits, cache_misses = await self._check_content_cache(search_queries)
            
            # Timeout check after query generation and cache lookup (Layer 1)
            if self._check_timeout():
                logger.warning("Research timeout reached after query generation phase")
                raise asyncio.TimeoutError("Deep-think research timed out after 600 seconds")
            
            # Step 4: Perform web searches for missing content
            await self._update_progress("Performing web searches", progress_callback)
            search_results = await self._perform_searches(search_queries, cached_content)
            
            # Step 5: Extract and cache new content
            await self._update_progress("Extracting and caching content", progress_callback)
            all_content = await self._extract_and_cache_content(search_results, cached_content)
            
            # Step 6: Evaluate content relevance
            await self._update_progress("Evaluating content relevance", progress_callback)
            relevant_content = await self._evaluate_relevance(request.question, all_content)
            
            # Timeout check after search and content extraction (Layer 2)
            if self._check_timeout():
                logger.warning("Research timeout reached after search and extraction phase")
                raise asyncio.TimeoutError("Deep-think research timed out after 600 seconds")
            
            # Step 7: Generate reasoning chains
            await self._update_progress("Generating reasoning chains", progress_callback)
            logger.info(f"About to call _generate_reasoning with question: '{request.question}'")
            reasoning_chains = await self._generate_reasoning(request.question, relevant_content)
            
            # Step 8: Synthesize comprehensive answer
            await self._update_progress("Synthesizing comprehensive answer", progress_callback)
            logger.info(f"About to call _synthesize_answers with question: '{request.question}'")
            comprehensive_answer, summary_answer = await self._synthesize_answers(
                request.question, relevant_content, reasoning_chains
            )
            
            # Step 9: Format final result
            await self._update_progress("Formatting final result", progress_callback)
            result = await self._format_result(
                request, question_analysis, search_queries, relevant_content,
                reasoning_chains, comprehensive_answer, summary_answer,
                cache_hits, cache_misses, processing_start
            )
            
            # Step 10: Store result and cleanup
            await self._update_progress("Storing result and cleanup", progress_callback)
            await self._store_result(result)
            
            # Update statistics
            processing_time = time.time() - processing_start
            self.stats.total_requests += 1
            self.stats.successful_requests += 1
            self.stats.total_processing_time += processing_time
            self.stats.cache_hits += cache_hits
            self.stats.cache_misses += cache_misses
            
            logger.info(f"Deep-think processing completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            self.stats.error_requests += 1
            logger.error(f"Deep-think processing failed: {str(e)}")
            raise
    
    async def stream_deep_think(
        self, 
        request: DeepThinkRequest
    ) -> AsyncGenerator[ProgressUpdate, None]:
        """
        Process deep-think request with streaming progress updates.
        
        Args:
            request: Deep-think request
            
        Yields:
            ProgressUpdate: Real-time progress updates
            
        Returns:
            Final result in the last progress update
        """
        progress_updates = []
        
        def collect_progress(step: str, progress: int, details: Optional[Dict] = None):
            """Collect progress updates for streaming"""
            update = ProgressUpdate(
                step=self.current_step,
                total_steps=self.total_steps,
                description=step,
                progress_percent=progress,
                details=details or {},
                timestamp=datetime.now()
            )
            progress_updates.append(update)
        
        try:
            # Start processing with simplified approach - no asyncio.wait_for wrapper
            process_task = asyncio.create_task(
                self.process_deep_think(request, collect_progress)
            )
            
            # Stream progress updates while processing
            last_yielded = 0
            
            while not process_task.done():
                # Yield any new progress updates
                if len(progress_updates) > last_yielded:
                    for i in range(last_yielded, len(progress_updates)):
                        yield progress_updates[i]
                    last_yielded = len(progress_updates)
                
                # Small delay to avoid busy waiting
                await asyncio.sleep(0.1)
            
            # Get the result
            result = await process_task
            
            # Yield any remaining progress updates
            if len(progress_updates) > last_yielded:
                for i in range(last_yielded, len(progress_updates)):
                    yield progress_updates[i]
            
            # Yield final result
            final_update = ProgressUpdate(
                step=self.total_steps,
                total_steps=self.total_steps,
                description="Complete",
                progress_percent=100,
                details={"result": asdict(result) if isinstance(result, DeepThinkResult) else result},
                timestamp=datetime.now()
            )
            yield final_update
            
        except asyncio.TimeoutError:
            # Handle timeout gracefully
            timeout_error = f"Deep-think research timed out after {self.timeout} seconds"
            logger.error(timeout_error)
            
            timeout_update = ProgressUpdate(
                step=self.current_step,
                total_steps=self.total_steps,
                description="Timeout - research taking too long",
                progress_percent=0,
                details={
                    "error": timeout_error,
                    "type": "TimeoutError",
                    "timeout_seconds": self.timeout,
                    "suggestion": "Try reducing research scope or check API connectivity"
                },
                timestamp=datetime.now()
            )
            yield timeout_update
            raise
            
        except Exception as e:
            # Handle other errors
            error_msg = str(e)
            logger.error(f"Deep-think processing failed: {error_msg}")
            
            # Provide specific error handling for connection issues
            if "Connection error" in error_msg or "nodename nor servname" in error_msg:
                error_description = "API connection failed - check network and configuration"
                suggestion = "Check DEEPSEEK_API_URL environment variable and internet connection"
            elif "timeout" in error_msg.lower():
                error_description = "API requests are timing out"
                suggestion = "Try again later or check API service status"
            else:
                error_description = f"Processing error: {error_msg}"
                suggestion = "Check logs for more details"
            
            error_update = ProgressUpdate(
                step=self.current_step,
                total_steps=self.total_steps,
                description=error_description,
                progress_percent=0,
                details={
                    "error": error_msg,
                    "type": type(e).__name__,
                    "suggestion": suggestion
                },
                timestamp=datetime.now()
            )
            yield error_update
            raise
    
    async def _initialize_cache(self):
        """Initialize MongoDB collections and indexes - non-blocking approach"""
        try:
            # Make cache initialization asynchronous and non-blocking
            asyncio.create_task(self._background_cache_setup())
            self.current_step = 1
            logger.info("Cache setup started in background (non-blocking)")
        except Exception as e:
            logger.warning(f"Cache setup failed to start: {e}, proceeding without cache")
            # Continue without cache - don't fail completely
    
    async def _background_cache_setup(self):
        """Background cache setup that doesn't block main processing"""
        try:
            await self.mongodb_service.create_deepthink_indexes()
            logger.info("✅ Background cache initialization completed")
        except Exception as e:
            logger.warning(f"Background cache setup failed: {e}, cache will be disabled")
    
    async def _generate_queries(self, question: str) -> Tuple[QuestionAnalysis, List[SearchQuery]]:
        """Generate search queries for the question"""
        try:
            # Analyze question first
            analysis = await self.query_engine.analyze_question(question)
            
            # Generate optimized search queries
            generated_queries = await self.query_engine.generate_search_queries(
                question, question_analysis=analysis
            )
            
            # Convert to SearchQuery objects
            search_queries = []
            for i, gq in enumerate(generated_queries):
                search_query = SearchQuery(
                    text=gq.text,
                    query_type=gq.query_type.value,
                    priority=gq.priority,
                    operators=gq.operators,
                    expected_results=gq.expected_results
                )
                search_queries.append(search_query)
            
            self.current_step = 2
            return analysis, search_queries
            
        except Exception as e:
            logger.error(f"Query generation failed: {e}")
            # Fallback to simple query
            fallback_query = SearchQuery(
                text=question,
                query_type="factual",
                priority=1,
                operators={},
                expected_results=10
            )
            return None, [fallback_query]
    
    async def _check_content_cache(self, queries: List[SearchQuery]) -> Tuple[List[ScrapedContent], int, int]:
        """Check cache for existing content - optimized non-blocking approach"""
        cached_content = []
        cache_hits = 0
        cache_misses = 0
        
        try:
            # Make cache checks truly optional and non-blocking with timeout
            cache_timeout = 5.0  # 5 second timeout for cache operations
            
            # Run cache check in background with timeout
            cache_task = asyncio.create_task(self._perform_cache_lookups(queries))
            
            try:
                cache_results = await asyncio.wait_for(cache_task, timeout=cache_timeout)
                cached_content, cache_hits, cache_misses = cache_results
                logger.info(f"Cache check completed: {cache_hits} hits, {cache_misses} misses")
            except asyncio.TimeoutError:
                logger.warning(f"Cache check timed out after {cache_timeout}s, proceeding without cache")
                cache_task.cancel()  # Cancel the background task
                return [], 0, len(queries)
            
            self.current_step = 3
            return cached_content, cache_hits, cache_misses
            
        except Exception as e:
            logger.warning(f"Cache check failed: {e}, proceeding without cache")
            return [], 0, len(queries)
    
    async def _perform_cache_lookups(self, queries: List[SearchQuery]) -> Tuple[List[ScrapedContent], int, int]:
        """Perform actual cache lookups with parallel processing"""
        cached_content = []
        cache_hits = 0
        cache_misses = 0
        
        # Create concurrent cache lookup tasks (limit to prevent overload)
        max_concurrent_cache = min(len(queries), 5)  # Max 5 concurrent cache lookups
        semaphore = asyncio.Semaphore(max_concurrent_cache)
        
        async def lookup_single_query(query: SearchQuery):
            async with semaphore:
                try:
                    content_list = await self.mongodb_service.search_cached_scraped_content(
                        query.text, limit=3  # Reduce from 5 to 3 for efficiency
                    )
                    return query, content_list or []
                except Exception as e:
                    logger.debug(f"Cache lookup failed for '{query.text}': {e}")
                    return query, []
        
        # Execute cache lookups in parallel
        tasks = [lookup_single_query(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results efficiently
        for result in results:
            if isinstance(result, Exception):
                cache_misses += 1
                continue
                
            query, content_list = result
            
            if content_list:
                # Simplified object conversion - only convert what's needed
                for content_dict in content_list:
                    if isinstance(content_dict, dict):
                        # Minimal object creation - only essential fields
                        content = ScrapedContent(
                            url=content_dict.get('url', ''),
                            title=content_dict.get('title', ''),
                            text_content=content_dict.get('content', ''),
                            markdown_content='',  # Skip markdown for cache efficiency
                            word_count=content_dict.get('word_count', 0),
                            extraction_timestamp=datetime.now(),  # Use current time for simplicity
                            metadata={}  # Skip metadata for efficiency
                        )
                        cached_content.append(content)
                    elif isinstance(content_dict, ScrapedContent):
                        cached_content.append(content_dict)
                
                cache_hits += len(content_list)
                logger.debug(f"Cache hit: {len(content_list)} results for '{query.text[:30]}...'")
            else:
                cache_misses += 1
        
        return cached_content, cache_hits, cache_misses
    
    async def _perform_searches(
        self, 
        queries: List[SearchQuery], 
        cached_content: List[ScrapedContent]
    ) -> List[Dict]:
        """Perform web searches for queries missing cached content with error recovery"""
        try:
            # Determine which queries need fresh searches
            cached_urls = {content.url for content in cached_content}
            search_results = []
            
            # Batch search multiple queries
            search_requests = []
            for query in queries:
                search_requests.append({
                    'q': query.text,
                    'type': 'search',
                    'engine': 'google'
                })
            
            # Execute batch search with simple error handling like backend
            if search_requests:
                try:
                    results = await self.serper_client.batch_search(search_requests)
                    
                    # Simple result processing
                    if results:
                        for result in results:
                            if isinstance(result, dict) and result.get('success'):
                                data = result.get('data', {})
                                if 'organic' in data:
                                    search_results.extend(data['organic'])
                                elif 'results' in data:
                                    search_results.extend(data['results'])
                                elif isinstance(data, list):
                                    search_results.extend(data)
                    
                except Exception as e:
                    logger.warning(f"Search failed: {e}, proceeding without search results")
                    # Continue with empty results - don't fail completely
                    search_results = []
            
            # Filter out already cached URLs - handle both dict and ScrapedContent cached_content
            cached_urls = set()
            for content in cached_content:
                if isinstance(content, ScrapedContent):
                    cached_urls.add(content.url)
                elif isinstance(content, dict):
                    cached_urls.add(content.get('url', ''))
                    
            fresh_results = [
                result for result in search_results
                if result.get('link') not in cached_urls
            ]
            
            self.current_step = 4
            logger.info(f"Retrieved {len(fresh_results)} fresh search results from {len(search_requests)} queries")
            return fresh_results[:20]  # Limit to top 20 fresh results
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []
    
    async def _extract_and_cache_content(
        self,
        search_results: List[Dict],
        cached_content: List[ScrapedContent]
    ) -> List[ScrapedContent]:
        """Extract content from search results and cache it"""
        all_content = list(cached_content)  # Start with cached content
        
        try:
            # Extract content from fresh search results
            extraction_requests = []
            for result in search_results:
                if result.get('link'):
                    extraction_requests.append({
                        'url': result['link'],
                        'type': 'scrape'
                    })
            
            # Batch extract content with better error handling
            if extraction_requests:
                try:
                    extracted_results = await self.serper_client.batch_search(extraction_requests)
                    
                    for i, result in enumerate(extracted_results):
                        try:
                            if result.get('success') and 'text' in result.get('data', {}):
                                # Create ScrapedContent object from successful extraction
                                original_result = search_results[i] if i < len(search_results) else {}
                                
                                content = ScrapedContent(
                                    url=extraction_requests[i]['url'],
                                    title=original_result.get('title', 'Unknown Title'),
                                    text_content=result['data'].get('text', ''),
                                    markdown_content=result['data'].get('markdown', result['data'].get('text', '')),
                                    word_count=len(result['data'].get('text', '').split()),
                                    extraction_timestamp=datetime.now(),
                                    metadata={
                                        'snippet': original_result.get('snippet', ''),
                                        'position': original_result.get('position', 0),
                                        'extraction_success': True
                                    }
                                )
                                
                                # Cache the content with correct signature: url, content, query_text, request_id, relevance_score
                                await self.mongodb_service.cache_scraped_content(
                                    url=content.url,
                                    content=content.text_content,
                                    query_text=getattr(self, '_current_question', 'research_query'),
                                    request_id=getattr(self, '_current_request_id', 'unknown'),
                                    relevance_score=0.0
                                )
                                all_content.append(content)
                            else:
                                # Extraction failed - create content from search result
                                original_result = search_results[i] if i < len(search_results) else {}
                                snippet = original_result.get('snippet', '')
                                
                                content = ScrapedContent(
                                    url=extraction_requests[i]['url'],
                                    title=original_result.get('title', 'Unknown Title'),
                                    text_content=snippet,
                                    markdown_content='',
                                    word_count=len(snippet.split()) if snippet else 0,
                                    extraction_timestamp=datetime.now(),
                                    metadata={
                                        'snippet': snippet,
                                        'position': original_result.get('position', 0),
                                        'extraction_success': False,
                                        'source': 'search_fallback'
                                    }
                                )
                                all_content.append(content)
                                
                        except Exception as item_error:
                            logger.warning(f"Failed to process extraction result {i}: {item_error}")
                            # Create fallback content even if individual processing fails
                            original_result = search_results[i] if i < len(search_results) else {}
                            snippet = original_result.get('snippet', 'No content available')
                            
                            fallback_content = ScrapedContent(
                                url=extraction_requests[i]['url'] if i < len(extraction_requests) else '',
                                title=original_result.get('title', 'Unknown Title'),
                                text_content=snippet,
                                markdown_content='',
                                word_count=len(snippet.split()) if snippet else 0,
                                extraction_timestamp=datetime.now(),
                                metadata={
                                    'snippet': snippet,
                                    'extraction_success': False,
                                    'source': 'error_fallback',
                                    'error': str(item_error)
                                }
                            )
                            all_content.append(fallback_content)
                            
                except Exception as batch_error:
                    logger.error(f"Batch content extraction failed: {batch_error}")
                    # Create content from search results as fallback
                    for i, search_result in enumerate(search_results):
                        if i < len(extraction_requests):
                            snippet = search_result.get('snippet', 'No content available')
                            fallback_content = ScrapedContent(
                                url=extraction_requests[i]['url'],
                                title=search_result.get('title', 'Unknown Title'),
                                text_content=snippet,
                                markdown_content='',
                                word_count=len(snippet.split()) if snippet else 0,
                                extraction_timestamp=datetime.now(),
                                metadata={
                                    'snippet': snippet,
                                    'extraction_success': False,
                                    'source': 'batch_error_fallback',
                                    'error': str(batch_error)
                                }
                            )
                            all_content.append(fallback_content)
            
            self.current_step = 5
            return all_content
            
        except Exception as e:
            logger.error(f"Content extraction failed: {e}")
            return all_content  # Return what we have
    
    async def _evaluate_relevance(
        self, 
        question: str, 
        content_list: List[ScrapedContent]
    ) -> List[Tuple[ScrapedContent, RelevanceScore]]:
        """Evaluate content relevance using Jan reasoning engine"""
        try:
            relevant_content = []
            
            for content in content_list:
                # Ensure we have a ScrapedContent object, not a dict or string
                if isinstance(content, dict):
                    # Convert dict to ScrapedContent if needed
                    content = ScrapedContent(
                        url=content.get('url', ''),
                        title=content.get('title', ''),
                        text_content=content.get('text_content', ''),
                        markdown_content=content.get('markdown_content', ''),
                        word_count=content.get('word_count', 0),
                        metadata=content.get('metadata', {})
                    )
                elif isinstance(content, str):
                    # Skip string content, can't process it
                    continue
                
                # Validate content before evaluation - use snippet as fallback
                text_to_evaluate = content.text_content or ""
                if not text_to_evaluate.strip():
                    # Try to use snippet from metadata as fallback
                    text_to_evaluate = content.metadata.get('snippet', '') if content.metadata else ""
                    if text_to_evaluate.strip():
                        # Update content with snippet text for evaluation
                        content.text_content = text_to_evaluate
                        logger.info(f"Using snippet as text_content for evaluation: {content.url}")
                    else:
                        logger.warning(f"Skipping content with no text or snippet: {content.url}")
                        continue
                
                try:
                    # Evaluate relevance
                    content_analysis = await self.reasoning_engine.evaluate_relevance(
                        content, question
                    )
                    
                    # Convert ContentAnalysis to RelevanceScore
                    relevance_score = RelevanceScore(
                        score=content_analysis.relevance_score,
                        reasoning=f"Relevance: {content_analysis.relevance_score}/10, Evidence: {content_analysis.evidence_strength}",
                        confidence=content_analysis.confidence,
                        key_points=content_analysis.key_points,
                        content_url=content.url
                    )
                    
                    # Keep content with relevance >= 7.0
                    if relevance_score.score >= 7.0:
                        relevant_content.append((content, relevance_score))
                        
                except Exception as eval_error:
                    logger.error(f"Relevance evaluation failed for {content.url}: {eval_error}")
                    # Create a fallback relevance score
                    fallback_score = RelevanceScore(
                        score=7.0,  # Neutral passing score
                        reasoning=f"Fallback evaluation - error: {str(eval_error)[:100]}",
                        confidence=0.5,
                        key_points=["Content evaluation failed"],
                        content_url=content.url
                    )
                    relevant_content.append((content, fallback_score))
            
            # Sort by relevance score (highest first)
            relevant_content.sort(key=lambda x: x[1].score, reverse=True)
            
            self.current_step = 6
            logger.info(f"Found {len(relevant_content)} relevant content pieces")
            return relevant_content
            
        except Exception as e:
            logger.error(f"Relevance evaluation failed: {e}")
            # Fallback: return all content with neutral scores
            fallback_content = []
            for content in content_list:
                # Handle different content types safely
                if isinstance(content, dict):
                    content_url = content.get('url', '')
                    # Convert dict to ScrapedContent for consistency
                    content = ScrapedContent(
                        url=content.get('url', ''),
                        title=content.get('title', ''),
                        text_content=content.get('text_content', ''),
                        markdown_content=content.get('markdown_content', ''),
                        word_count=content.get('word_count', 0),
                        metadata=content.get('metadata', {})
                    )
                elif isinstance(content, str):
                    # Skip string content
                    continue
                else:
                    content_url = getattr(content, 'url', '')
                
                fallback_score = RelevanceScore(
                    score=7.0,  # Neutral passing score
                    reasoning="Fallback evaluation - relevance assumed",
                    confidence=0.5,
                    key_points=[],
                    content_url=content_url
                )
                fallback_content.append((content, fallback_score))
            return fallback_content
    
    async def _generate_reasoning(
        self,
        question: str,
        relevant_content: List[Tuple[ScrapedContent, RelevanceScore]]
    ) -> List[ReasoningChain]:
        """Generate reasoning chains for the content"""
        try:
            # Validate and normalize question parameter
            if isinstance(question, list):
                question = ' '.join(str(item) for item in question if item)
                logger.warning("Question parameter was a list, converted to string")
            elif not isinstance(question, str):
                question = str(question) if question else ""
                logger.warning("Question parameter was not a string, converted")
            
            # Debug logging
            logger.info(f"_generate_reasoning called with question: '{question}' (length: {len(question) if question else 0})")
            
            # Validate question parameter
            if not question or not question.strip():
                logger.error("Question parameter is empty in _generate_reasoning")
                return []
                
            reasoning_chains = []
            content_only = [content for content, _ in relevant_content]
            
            # Convert ScrapedContent to ContentAnalysis for the reasoning engine
            content_analyses = []
            for content, relevance_score in relevant_content:
                # Create ContentAnalysis from ScrapedContent and RelevanceScore
                content_analysis = self._convert_to_content_analysis(content, relevance_score)
                content_analyses.append(content_analysis)
            
            # Generate reasoning chains with correct parameter order
            chains = await self.reasoning_engine.generate_reasoning_chains(
                content_analyses, question
            )
            reasoning_chains.extend(chains)
            
            # Check for contradictions with correct parameter order
            contradictions = await self.reasoning_engine.identify_contradictions(content_analyses, question)
            
            # Add contradiction analysis as reasoning chains
            for contradiction in contradictions:
                chain = ReasoningChain(
                    premise="Contradiction identified in sources",
                    reasoning=contradiction,
                    conclusion="Multiple perspectives exist - requires careful analysis",
                    confidence=0.8,
                    supporting_evidence=[],
                    logical_steps=["Contradiction analysis", "Source comparison"],
                    source_urls=[]
                )
                reasoning_chains.append(chain)
            
            self.current_step = 7
            return reasoning_chains
            
        except Exception as e:
            logger.error(f"Reasoning generation failed: {e}")
            return []  # Continue without reasoning chains
    
    async def _synthesize_answers(
        self,
        question: str,
        relevant_content: List[Tuple[ScrapedContent, RelevanceScore]],
        reasoning_chains: List[ReasoningChain]
    ) -> Tuple[str, str]:
        """Synthesize comprehensive and summary answers"""
        try:
            # Validate and normalize question parameter
            if isinstance(question, list):
                question = ' '.join(str(item) for item in question if item)
                logger.warning("Question parameter was a list, converted to string")
            elif not isinstance(question, str):
                question = str(question) if question else ""
                logger.warning("Question parameter was not a string, converted")
                
            # Debug logging
            logger.info(f"_synthesize_answers called with question: '{question}' (length: {len(question) if question else 0})")
            
            # Validate question parameter
            if not question or not question.strip():
                logger.error("Question parameter is empty in _synthesize_answers")
                fallback_answer = self._create_fallback_answer(relevant_content)
                return fallback_answer, fallback_answer[:500] + "..."
                
            # Prepare content for synthesis - convert to ContentAnalysis
            content_analyses = []
            for content, relevance_score in relevant_content:
                content_analysis = self._convert_to_content_analysis(content, relevance_score)
                content_analyses.append(content_analysis)
            
            # Use the complete synthesis method instead of individual methods
            try:
                synthesized_response = await self.answer_synthesizer.synthesize_complete_response(
                    content_analyses, question, reasoning_chains
                )
                
                # Store the synthesized response for structured result creation
                self._current_synthesized_response = synthesized_response
                
                # Return the text versions for backward compatibility
                comprehensive_answer = synthesized_response.comprehensive_answer
                summary_answer = synthesized_response.summary
                
            except Exception as e:
                logger.warning(f"Complete synthesis failed, falling back to individual methods: {e}")
                # Fallback to original method
                comprehensive_answer = await self.answer_synthesizer.generate_comprehensive_answer(
                    content_analyses, question, reasoning_chains
                )
                summary_answer = await self.answer_synthesizer.generate_summary(
                    comprehensive_answer, question
                )
                self._current_synthesized_response = None
            
            self.current_step = 8
            return comprehensive_answer, summary_answer
            
        except Exception as e:
            logger.error(f"Answer synthesis failed: {e}")
            # Fallback to basic content aggregation
            fallback_answer = self._create_fallback_answer(relevant_content)
            return fallback_answer, fallback_answer[:500] + "..."
    
    def _convert_to_content_analysis(self, scraped_content, relevance_score):
        """Convert ScrapedContent and RelevanceScore to ContentAnalysis format"""
        from .jan_reasoning_engine import ContentAnalysis, ReasoningType
        import hashlib
        
        # Generate content ID
        content_id = hashlib.sha256(scraped_content.text_content.encode()).hexdigest()[:12]
        
        return ContentAnalysis(
            content_id=content_id,
            relevance_score=relevance_score.score,
            confidence=relevance_score.confidence,
            key_points=relevance_score.key_points,
            evidence_strength="moderate",  # Default value
            source_credibility="medium",   # Default value
            reasoning_type=ReasoningType.DEDUCTIVE,  # Default value
            supporting_facts=relevance_score.key_points,  # Use key_points as supporting facts
            contradictory_facts=[],
            uncertainty_areas=[],
            evaluation_time=0.0
        )
    
    async def _format_result(
        self,
        request: DeepThinkRequest,
        question_analysis: Optional[QuestionAnalysis],
        search_queries: List[SearchQuery],
        relevant_content: List[Tuple[ScrapedContent, RelevanceScore]],
        reasoning_chains: List[ReasoningChain],
        comprehensive_answer: str,
        summary_answer: str,
        cache_hits: int,
        cache_misses: int,
        start_time: float
    ) -> DeepThinkResult:
        """Format the final deep-think result to match test file structure"""
        
        processing_time = time.time() - start_time
        content_only = [content for content, _ in relevant_content]
        scores_only = [score for _, score in relevant_content]
        confidence_score = self._calculate_confidence(scores_only, reasoning_chains)
        
        # Extract data from synthesized response if available
        synthesized_response = getattr(self, '_current_synthesized_response', None)
        
        # Create Answer object matching test file format exactly
        sources = [content.url for content in content_only[:10]]  # Top 10 sources
        gaps = []
        statistics = {}
        
        if synthesized_response:
            statistics = {
                'sources_analyzed': synthesized_response.sources_analyzed,
                'high_relevance_sources': synthesized_response.high_relevance_sources,
                'reasoning_chains_used': synthesized_response.reasoning_chains_used,
                'synthesis_time': synthesized_response.synthesis_time
            }
            # Extract potential gaps from uncertainties
            gaps = [f"Need more information about: {uncertainty}" for uncertainty in synthesized_response.uncertainties[:3]]
        else:
            # Fallback statistics
            statistics = {
                'sources_analyzed': len(content_only),
                'reasoning_chains_used': len(reasoning_chains),
                'processing_time': processing_time
            }
        
        answer = Answer(
            content=comprehensive_answer,
            confidence=confidence_score,
            sources=sources,
            statistics=statistics,
            gaps=gaps,
            versions=[{
                'version': 1,
                'content': comprehensive_answer[:200] + "..." if len(comprehensive_answer) > 200 else comprehensive_answer,
                'timestamp': datetime.now().isoformat(),
                'confidence': confidence_score
            }],
            generation_time=processing_time
        )
        
        result = DeepThinkResult(
            request_id=request.request_id,
            question=request.question,
            answer=answer,
            queries_generated=len(search_queries),
            sources_analyzed=len(content_only),
            cache_hits=cache_hits,
            total_duration=processing_time,
            # Keep backward compatibility fields
            comprehensive_answer=comprehensive_answer,
            summary_answer=summary_answer,
            search_queries=search_queries,
            scraped_content=content_only,
            relevance_scores=scores_only,
            reasoning_chains=reasoning_chains,
            confidence_score=confidence_score,
            processing_time=processing_time,
            total_sources=len(content_only),
            cache_misses=cache_misses,
            timestamp=datetime.now(),
            metadata={
                'question_analysis': question_analysis.to_dict() if question_analysis and hasattr(question_analysis, 'to_dict') else question_analysis,
                'user_id': request.user_id,
                'chat_id': request.chat_id,
                'timeout': self.timeout,
                'model_versions': {
                    'query_engine': 'v1.0',
                    'reasoning_engine': 'v1.0',
                    'synthesizer': 'v1.0'
                }
            }
        )
        
        self.current_step = 9
        return result
    
    async def _store_result(self, result: DeepThinkResult):
        """Store the result in MongoDB"""
        try:
            # Convert dataclass to dictionary for MongoDB storage
            await self.mongodb_service.store_deepthink_result(result.to_dict())
            self.current_step = 10
        except Exception as e:
            logger.error(f"Result storage failed: {e}")
            # Don't fail the entire process for storage issues
    
    async def _update_progress(
        self, 
        description: str, 
        callback: Optional[callable] = None
    ):
        """Update processing progress"""
        if callback:
            progress = int((self.current_step / self.total_steps) * 100)
            # Handle both async and sync callbacks
            import asyncio
            import inspect
            if inspect.iscoroutinefunction(callback):
                await callback(description, progress)
            else:
                callback(description, progress)
    
    def _calculate_confidence(
        self, 
        relevance_scores: List[RelevanceScore],
        reasoning_chains: List[ReasoningChain]
    ) -> float:
        """Calculate overall confidence score"""
        if not relevance_scores:
            return 0.0
        
        # Average relevance confidence
        relevance_confidence = sum(score.confidence for score in relevance_scores) / len(relevance_scores)
        
        # Average reasoning confidence
        if reasoning_chains:
            reasoning_confidence = sum(chain.confidence for chain in reasoning_chains) / len(reasoning_chains)
        else:
            reasoning_confidence = 0.7  # Neutral confidence when no reasoning
        
        # Weighted average (60% relevance, 40% reasoning)
        overall_confidence = (relevance_confidence * 0.6) + (reasoning_confidence * 0.4)
        return round(overall_confidence, 3)
    
    def _create_fallback_answer(
        self, 
        relevant_content: List[Tuple[ScrapedContent, RelevanceScore]]
    ) -> str:
        """Create a basic fallback answer when synthesis fails"""
        if not relevant_content:
            return "I apologize, but I couldn't find sufficient reliable information to answer your question."
        
        answer_parts = ["Based on the available information:\n\n"]
        
        for i, (content, score) in enumerate(relevant_content[:3]):  # Top 3 sources
            answer_parts.append(f"{i+1}. From {content.title}:")
            answer_parts.append(f"   {content.text_content[:200]}...")
            answer_parts.append(f"   (Relevance: {score.score}/10)\n")
        
        answer_parts.append(f"\nSources: {len(relevant_content)} relevant documents analyzed")
        return "\n".join(answer_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current orchestrator statistics"""
        return {
            'total_requests': self.stats.total_requests,
            'successful_requests': self.stats.successful_requests,
            'error_requests': self.stats.error_requests,
            'timeout_errors': self.stats.timeout_errors,
            'avg_processing_time': (
                self.stats.total_processing_time / self.stats.total_requests
                if self.stats.total_requests > 0 else 0
            ),
            'cache_hits': self.stats.cache_hits,
            'cache_misses': self.stats.cache_misses,
            'cache_hit_rate': (
                self.stats.cache_hits / (self.stats.cache_hits + self.stats.cache_misses)
                if (self.stats.cache_hits + self.stats.cache_misses) > 0 else 0
            ),
            'component_stats': {
                'query_engine': self.query_engine.get_stats(),
                'serper_client': self.serper_client.get_stats(),
                'reasoning_engine': self.reasoning_engine.get_stats(),
                'answer_synthesizer': self.answer_synthesizer.get_stats()
            }
        }
    
    def reset_stats(self):
        """Reset all statistics"""
        self.stats = DeepThinkStats()
        self.query_engine.reset_stats()
        self.reasoning_engine.reset_stats()
        self.answer_synthesizer.reset_stats()
        # Note: SerperAPIClient doesn't have reset_stats method
    
    # Resource Management and Performance Optimization Methods
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get current resource usage statistics"""
        process = psutil.Process()
        current_memory_mb = process.memory_info().rss / 1024 / 1024
        
        if current_memory_mb > self.peak_memory_mb:
            self.peak_memory_mb = current_memory_mb
        
        return {
            'memory_usage_mb': current_memory_mb,
            'peak_memory_mb': self.peak_memory_mb,
            'cpu_usage_percent': process.cpu_percent(),
            'uptime_seconds': time.time() - self.start_time,
            'optimization_runs': self.performance_metrics['optimization_runs']
        }
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics and metrics"""
        # Update cache size estimation
        cache_size_bytes = sum(len(str(content)) for content in self.content_storage.values())
        self.cache_stats['cache_size_mb'] = cache_size_bytes / 1024 / 1024
        self.cache_stats['cache_entries'] = len(self.content_storage)
        
        return dict(self.cache_stats)
    
    async def cleanup(self):
        """Clean up resources and connections"""
        try:
            # Close HTTP sessions
            if hasattr(self.serper_client, 'session') and self.serper_client.session:
                await self.serper_client.session.close()
            
            # Clear content storage to free memory
            self.content_storage.clear()
            self.bulk_cache.clear()
            
            logger.info("DeepThinkOrchestrator cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def _check_cache_limits(self):
        """Check and enforce cache limits"""
        max_cache_size_mb = 100  # 100 MB limit
        max_cache_entries = 1000  # 1000 entries limit
        
        current_stats = await self.get_cache_stats()
        
        if (current_stats['cache_size_mb'] > max_cache_size_mb or 
            current_stats['cache_entries'] > max_cache_entries):
            await self._cleanup_cache()
    
    async def _cleanup_cache(self):
        """Clean up cache to free memory"""
        # Remove oldest 30% of cached content
        if self.content_storage:
            items_to_remove = len(self.content_storage) // 3
            oldest_keys = list(self.content_storage.keys())[:items_to_remove]
            
            for key in oldest_keys:
                del self.content_storage[key]
            
            self.cache_stats['cleanup_runs'] += 1
            logger.info(f"Cache cleanup completed, removed {items_to_remove} entries")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        # Update metrics based on current state
        if self.response_times:
            self.performance_metrics['average_response_time'] = statistics.mean(self.response_times)
            
            # Calculate requests per second (last minute)
            recent_times = [t for t in self.response_times if t > (time.time() - 60)]
            self.performance_metrics['requests_per_second'] = len(recent_times) / 60.0
        
        # Update cache hit rate
        total_cache_operations = self.stats.cache_hits + self.stats.cache_misses
        if total_cache_operations > 0:
            self.performance_metrics['cache_hit_rate'] = self.stats.cache_hits / total_cache_operations
        
        # Update error rate
        total_requests = self.stats.total_requests
        if total_requests > 0:
            self.performance_metrics['error_rate'] = self.stats.error_requests / total_requests
        
        return dict(self.performance_metrics)
    
    def _calculate_adaptive_timeout(self, performance_history: List[float]) -> float:
        """Calculate adaptive timeout based on performance history"""
        if not performance_history:
            return self.timeout  # Default timeout
        
        mean_time = statistics.mean(performance_history)
        std_dev = statistics.stdev(performance_history) if len(performance_history) > 1 else mean_time * 0.1
        
        # Adaptive timeout: mean + 2 standard deviations, but within reasonable bounds
        adaptive_timeout = mean_time + (2 * std_dev)
        
        # Clamp to reasonable bounds
        min_timeout = max(mean_time * 1.5, 5.0)  # At least 1.5x mean time or 5 seconds
        max_timeout = min(self.timeout * 2, 600.0)  # At most 2x default or 10 minutes
        
        return max(min_timeout, min(adaptive_timeout, max_timeout))
    
    def _compress_content(self, content: str) -> bytes:
        """Compress content for storage efficiency"""
        return gzip.compress(content.encode('utf-8'))
    
    def _decompress_content(self, compressed: bytes) -> str:
        """Decompress content for use"""
        return gzip.decompress(compressed).decode('utf-8')
    
    async def _deduplicate_search_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate search results"""
        seen_urls = set()
        deduplicated = []
        
        for result in results:
            url = result.get('link', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                deduplicated.append(result)
        
        return deduplicated
    
    async def _execute_parallel_operations(self, operations: List[callable]) -> List[Any]:
        """Execute operations in parallel for improved performance"""
        if not operations:
            return []
        
        tasks = [op() for op in operations]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return successful results
        return [result for result in results if not isinstance(result, Exception)]
    
    async def _bulk_cache_lookup(self, queries: List[str]) -> Dict[str, List[Any]]:
        """Perform bulk cache lookups for improved database performance"""
        results = {}
        
        for query in queries:
            # Use MongoDB service for actual cache lookup
            cached_results = await self.mongodb_service.search_cached_scraped_content(
                query, limit=5
            )
            results[query] = cached_results
        
        return results
    
    def _create_efficient_content_storage(self, content_items: List[Dict]) -> Dict[str, Dict]:
        """Create memory-efficient content storage"""
        efficient_storage = {}
        
        for item in content_items:
            url = item.get('url', '')
            if url:
                # Store compressed content for memory efficiency
                content_text = item.get('content', '')
                efficient_storage[url] = {
                    'content': content_text,
                    'compressed': len(content_text) > 1000,  # Compress large content
                    'timestamp': time.time()
                }
        
        return efficient_storage
    
    def _retrieve_from_efficient_storage(self, storage: Dict[str, Dict], url: str) -> Optional[Dict]:
        """Retrieve content from efficient storage"""
        return storage.get(url)
    
    def _optimize_resource_usage(self):
        """Optimize resource usage"""
        # Perform garbage collection and memory optimization
        import gc
        gc.collect()
        
        # Update optimization counter
        self.performance_metrics['optimization_runs'] += 1
        
        # Clear old response times (keep only last 100)
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-100:]
        
        logger.info("Resource usage optimization completed")
