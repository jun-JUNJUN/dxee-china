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
    RelevanceScore, ReasoningChain, ProgressUpdate, DeepThinkStats
)
from .query_generation_engine import QueryGenerationEngine, QuestionAnalysis
from .serper_api_client import SerperAPIClient
from .jan_reasoning_engine import JanReasoningEngine
from .answer_synthesizer import AnswerSynthesizer
from .mongodb_service import MongoDBService
from .error_recovery_system import error_recovery

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
        start_time = time.time()
        self.current_step = 0
        
        try:
            # Validate request
            if not request.question or not request.question.strip():
                raise ValueError("Question cannot be empty")
            
            logger.info(f"Starting deep-think processing for: {request.question[:100]}...")
            
            # Wrap the entire processing in timeout
            return await asyncio.wait_for(
                self._process_deep_think_internal(request, progress_callback),
                timeout=self.timeout
            )
            
        except asyncio.TimeoutError:
            self.stats.timeout_errors += 1
            logger.error(f"Deep-think processing timed out after {self.timeout}s")
            raise
        except Exception as e:
            self.stats.error_requests += 1
            logger.error(f"Deep-think processing failed: {str(e)}")
            raise
    
    async def _process_deep_think_internal(
        self, 
        request: DeepThinkRequest,
        progress_callback: Optional[callable] = None
    ) -> DeepThinkResult:
        """Internal processing method with timeout wrapper"""
        start_time = time.time()
        
        try:
            # Step 1: Initialize MongoDB cache
            await self._update_progress("Initializing MongoDB cache", progress_callback)
            await self._initialize_cache()
            
            # Step 2: Analyze question and generate search queries  
            await self._update_progress("Analyzing question and generating search queries", progress_callback)
            question_analysis, search_queries = await self._generate_queries(request.question)
            
            # Step 3: Check cache for existing content
            await self._update_progress("Checking cache for existing content", progress_callback)
            cached_content, cache_hits, cache_misses = await self._check_content_cache(search_queries)
            
            # Step 4: Perform web searches for missing content
            await self._update_progress("Performing web searches", progress_callback)
            search_results = await self._perform_searches(search_queries, cached_content)
            
            # Step 5: Extract and cache new content
            await self._update_progress("Extracting and caching content", progress_callback)
            all_content = await self._extract_and_cache_content(search_results, cached_content)
            
            # Step 6: Evaluate content relevance
            await self._update_progress("Evaluating content relevance", progress_callback)
            relevant_content = await self._evaluate_relevance(request.question, all_content)
            
            # Step 7: Generate reasoning chains
            await self._update_progress("Generating reasoning chains", progress_callback)
            reasoning_chains = await self._generate_reasoning(request.question, relevant_content)
            
            # Step 8: Synthesize comprehensive answer
            await self._update_progress("Synthesizing comprehensive answer", progress_callback)
            comprehensive_answer, summary_answer = await self._synthesize_answers(
                request.question, relevant_content, reasoning_chains
            )
            
            # Step 9: Format final result
            await self._update_progress("Formatting final result", progress_callback)
            result = await self._format_result(
                request, question_analysis, search_queries, relevant_content,
                reasoning_chains, comprehensive_answer, summary_answer,
                cache_hits, cache_misses, start_time
            )
            
            # Step 10: Store result and cleanup
            await self._update_progress("Storing result and cleanup", progress_callback)
            await self._store_result(result)
            
            # Update statistics
            processing_time = time.time() - start_time
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
            # Start processing
            process_task = asyncio.create_task(
                asyncio.wait_for(
                    self.process_deep_think(request, collect_progress),
                    timeout=self.timeout
                )
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
            
        except Exception as e:
            # Yield error update
            error_update = ProgressUpdate(
                step=self.current_step,
                total_steps=self.total_steps,
                description=f"Error: {str(e)}",
                progress_percent=0,
                details={"error": str(e), "type": type(e).__name__},
                timestamp=datetime.now()
            )
            yield error_update
            raise
    
    async def _initialize_cache(self):
        """Initialize MongoDB collections and indexes"""
        try:
            await self.mongodb_service.create_deepthink_indexes()
            self.current_step = 1
        except Exception as e:
            logger.error(f"Cache initialization failed: {e}")
            # Continue without cache - don't fail completely
    
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
        """Check cache for existing content"""
        cached_content = []
        cache_hits = 0
        cache_misses = 0
        
        try:
            for query in queries:
                # Search for cached content matching this query
                content_list = await self.mongodb_service.search_cached_scraped_content(
                    query.text, limit=5
                )
                
                if content_list:
                    cached_content.extend(content_list)
                    cache_hits += len(content_list)
                    logger.info(f"Cache hit: {len(content_list)} results for '{query.text}'")
                else:
                    cache_misses += 1
                    logger.info(f"Cache miss for '{query.text}'")
            
            self.current_step = 3
            return cached_content, cache_hits, cache_misses
            
        except Exception as e:
            logger.error(f"Cache check failed: {e}")
            return [], 0, len(queries)
    
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
            
            # Execute batch search with error recovery
            if search_requests:
                async def search_with_recovery():
                    return await self.serper_client.batch_search(search_requests)
                
                results = await error_recovery.execute_with_recovery(
                    'serper_api', search_with_recovery
                )
                
                # Handle partial failures - extract successful results
                if results:
                    for result in results:
                        if isinstance(result, dict):
                            if result.get('success') and 'organic' in result.get('data', {}):
                                search_results.extend(result['data']['organic'])
                            elif result.get('success') and 'data' in result:
                                # Handle different response formats
                                data = result['data']
                                if isinstance(data, list):
                                    search_results.extend(data)
                                elif isinstance(data, dict) and 'results' in data:
                                    search_results.extend(data['results'])
            
            # Filter out already cached URLs
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
            
            # Batch extract content
            if extraction_requests:
                extracted_results = await self.serper_client.batch_search(extraction_requests)
                
                for i, result in enumerate(extracted_results):
                    if result.get('success') and 'text' in result.get('data', {}):
                        # Create ScrapedContent object
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
                        
                        # Cache the content
                        await self.mongodb_service.cache_scraped_content(content)
                        all_content.append(content)
            
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
                # Evaluate relevance
                relevance_score = await self.reasoning_engine.evaluate_relevance(
                    question, content
                )
                
                # Keep content with relevance >= 7.0
                if relevance_score.score >= 7.0:
                    relevant_content.append((content, relevance_score))
            
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
                fallback_score = RelevanceScore(
                    score=7.0,  # Neutral passing score
                    reasoning="Fallback evaluation - relevance assumed",
                    confidence=0.5,
                    key_points=[],
                    content_url=content.url
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
            reasoning_chains = []
            content_only = [content for content, _ in relevant_content]
            
            # Generate reasoning chains
            chains = await self.reasoning_engine.generate_reasoning_chains(
                question, content_only
            )
            reasoning_chains.extend(chains)
            
            # Check for contradictions
            contradictions = await self.reasoning_engine.identify_contradictions(content_only)
            
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
            # Prepare content for synthesis
            content_only = [content for content, _ in relevant_content]
            relevance_scores = [score for _, score in relevant_content]
            
            # Generate both formats
            comprehensive_answer = await self.answer_synthesizer.generate_comprehensive_answer(
                question, content_only, relevance_scores, reasoning_chains
            )
            
            summary_answer = await self.answer_synthesizer.generate_summary(
                question, content_only, relevance_scores
            )
            
            self.current_step = 8
            return comprehensive_answer, summary_answer
            
        except Exception as e:
            logger.error(f"Answer synthesis failed: {e}")
            # Fallback to basic content aggregation
            fallback_answer = self._create_fallback_answer(relevant_content)
            return fallback_answer, fallback_answer[:500] + "..."
    
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
        """Format the final deep-think result"""
        
        processing_time = time.time() - start_time
        content_only = [content for content, _ in relevant_content]
        scores_only = [score for _, score in relevant_content]
        
        result = DeepThinkResult(
            request_id=request.request_id,
            question=request.question,
            comprehensive_answer=comprehensive_answer,
            summary_answer=summary_answer,
            search_queries=search_queries,
            scraped_content=content_only,
            relevance_scores=scores_only,
            reasoning_chains=reasoning_chains,
            confidence_score=self._calculate_confidence(scores_only, reasoning_chains),
            processing_time=processing_time,
            total_sources=len(content_only),
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            timestamp=datetime.now(),
            metadata={
                'question_analysis': question_analysis.dict() if question_analysis and hasattr(question_analysis, 'dict') else question_analysis,
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
            await self.mongodb_service.store_deepthink_result(result)
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
