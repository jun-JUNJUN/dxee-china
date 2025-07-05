#!/usr/bin/env python3
"""
Research Orchestrator - Main coordination service
Orchestrates the complete research process with iterative improvement and progress tracking
"""

import logging
import asyncio
import time
import re
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime
from dataclasses import asdict
from .interfaces import (
    IResearchOrchestrator, IProgressCallback, ResearchQuery, ResearchResult,
    SearchResult, ExtractedContent, AnalysisResult
)
from .config import get_config_manager
from .metrics import get_metrics_collector
from .cache import create_cache_service
from .web_search import create_web_search_service
from .content_extractor import create_content_extractor
from .ai_reasoning import create_ai_reasoning_service

logger = logging.getLogger(__name__)


class ProgressTracker(IProgressCallback):
    """Default progress tracker implementation"""
    
    def __init__(self):
        self.progress_history = []
        self.current_step = ""
        self.start_time = time.time()
    
    async def on_progress(self, step: str, data: Dict[str, Any]):
        """Called when progress is made in research"""
        self.current_step = step
        progress_entry = {
            'step': step,
            'data': data,
            'timestamp': datetime.utcnow(),
            'elapsed_time': time.time() - self.start_time
        }
        self.progress_history.append(progress_entry)
        logger.info(f"📊 Progress: {step} - {data.get('description', '')}")
    
    async def on_error(self, step: str, error: str, data: Dict[str, Any] = None):
        """Called when an error occurs"""
        error_entry = {
            'step': step,
            'error': error,
            'data': data or {},
            'timestamp': datetime.utcnow(),
            'elapsed_time': time.time() - self.start_time
        }
        self.progress_history.append(error_entry)
        logger.error(f"❌ Error in {step}: {error}")
    
    async def on_complete(self, result: ResearchResult):
        """Called when research is complete"""
        completion_entry = {
            'step': 'complete',
            'success': result.success,
            'research_type': result.research_type,
            'timestamp': datetime.utcnow(),
            'total_time': time.time() - self.start_time
        }
        self.progress_history.append(completion_entry)
        logger.info(f"✅ Research completed: {result.research_type} - Success: {result.success}")


class EnhancedResearchOrchestrator(IResearchOrchestrator):
    """Enhanced research orchestrator with iterative improvement and MongoDB caching"""
    
    def __init__(self, progress_callback: Optional[IProgressCallback] = None):
        self.config = get_config_manager()
        self.metrics = get_metrics_collector()
        self.progress_callback = progress_callback or ProgressTracker()
        
        # Initialize services
        self.cache_service = None
        self.web_search_service = None
        self.content_extractor = None
        self.ai_reasoning_service = None
        
        logger.info("Enhanced research orchestrator initialized")
    
    async def _initialize_services(self):
        """Initialize all research services"""
        if self.cache_service is None:
            timing_id = self.metrics.start_timing("service_initialization")
            
            try:
                # Initialize cache service
                self.cache_service = create_cache_service()
                if hasattr(self.cache_service, 'connect'):
                    await self.cache_service.connect()
                
                # Initialize other services
                self.web_search_service = create_web_search_service()
                self.content_extractor = create_content_extractor(self.cache_service)
                self.ai_reasoning_service = create_ai_reasoning_service()
                
                logger.info("✅ All research services initialized")
                
                # Display cache stats
                if hasattr(self.cache_service, 'get_cache_stats'):
                    cache_stats = await self.cache_service.get_cache_stats()
                    logger.info(f"📊 Cache stats: {cache_stats.get('total_entries', 0)} total entries, {cache_stats.get('fresh_entries', 0)} fresh")
                
            except Exception as e:
                logger.error(f"❌ Service initialization failed: {e}")
                raise
            finally:
                self.metrics.end_timing(timing_id)
    
    async def conduct_research(self, query: ResearchQuery) -> ResearchResult:
        """Conduct complete research process with iterative improvement"""
        await self._initialize_services()
        
        timing_id = self.metrics.start_timing("complete_research", {
            "search_mode": query.search_mode,
            "target_relevance": str(query.target_relevance)
        })
        
        logger.info(f"🚀 Starting enhanced research for: {query.question}")
        logger.info(f"🎯 Target relevance score: {query.target_relevance}/10")
        
        await self.progress_callback.on_progress("research_start", {
            "description": "Starting enhanced research",
            "question": query.question,
            "target_relevance": query.target_relevance
        })
        
        result = ResearchResult(
            query=query,
            research_type='enhanced_iterative_research',
            success=False,
            iterations=[],
            timestamp=datetime.utcnow()
        )
        
        current_relevance = 0
        iteration = 0
        all_extracted_contents = []
        
        try:
            while current_relevance < query.target_relevance and iteration < query.max_iterations:
                iteration += 1
                logger.info(f"🔄 Starting iteration {iteration}/{query.max_iterations}")
                
                await self.progress_callback.on_progress("iteration_start", {
                    "description": f"Starting iteration {iteration}",
                    "iteration": iteration,
                    "current_relevance": current_relevance
                })
                
                iteration_result = await self._conduct_iteration(
                    query, iteration, all_extracted_contents, result.iterations
                )
                
                result.iterations.append(iteration_result)
                current_relevance = iteration_result.get('relevance_achieved', 0)
                
                # Update extracted contents
                if 'extracted_contents' in iteration_result:
                    all_extracted_contents.extend(iteration_result['extracted_contents'])
                
                logger.info(f"✅ Iteration {iteration} completed: Relevance score {current_relevance}/10")
                
                if current_relevance >= query.target_relevance:
                    logger.info(f"🎉 Target relevance {query.target_relevance} achieved with score {current_relevance}!")
                    break
                elif iteration < query.max_iterations:
                    logger.info(f"🔄 Target not met ({current_relevance} < {query.target_relevance}), continuing to iteration {iteration + 1}")
            
            # Finalize result
            result.success = True
            result.extracted_contents = all_extracted_contents
            
            if result.iterations:
                final_iteration = result.iterations[-1]
                if 'analysis' in final_iteration:
                    result.analysis = final_iteration['analysis']
                if 'search_results' in final_iteration:
                    result.search_results = final_iteration['search_results']
            
            # Generate comprehensive metrics
            result.metrics = self._generate_final_metrics(result, current_relevance, iteration)
            
            logger.info(f"✅ Enhanced research completed!")
            logger.info(f"📊 Final relevance score: {current_relevance}/10 (Target: {query.target_relevance})")
            logger.info(f"🔄 Iterations completed: {iteration}/{query.max_iterations}")
            
            await self.progress_callback.on_complete(result)
            
        except Exception as e:
            logger.error(f"❌ Enhanced research failed: {e}")
            result.error = str(e)
            result.success = False
            
            await self.progress_callback.on_error("research_failed", str(e), {
                "iteration": iteration,
                "current_relevance": current_relevance
            })
        finally:
            self.metrics.end_timing(timing_id)
        
        return result
    
    async def _conduct_iteration(self, query: ResearchQuery, iteration: int, 
                               all_extracted_contents: List[ExtractedContent],
                               previous_iterations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Conduct a single research iteration"""
        iteration_result = {
            'iteration': iteration,
            'steps': {},
            'relevance_achieved': 0
        }
        
        try:
            # Step 1: Generate search queries
            if iteration == 1:
                # First iteration: check necessity and generate initial queries
                necessity_check = await self.ai_reasoning_service.check_web_search_necessity(query.question)
                
                iteration_result['steps']['necessity_check'] = {
                    'description': 'Check web search necessity',
                    'web_search_needed': necessity_check['web_search_needed'],
                    'success': True
                }
                
                if not necessity_check['web_search_needed']:
                    # Return direct answer
                    iteration_result['direct_answer'] = necessity_check['direct_answer']
                    iteration_result['relevance_achieved'] = 10  # Direct answer is considered complete
                    return iteration_result
                
                # Generate initial queries
                queries = await self.ai_reasoning_service.generate_search_queries(query.question, query.context)
                search_query = necessity_check.get('search_query')
                if search_query and search_query not in queries:
                    queries.insert(0, search_query)
            else:
                # Subsequent iterations: generate follow-up queries
                previous_analysis = previous_iterations[-1].get('analysis')
                if previous_analysis:
                    queries = await self.ai_reasoning_service.generate_followup_queries(
                        query.question, previous_analysis
                    )
                else:
                    queries = [query.question]
            
            iteration_result['steps']['query_generation'] = {
                'description': f'Generate queries for iteration {iteration}',
                'queries': queries,
                'query_count': len(queries),
                'success': True
            }
            
            await self.progress_callback.on_progress("query_generation", {
                "description": f"Generated {len(queries)} search queries",
                "queries": queries
            })
            
            # Step 2: Search existing cache (only in first iteration)
            cached_content = []
            if iteration == 1 and hasattr(self.cache_service, 'search_cached_content'):
                cached_content = await self._search_existing_cache(queries)
                iteration_result['steps']['cache_search'] = {
                    'description': 'Search MongoDB cache',
                    'cached_results': len(cached_content),
                    'success': True
                }
                
                if cached_content:
                    logger.info(f"💾 Using {len(cached_content)} cached results from MongoDB")
                    all_extracted_contents.extend(cached_content)
            
            # Step 3: Perform web search
            search_results = await self._perform_comprehensive_search(queries)
            iteration_result['steps']['web_search'] = {
                'description': f'Comprehensive search iteration {iteration}',
                'search_results': [asdict(sr) for sr in search_results],
                'total_results': len(search_results),
                'success': len(search_results) > 0
            }
            
            await self.progress_callback.on_progress("web_search", {
                "description": f"Found {len(search_results)} search results",
                "results_count": len(search_results)
            })
            
            # Step 4: Extract content (if new search results)
            new_extracted_contents = []
            if search_results:
                new_extracted_contents = await self._extract_and_analyze_content(search_results, queries)
                all_extracted_contents.extend(new_extracted_contents)
            
            iteration_result['steps']['content_extraction'] = {
                'description': f'Content extraction iteration {iteration}',
                'new_extractions': len(new_extracted_contents),
                'cached_extractions': len(cached_content) if iteration == 1 else 0,
                'total_extractions': len(all_extracted_contents),
                'successful_new': sum(1 for c in new_extracted_contents if c.success),
                'success': sum(1 for c in new_extracted_contents if c.success) > 0 or len(cached_content) > 0
            }
            
            await self.progress_callback.on_progress("content_extraction", {
                "description": f"Extracted content from {len(new_extracted_contents)} sources",
                "successful": sum(1 for c in new_extracted_contents if c.success),
                "total": len(new_extracted_contents)
            })
            
            # Step 5: Analyze content relevance
            if all_extracted_contents:
                analysis = await self.ai_reasoning_service.analyze_content_relevance(
                    query.question, all_extracted_contents
                )
                
                current_relevance = analysis.relevance_score or 0
                iteration_result['analysis'] = analysis
                iteration_result['relevance_achieved'] = current_relevance
                
                iteration_result['steps']['content_analysis'] = {
                    'description': f'Comprehensive analysis iteration {iteration}',
                    'relevance_score': current_relevance,
                    'sources_analyzed': len(all_extracted_contents),
                    'success': analysis.error is None
                }
                
                await self.progress_callback.on_progress("content_analysis", {
                    "description": f"Analysis completed with relevance score {current_relevance}/10",
                    "relevance_score": current_relevance,
                    "sources_analyzed": len(all_extracted_contents)
                })
            
            # Store search results and extracted contents for this iteration
            iteration_result['search_results'] = search_results
            iteration_result['extracted_contents'] = new_extracted_contents
            
        except Exception as e:
            logger.error(f"❌ Iteration {iteration} failed: {e}")
            iteration_result['error'] = str(e)
            await self.progress_callback.on_error(f"iteration_{iteration}", str(e))
        
        return iteration_result
    
    async def _search_existing_cache(self, queries: List[str]) -> List[ExtractedContent]:
        """Search for existing cached content based on queries"""
        if not hasattr(self.cache_service, 'search_cached_content'):
            return []
        
        timing_id = self.metrics.start_timing("cache_search")
        
        try:
            logger.info(f"🔍 Searching cache for {len(queries)} queries")
            
            # Extract keywords from all queries
            all_keywords = []
            for query in queries:
                keywords = [word.strip().lower() for word in re.split(r'[^\w]+', query) if len(word.strip()) > 2]
                all_keywords.extend(keywords)
            
            # Remove duplicates while preserving order
            unique_keywords = list(dict.fromkeys(all_keywords))
            
            cached_results = await self.cache_service.search_cached_content(unique_keywords)
            
            if cached_results:
                logger.info(f"💾 Found {len(cached_results)} cached results")
                self.metrics.increment_counter("cache_search_hits", len(cached_results))
            else:
                logger.info("📭 No cached results found")
                self.metrics.increment_counter("cache_search_misses")
            
            return cached_results
            
        except Exception as e:
            logger.error(f"❌ Cache search failed: {e}")
            return []
        finally:
            self.metrics.end_timing(timing_id)
    
    async def _perform_comprehensive_search(self, queries: List[str]) -> List[SearchResult]:
        """Perform comprehensive search across multiple queries"""
        timing_id = self.metrics.start_timing("comprehensive_search")
        
        try:
            research_settings = self.config.get_research_settings()
            max_results_per_query = research_settings['default_results_per_query']
            
            if hasattr(self.web_search_service, 'search_multiple_queries'):
                # Use enhanced multi-query search if available
                results = await self.web_search_service.search_multiple_queries(
                    queries, max_results_per_query
                )
            else:
                # Fallback to individual searches
                all_results = []
                seen_urls = set()
                
                for query in queries:
                    query_results = await self.web_search_service.search(query, max_results_per_query)
                    
                    # Filter duplicates
                    for result in query_results:
                        if result.url not in seen_urls:
                            seen_urls.add(result.url)
                            all_results.append(result)
                
                results = all_results
            
            logger.info(f"✅ Comprehensive search completed: {len(results)} unique results")
            return results
            
        except Exception as e:
            logger.error(f"❌ Comprehensive search failed: {e}")
            return []
        finally:
            self.metrics.end_timing(timing_id)
    
    async def _extract_and_analyze_content(self, search_results: List[SearchResult], 
                                         queries: List[str]) -> List[ExtractedContent]:
        """Extract content with caching and quality filtering"""
        timing_id = self.metrics.start_timing("content_extraction_batch")
        
        try:
            # Extract keywords for caching
            keywords = []
            for query in queries:
                query_keywords = [word.strip().lower() for word in re.split(r'[^\w]+', query) if len(word.strip()) > 2]
                keywords.extend(query_keywords)
            keywords = list(dict.fromkeys(keywords))  # Remove duplicates
            
            # Convert SearchResult to URLs for extraction
            urls = [result.url for result in search_results]
            
            # Use batch extraction if available
            if hasattr(self.content_extractor, 'extract_multiple_contents'):
                extracted_contents = await self.content_extractor.extract_multiple_contents(urls, keywords)
            else:
                # Fallback to individual extraction
                extracted_contents = []
                for url in urls:
                    content = await self.content_extractor.extract_content(url, keywords)
                    extracted_contents.append(content)
                    
                    # Add delay for rate limiting
                    if not content.from_cache:
                        await asyncio.sleep(self.config.get_research_settings()['request_delay_seconds'])
            
            # Add search result metadata to extracted contents
            for i, content in enumerate(extracted_contents):
                if i < len(search_results):
                    # Store the original search result for reference
                    content.search_result = search_results[i]
            
            successful_extractions = sum(1 for c in extracted_contents if c.success)
            logger.info(f"✅ Content extraction completed: {successful_extractions}/{len(extracted_contents)} successful")
            
            return extracted_contents
            
        except Exception as e:
            logger.error(f"❌ Content extraction batch failed: {e}")
            return []
        finally:
            self.metrics.end_timing(timing_id)
    
    def _generate_final_metrics(self, result: ResearchResult, final_relevance: int, 
                              iterations_completed: int) -> Dict[str, Any]:
        """Generate comprehensive final metrics"""
        research_settings = self.config.get_research_settings()
        
        # Calculate success metrics
        target_achieved = final_relevance >= result.query.target_relevance
        
        # Aggregate iteration metrics
        total_queries = 0
        total_search_results = 0
        total_extractions = 0
        successful_extractions = 0
        cache_hits = 0
        cache_misses = 0
        
        for iteration_data in result.iterations:
            steps = iteration_data.get('steps', {})
            
            if 'query_generation' in steps:
                total_queries += steps['query_generation'].get('query_count', 0)
            
            if 'web_search' in steps:
                total_search_results += steps['web_search'].get('total_results', 0)
            
            if 'content_extraction' in steps:
                total_extractions += steps['content_extraction'].get('new_extractions', 0)
                successful_extractions += steps['content_extraction'].get('successful_new', 0)
            
            if 'cache_search' in steps:
                cache_hits += steps['cache_search'].get('cached_results', 0)
        
        # Calculate rates
        extraction_success_rate = (successful_extractions / max(1, total_extractions)) * 100
        cache_hit_rate = (cache_hits / max(1, cache_hits + cache_misses)) * 100 if cache_hits + cache_misses > 0 else 0
        
        # Get cache performance
        cache_performance = {}
        if hasattr(self.cache_service, 'get_cache_stats'):
            try:
                cache_stats = asyncio.create_task(self.cache_service.get_cache_stats())
                # Note: In a real implementation, you'd await this properly
                cache_performance = {
                    'cache_enabled': True,
                    'total_cache_entries': 0,  # Would be filled from cache_stats
                    'fresh_cache_entries': 0   # Would be filled from cache_stats
                }
            except:
                cache_performance = {'cache_enabled': False}
        
        return {
            'target_achieved': target_achieved,
            'final_relevance_score': final_relevance,
            'iterations_completed': iterations_completed,
            'max_iterations': result.query.max_iterations,
            'search_metrics': {
                'total_queries': total_queries,
                'total_search_results': total_search_results,
                'total_extractions': total_extractions,
                'successful_extractions': successful_extractions,
                'extraction_success_rate': extraction_success_rate,
                'cache_hits': cache_hits,
                'cache_misses': cache_misses,
                'cache_hit_rate': cache_hit_rate
            },
            'cache_performance': cache_performance,
            'relevance_progression': [
                iter_data.get('relevance_achieved', 0) for iter_data in result.iterations
            ]
        }
    
    async def conduct_research_stream(self, query: ResearchQuery) -> AsyncGenerator[Dict[str, Any], None]:
        """Conduct research with streaming progress updates"""
        # Create a streaming progress callback
        class StreamingProgressCallback(IProgressCallback):
            def __init__(self, generator):
                self.generator = generator
            
            async def on_progress(self, step: str, data: Dict[str, Any]):
                await self.generator.asend({
                    'type': 'progress',
                    'step': step,
                    'data': data,
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            async def on_error(self, step: str, error: str, data: Dict[str, Any] = None):
                await self.generator.asend({
                    'type': 'error',
                    'step': step,
                    'error': error,
                    'data': data or {},
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            async def on_complete(self, result: ResearchResult):
                await self.generator.asend({
                    'type': 'complete',
                    'result': asdict(result),
                    'timestamp': datetime.utcnow().isoformat()
                })
        
        # This is a simplified streaming implementation
        # In a real implementation, you'd need proper async generator handling
        yield {'type': 'start', 'query': asdict(query)}
        
        # Conduct regular research with streaming callback
        original_callback = self.progress_callback
        
        try:
            # Note: This is a simplified approach. Real streaming would require
            # more sophisticated async generator coordination
            result = await self.conduct_research(query)
            yield {'type': 'complete', 'result': asdict(result)}
        finally:
            self.progress_callback = original_callback
    
    async def cleanup(self):
        """Clean up resources"""
        if self.cache_service and hasattr(self.cache_service, 'close'):
            await self.cache_service.close()
        
        logger.info("🧹 Research orchestrator cleanup completed")


# Factory function for creating research orchestrator
def create_research_orchestrator(progress_callback: Optional[IProgressCallback] = None) -> IResearchOrchestrator:
    """Create and return a research orchestrator instance"""
    return EnhancedResearchOrchestrator(progress_callback)
