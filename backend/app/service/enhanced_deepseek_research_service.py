#!/usr/bin/env python3
"""
Enhanced DeepSeek Research Service for Production Use

Extracted and refactored from test_deepseek_advanced_web_research3_07.py
This service integrates advanced web research capabilities into the chat system.

Key Features:
- Multi-angle search query generation
- MongoDB caching for scraped web content
- Content deduplication and smart caching
- Enhanced content filtering and source diversification
- Iterative query refinement based on gaps
- Comprehensive logging and performance analysis
- Statistical summary generation using DeepSeek API reasoning
- Automatic extraction of numerical metrics from scraped content
- Time-limited research sessions (10 minutes)
- Token optimization and content summarization
"""

import os
import json
import asyncio
import logging
import time
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

# Third-party imports
import aiohttp
import tiktoken
from openai import AsyncOpenAI

# Internal imports
from .mongodb_service import MongoDBService

# Get logger
logger = logging.getLogger(__name__)

# Constants
MAX_RESEARCH_TIME = 600  # 10 minutes in seconds
MAX_CONTENT_LENGTH = 2000  # Max characters per content piece for DeepSeek
MAX_TOTAL_TOKENS = 50000  # Conservative limit for DeepSeek input
TARGET_SOURCES_PER_ITERATION = 8  # Optimal number of sources per analysis

# Utility functions
def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens in text using tiktoken"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        # Fallback: rough estimation (1 token ≈ 4 characters)
        return len(text) // 4

def summarize_content(content: str, max_length: int = MAX_CONTENT_LENGTH) -> str:
    """Summarize content to fit within max_length"""
    if len(content) <= max_length:
        return content
    
    # Extract key sentences (first, last, and middle parts)
    sentences = content.split('. ')
    if len(sentences) <= 3:
        return content[:max_length] + "..."
    
    # Take first 2 sentences, middle sentence, and last sentence
    first_part = '. '.join(sentences[:2])
    middle_idx = len(sentences) // 2
    middle_part = sentences[middle_idx]
    last_part = sentences[-1]
    
    summarized = f"{first_part}. ... {middle_part}. ... {last_part}"
    
    if len(summarized) > max_length:
        return content[:max_length] + "..."
    
    return summarized

def check_time_limit(start_time: float, max_duration: float = MAX_RESEARCH_TIME) -> bool:
    """Check if time limit has been exceeded"""
    return (time.time() - start_time) >= max_duration

@dataclass
class TimingMetrics:
    """Track timing metrics for performance analysis with time limits"""
    start_time: float
    end_time: Optional[float] = None
    phase_times: Dict[str, float] = None
    time_limit_exceeded: bool = False
    
    def __post_init__(self):
        if self.phase_times is None:
            self.phase_times = {}
    
    def start_phase(self, phase_name: str):
        """Start timing a phase"""
        if check_time_limit(self.start_time):
            self.time_limit_exceeded = True
            logger.warning(f"⏰ Time limit exceeded, skipping phase: {phase_name}")
            return False
            
        self.phase_times[f"{phase_name}_start"] = time.time()
        logger.info(f"⏱️ Starting phase: {phase_name}")
        return True
    
    def end_phase(self, phase_name: str):
        """End timing a phase"""
        start_key = f"{phase_name}_start"
        duration_key = f"{phase_name}_duration"
        
        if start_key in self.phase_times:
            duration = time.time() - self.phase_times[start_key]
            self.phase_times[duration_key] = duration
            logger.info(f"⏱️ Completed phase: {phase_name} in {duration:.2f}s")
            return duration
        return 0
    
    def get_total_duration(self) -> float:
        """Get total duration"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    def get_phase_summary(self) -> Dict[str, float]:
        """Get summary of all phase durations"""
        return {k: v for k, v in self.phase_times.items() if k.endswith('_duration')}

@dataclass
class SearchMetrics:
    """Track search and analysis metrics with v3.07 enhancements"""
    total_queries: int = 0
    total_results: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    source_types: Dict[str, int] = None
    relevance_scores: List[float] = None
    content_summarized: int = 0
    tokens_saved: int = 0
    statistical_data_found: int = 0  # v3.07: Track statistical data extraction
    
    def __post_init__(self):
        if self.source_types is None:
            self.source_types = defaultdict(int)
        if self.relevance_scores is None:
            self.relevance_scores = []

class TimeManager:
    """Manages research session timing and constraints"""
    
    def __init__(self, max_duration: int = MAX_RESEARCH_TIME):
        self.max_duration = max_duration
        self.start_time = None
        self.warning_threshold = max_duration * 0.8  # 80% warning
        self.phase_times = {}
        self.warning_issued = False
        
    def start_session(self):
        """Start timing a research session"""
        self.start_time = time.time()
        logger.info(f"🕒 Research session started with {self.max_duration}s time limit")
        return self.start_time
    
    def get_remaining_time(self) -> float:
        """Get remaining time in seconds"""
        if not self.start_time:
            return self.max_duration
        
        elapsed = time.time() - self.start_time
        remaining = max(0, self.max_duration - elapsed)
        
        # Issue warning if close to time limit
        if remaining < (self.max_duration - self.warning_threshold) and not self.warning_issued:
            logger.warning(f"⚠️ Time warning: {remaining:.1f}s remaining")
            self.warning_issued = True
            
        return remaining
    
    def is_time_exceeded(self) -> bool:
        """Check if time limit is exceeded"""
        return self.get_remaining_time() <= 0
    
    def should_continue_phase(self, phase_name: str, estimated_duration: float = 60) -> bool:
        """Check if there's enough time to continue with a phase"""
        remaining = self.get_remaining_time()
        if remaining < estimated_duration:
            logger.warning(f"⏰ Skipping {phase_name}: insufficient time ({remaining:.1f}s < {estimated_duration}s)")
            return False
        return True
    
    def get_phase_summary(self) -> Dict[str, float]:
        """Get summary of all phase durations"""
        return self.phase_times

class TokenOptimizer:
    """Optimizes content for DeepSeek token limits"""
    
    def __init__(self, max_tokens: int = MAX_TOTAL_TOKENS):
        self.max_tokens = max_tokens
        self.safety_margin = 5000  # Keep some buffer for API overhead
        self.effective_limit = max_tokens - self.safety_margin
        
    def prepare_content_for_analysis(self, contents: List[Dict[str, Any]], 
                                   max_sources: int = TARGET_SOURCES_PER_ITERATION) -> Tuple[str, int]:
        """Prepare content for analysis within token limits"""
        
        # Prioritize successful extractions
        successful_contents = [c for c in contents if c.get('success', False)]
        
        if not successful_contents:
            return "No successful content extractions available.", 0
        
        # Limit number of sources
        selected_contents = successful_contents[:max_sources]
        
        # Build analysis text and monitor token count
        analysis_parts = []
        total_tokens = 0
        
        for i, content in enumerate(selected_contents):
            # Create content summary
            content_text = content.get('content', '')
            summarized = summarize_content(content_text, MAX_CONTENT_LENGTH)
            
            content_part = f"""
Source {i+1}: {content.get('title', 'Unknown Title')}
URL: {content.get('url', 'Unknown URL')}
Content: {summarized}
---
"""
            
            # Check token count
            part_tokens = count_tokens(content_part)
            if total_tokens + part_tokens > self.effective_limit:
                # Further summarize if needed
                shorter_summary = summarize_content(content_text, MAX_CONTENT_LENGTH // 2)
                shorter_part = f"""
Source {i+1}: {content.get('title', 'Unknown Title')}
Content: {shorter_summary}
---
"""
                shorter_tokens = count_tokens(shorter_part)
                
                if total_tokens + shorter_tokens <= self.effective_limit:
                    analysis_parts.append(shorter_part)
                    total_tokens += shorter_tokens
                else:
                    logger.warning(f"⚠️ Token limit reached, stopping at source {i}")
                    break
            else:
                analysis_parts.append(content_part)
                total_tokens += part_tokens
        
        combined_content = '\n'.join(analysis_parts)
        logger.info(f"📊 Prepared content: ~{total_tokens} tokens from {len(selected_contents)} sources")
        
        return combined_content, total_tokens

class BrightDataContentExtractor:
    """Extract content using Bright Data API"""
    
    def __init__(self, mongodb_service):
        self.mongodb_service = mongodb_service
        self.api_key = os.environ.get('BRIGHTDATA_API_KEY', '')
        self.api_url = os.environ.get('BRIGHTDATA_API_URL', 'https://api.brightdata.com/datasets/v3/scrape')
        self.session = None
        
        if not self.api_key:
            logger.warning("⚠️ BRIGHTDATA_API_KEY not set - content extraction will be limited")
    
    async def get_session(self):
        """Get or create aiohttp session"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def extract_content(self, url: str) -> Dict[str, Any]:
        """Extract content from URL using Bright Data API or cache"""
        
        # Check cache first
        cached_content = await self.mongodb_service.get_cached_content(url)
        if cached_content:
            logger.info(f"📋 Cache hit for URL: {url}")
            return cached_content
        
        logger.info(f"🌐 Extracting content from: {url}")
        
        if not self.api_key:
            return {
                'url': url,
                'title': 'API Key Missing',
                'content': 'Bright Data API key not configured',
                'success': False,
                'error': 'API key not configured'
            }
        
        try:
            session = await self.get_session()
            
            # Prepare Bright Data API request
            payload = {
                "urls": [url],
                "format": "json",
                "extract": {
                    "title": "title",
                    "content": "body_text",
                    "meta_description": "meta[name='description']@content"
                }
            }
            
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            async with session.post(self.api_url, json=payload, headers=headers, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data and len(data) > 0:
                        result = data[0]
                        extracted_content = {
                            'url': url,
                            'title': result.get('title', 'Unknown Title'),
                            'content': result.get('content', ''),
                            'meta_description': result.get('meta_description', ''),
                            'success': True,
                            'method': 'brightdata_api'
                        }
                        
                        # Cache the result
                        await self.mongodb_service.cache_content(url, extracted_content, ['general'])
                        
                        return extracted_content
                    else:
                        error_content = {
                            'url': url,
                            'title': 'Empty Response',
                            'content': 'Bright Data API returned empty response',
                            'success': False,
                            'error': 'Empty API response'
                        }
                        await self.mongodb_service.cache_content(url, error_content, ['error'])
                        return error_content
                else:
                    error_msg = f"HTTP {response.status}"
                    error_content = {
                        'url': url,
                        'title': 'API Error',
                        'content': f'Bright Data API error: {error_msg}',
                        'success': False,
                        'error': error_msg
                    }
                    await self.mongodb_service.cache_content(url, error_content, ['error'])
                    return error_content
                    
        except Exception as e:
            logger.error(f"❌ Content extraction failed for {url}: {e}")
            error_content = {
                'url': url,
                'title': 'Extraction Failed',
                'content': f'Failed to extract content: {str(e)}',
                'success': False,
                'error': str(e)
            }
            await self.mongodb_service.cache_content(url, error_content, ['error'])
            return error_content

class EnhancedGoogleWebSearchService:
    """Enhanced Google web search with caching and optimization"""
    
    def __init__(self):
        self.api_key = os.environ.get('GOOGLE_API_KEY', '')
        self.cse_id = os.environ.get('GOOGLE_CSE_ID', '')
        
        if not self.api_key or not self.cse_id:
            logger.warning("⚠️ Google Search API not configured")
    
    async def search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """Perform Google search"""
        
        if not self.api_key or not self.cse_id:
            logger.error("❌ Google Search API credentials not configured")
            return []
        
        try:
            import aiohttp
            
            params = {
                'key': self.api_key,
                'cx': self.cse_id,
                'q': query,
                'num': min(num_results, 10),
                'safe': 'active'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get('https://www.googleapis.com/customsearch/v1', 
                                     params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = []
                        
                        if 'items' in data:
                            for item in data['items']:
                                results.append({
                                    'title': item.get('title', ''),
                                    'url': item.get('link', ''),
                                    'snippet': item.get('snippet', ''),
                                    'display_link': item.get('displayLink', ''),
                                    'formatted_url': item.get('formattedUrl', ''),
                                    'cache_id': item.get('cacheId', ''),
                                    'page_map': item.get('pagemap', {})
                                })
                        
                        logger.info(f"🔍 Google search found {len(results)} results for: {query}")
                        return results
                    else:
                        logger.error(f"❌ Google Search API error: HTTP {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"❌ Google search failed: {e}")
            return []

# MongoDB cache service is now integrated into the main MongoDBService

class RelevanceEvaluator:
    """Evaluate content relevance using DeepSeek API"""
    
    def __init__(self, client: AsyncOpenAI, threshold: float = 7.0):
        self.client = client
        self.threshold = threshold
        self.cache = {}
    
    async def evaluate_relevance(self, question: str, content: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate relevance of content to question on 0-10 scale"""
        
        cache_key = f"{hash(question)}_{hash(content.get('url', ''))}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            prompt = f"""Rate the relevance of this content to the question on a scale of 0-10.

Question: {question}

Content Title: {content.get('title', 'No title')}
Content URL: {content.get('url', 'No URL')}
Content Preview: {content.get('content', '')[:500]}...

Provide only a number from 0-10 where:
- 0-3: Not relevant
- 4-6: Somewhat relevant  
- 7-8: Highly relevant
- 9-10: Extremely relevant

Response format: Just the number (e.g., "7.5")
"""
            
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                timeout=10.0
            )
            
            score_text = response.choices[0].message.content.strip()
            try:
                score = float(re.findall(r'\d+\.?\d*', score_text)[0])
                score = max(0, min(10, score))  # Clamp to 0-10 range
            except:
                score = 5.0  # Default to medium relevance if parsing fails
            
            result = {
                'score': score,
                'meets_threshold': score >= self.threshold,
                'evaluation': score_text
            }
            
            self.cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"❌ Relevance evaluation failed: {e}")
            return {'score': 5.0, 'meets_threshold': False, 'error': str(e)}
    
    async def batch_evaluate(self, question: str, contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate relevance for multiple contents"""
        tasks = [self.evaluate_relevance(question, content) for content in contents]
        return await asyncio.gather(*tasks, return_exceptions=True)

class EnhancedDeepSeekResearchService:
    """Enhanced DeepSeek research service for production chat integration"""
    
    def __init__(self, mongodb_service=None, cache_expiry_days: int = 30):
        self.api_key = os.environ.get('DEEPSEEK_API_KEY', '')
        self.api_url = os.environ.get('DEEPSEEK_API_URL', 'https://api.deepseek.com')
        self.cache_expiry_days = cache_expiry_days
        
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable is required")
        
        # Use provided MongoDB service or create new one
        self.mongodb_service = mongodb_service if mongodb_service else MongoDBService()
        
        # Initialize components
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.api_url)
        self.web_search = EnhancedGoogleWebSearchService()
        self.content_extractor = BrightDataContentExtractor(self.mongodb_service)
        
        # Timing and optimization
        self.time_manager = TimeManager(max_duration=600)  # 10 minutes
        self.token_optimizer = TokenOptimizer(max_tokens=50000)
        
        # Research tracking
        self.metrics = SearchMetrics()
        self.current_session = None
        
        # Relevance evaluation system
        self.relevance_evaluator = RelevanceEvaluator(self.client, threshold=7.0)
        
        logger.info("✅ EnhancedDeepSeekResearchService initialized")
    
    async def initialize(self):
        """Initialize the service and MongoDB connections"""
        # Create research indexes with configurable TTL
        await self.mongodb_service.create_research_indexes(self.cache_expiry_days)
        
        # Display cache stats
        cache_stats = await self.mongodb_service.get_cache_stats()
        logger.info(f"📊 Cache stats: {cache_stats['total_entries']} total, {cache_stats.get('successful_entries', 0)} successful")
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.content_extractor.close_session()
    
    async def conduct_deepseek_research(self, question: str, chat_id: str) -> Dict[str, Any]:
        """
        Conduct comprehensive research using the enhanced algorithm
        
        Args:
            question: The research question
            chat_id: Chat ID for progress tracking
            
        Returns:
            Complete research results
        """
        
        logger.info(f"🔬 Starting enhanced DeepSeek research for: {question}")
        
        # Initialize timing
        self.time_manager.start_session()
        
        # Initialize metrics
        self.metrics = SearchMetrics()
        
        results = {
            'original_question': question,
            'chat_id': chat_id,
            'timestamp': datetime.utcnow().isoformat(),
            'research_type': 'enhanced_deepseek_research',
            'success': False,
            'steps': {},
            'timing_metrics': {}
        }
        
        try:
            # Step 1: Generate search queries
            if not self.time_manager.should_continue_phase('query_generation', 30):
                results['error'] = 'Time limit exceeded during initialization'
                return results
            
            search_queries = await self._generate_search_queries(question)
            results['steps']['query_generation'] = {
                'queries': search_queries,
                'query_count': len(search_queries)
            }
            
            # Step 2: Perform web search
            if not self.time_manager.should_continue_phase('web_search', 60):
                results['error'] = 'Time limit exceeded before web search'
                return results
            
            search_results = await self._perform_web_search(search_queries)
            results['steps']['web_search'] = {
                'total_results': len(search_results),
                'successful_queries': len([q for q in search_queries if any(r['query'] == q for r in search_results)])
            }
            
            # Step 3: Extract content
            if not self.time_manager.should_continue_phase('content_extraction', 120):
                results['error'] = 'Time limit exceeded before content extraction'
                return results
            
            extracted_contents = await self._extract_content_batch(search_results)
            results['steps']['content_extraction'] = {
                'total_sources': len(extracted_contents),
                'successful_extractions': sum(1 for c in extracted_contents if c.get('success', False)),
                'cache_hits': self.metrics.cache_hits,
                'cache_misses': self.metrics.cache_misses
            }
            
            # Step 4: Evaluate relevance
            if not self.time_manager.should_continue_phase('relevance_evaluation', 90):
                results['error'] = 'Time limit exceeded before relevance evaluation'
                return results
            
            relevance_results = await self._evaluate_relevance(question, extracted_contents)
            high_relevance_content = [c for c in relevance_results if c.get('relevance', {}).get('meets_threshold', False)]
            
            results['steps']['relevance_evaluation'] = {
                'total_evaluated': len(relevance_results),
                'high_relevance_count': len(high_relevance_content),
                'average_relevance': sum(c.get('relevance', {}).get('score', 0) for c in relevance_results) / len(relevance_results) if relevance_results else 0
            }
            
            # Step 5: Generate comprehensive analysis
            if not self.time_manager.should_continue_phase('analysis_generation', 60):
                # Return partial results with what we have
                results['partial_analysis'] = self._generate_partial_analysis(high_relevance_content)
            else:
                analysis = await self._generate_comprehensive_analysis(question, high_relevance_content)
                results['analysis'] = analysis
            
            # Compile final results
            results['success'] = True
            results['timing_metrics'] = self.time_manager.get_phase_summary()
            results['search_metrics'] = {
                'total_queries': self.metrics.total_queries,
                'total_results': self.metrics.total_results,
                'successful_extractions': self.metrics.successful_extractions,
                'failed_extractions': self.metrics.failed_extractions,
                'cache_performance': {
                    'hits': self.metrics.cache_hits,
                    'misses': self.metrics.cache_misses,
                    'hit_rate': self.metrics.cache_hits / (self.metrics.cache_hits + self.metrics.cache_misses) if (self.metrics.cache_hits + self.metrics.cache_misses) > 0 else 0
                }
            }
            
            # Include source information
            results['sources'] = [
                {
                    'title': c.get('title', 'Unknown'),
                    'url': c.get('url', ''),
                    'relevance_score': c.get('relevance', {}).get('score', 0),
                    'meets_threshold': c.get('relevance', {}).get('meets_threshold', False)
                }
                for c in relevance_results
            ]
            
            logger.info(f"✅ Enhanced DeepSeek research completed successfully in {self.time_manager.get_total_duration():.1f}s")
            
        except Exception as e:
            logger.error(f"❌ Enhanced DeepSeek research failed: {e}")
            results['error'] = str(e)
            results['timing_metrics'] = self.time_manager.get_phase_summary()
        
        return results
    
    async def _generate_search_queries(self, question: str) -> List[str]:
        """Generate multiple search queries for comprehensive research"""
        
        try:
            prompt = f"""Generate 3-4 different search queries to comprehensively research this question: {question}

Create queries that:
1. Directly address the main question
2. Focus on specific aspects or components
3. Include relevant industry terms and synonyms
4. Vary in scope (broad vs. specific)

Return the queries as a simple list, one per line, starting with "Query: "

Example format:
Query: main search terms
Query: specific aspect search
Query: alternative phrasing search
"""

            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                timeout=20.0
            )
            
            response_text = response.choices[0].message.content
            
            # Extract queries from response
            queries = []
            for line in response_text.split('\n'):
                if line.strip().startswith('Query:'):
                    query = line.replace('Query:', '').strip()
                    if query:
                        queries.append(query)
            
            # Fallback to original question if no queries extracted
            if not queries:
                queries = [question]
            
            self.metrics.total_queries = len(queries)
            logger.info(f"📝 Generated {len(queries)} search queries")
            return queries
            
        except Exception as e:
            logger.error(f"❌ Query generation failed: {e}")
            return [question]  # Fallback to original question
    
    async def _perform_web_search(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Perform web search for all queries"""
        
        all_results = []
        
        for query in queries:
            if self.time_manager.is_time_exceeded():
                logger.warning("⏰ Time limit reached during web search")
                break
            
            search_results = await self.web_search.search(query, num_results=8)
            self.metrics.total_results += len(search_results)
            
            # Add query context to results
            for result in search_results:
                result['query'] = query
                all_results.append(result)
        
        logger.info(f"🔍 Web search completed: {len(all_results)} total results from {len(queries)} queries")
        return all_results
    
    async def _extract_content_batch(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract content from search results with time management"""
        
        # Remove duplicates by URL
        unique_results = {}
        for result in search_results:
            url = result.get('url', '')
            if url and url not in unique_results:
                unique_results[url] = result
        
        unique_list = list(unique_results.values())
        logger.info(f"📄 Extracting content from {len(unique_list)} unique URLs")
        
        extracted_contents = []
        
        # Process in batches to manage time
        batch_size = 5
        for i in range(0, len(unique_list), batch_size):
            if self.time_manager.is_time_exceeded():
                logger.warning("⏰ Time limit reached during content extraction")
                break
            
            batch = unique_list[i:i + batch_size]
            
            # Extract content for batch
            tasks = [self.content_extractor.extract_content(result['url']) for result in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"❌ Content extraction error: {result}")
                    self.metrics.failed_extractions += 1
                    # Create error result
                    error_result = {
                        'url': batch[j]['url'],
                        'title': batch[j].get('title', 'Unknown'),
                        'content': f'Extraction failed: {str(result)}',
                        'success': False,
                        'search_result': batch[j]
                    }
                    extracted_contents.append(error_result)
                else:
                    if result.get('success', False):
                        self.metrics.successful_extractions += 1
                        if result.get('method') == 'cache':
                            self.metrics.cache_hits += 1
                        else:
                            self.metrics.cache_misses += 1
                    else:
                        self.metrics.failed_extractions += 1
                    
                    # Add search result context
                    result['search_result'] = batch[j]
                    extracted_contents.append(result)
        
        logger.info(f"✅ Content extraction completed: {self.metrics.successful_extractions} successful, {self.metrics.failed_extractions} failed")
        return extracted_contents
    
    async def _evaluate_relevance(self, question: str, contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate relevance of extracted content"""
        
        logger.info(f"🎯 Evaluating relevance for {len(contents)} pieces of content")
        
        # Only evaluate successful extractions
        successful_contents = [c for c in contents if c.get('success', False)]
        
        if not successful_contents:
            logger.warning("⚠️ No successful content extractions to evaluate")
            return contents
        
        # Batch evaluate relevance
        relevance_results = await self.relevance_evaluator.batch_evaluate(question, successful_contents)
        
        # Add relevance data to contents
        result_contents = []
        successful_index = 0
        
        for content in contents:
            if content.get('success', False) and successful_index < len(relevance_results):
                relevance_result = relevance_results[successful_index]
                if not isinstance(relevance_result, Exception):
                    content['relevance'] = relevance_result
                    self.metrics.relevance_scores.append(relevance_result.get('score', 0))
                else:
                    content['relevance'] = {'score': 0, 'meets_threshold': False, 'error': str(relevance_result)}
                successful_index += 1
            else:
                content['relevance'] = {'score': 0, 'meets_threshold': False}
            
            result_contents.append(content)
        
        high_relevance_count = sum(1 for c in result_contents if c.get('relevance', {}).get('meets_threshold', False))
        avg_relevance = sum(self.metrics.relevance_scores) / len(self.metrics.relevance_scores) if self.metrics.relevance_scores else 0
        
        logger.info(f"📊 Relevance evaluation completed: {high_relevance_count} high-relevance sources, average score {avg_relevance:.1f}")
        return result_contents
    
    async def _generate_comprehensive_analysis(self, question: str, relevant_contents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive analysis from relevant content"""
        
        if not relevant_contents:
            return {
                'analysis': 'No highly relevant content found to answer the question.',
                'confidence': 0.0,
                'sources_used': 0
            }
        
        # Prepare content for analysis
        prepared_content, token_count = self.token_optimizer.prepare_content_for_analysis(relevant_contents)
        
        try:
            prompt = f"""Based on the following research content, provide a comprehensive analysis answering this question: {question}

Research Content:
{prepared_content}

Instructions:
1. Directly answer the original question
2. Synthesize information from multiple sources
3. Identify key insights and findings
4. Note any limitations or conflicting information
5. Provide a confidence assessment (0-100%)

Structure your response as:
## Analysis
[Your comprehensive analysis here]

## Key Findings
- Finding 1
- Finding 2
- etc.

## Confidence Assessment
[0-100%] - [Brief explanation of confidence level]
"""

            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                timeout=45.0
            )
            
            analysis_text = response.choices[0].message.content
            
            # Extract confidence score if present
            confidence_match = re.search(r'(\d+)%', analysis_text)
            confidence = float(confidence_match.group(1)) / 100 if confidence_match else 0.7
            
            return {
                'analysis': analysis_text,
                'confidence': confidence,
                'sources_used': len(relevant_contents),
                'token_count': token_count
            }
            
        except Exception as e:
            logger.error(f"❌ Analysis generation failed: {e}")
            return {
                'analysis': f'Analysis generation failed: {str(e)}',
                'confidence': 0.0,
                'sources_used': len(relevant_contents),
                'error': str(e)
            }
    
    def _generate_partial_analysis(self, relevant_contents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate partial analysis when time is limited"""
        
        if not relevant_contents:
            return {
                'analysis': 'Time limit reached. No highly relevant content found.',
                'confidence': 0.0,
                'sources_used': 0,
                'partial': True
            }
        
        # Create simple summary from titles and snippets
        summaries = []
        for content in relevant_contents[:5]:  # Limit to top 5
            title = content.get('title', 'Unknown')
            snippet = content.get('search_result', {}).get('snippet', '')
            relevance_score = content.get('relevance', {}).get('score', 0)
            
            summaries.append(f"• {title} (Relevance: {relevance_score:.1f}/10)")
            if snippet:
                summaries.append(f"  {snippet[:200]}...")
        
        analysis = f"""⏰ Time Limit Reached - Partial Results

Based on the research conducted, here are the most relevant sources found:

{chr(10).join(summaries)}

Note: This is a partial analysis due to time constraints. The research identified {len(relevant_contents)} highly relevant sources that could provide more comprehensive insights if given more time.
"""
        
        return {
            'analysis': analysis,
            'confidence': 0.4,  # Lower confidence for partial results
            'sources_used': len(relevant_contents),
            'partial': True
        }


@dataclass
class AggregatedAnswer:
    """Aggregated answer data class"""
    content: str
    relevance_score: float
    source_urls: List[str]
    confidence_level: str
    extraction_time: datetime
    is_deduplicated: bool
    rank: int = 0


class AnswerAggregator:
    """High-relevance answer aggregation and ranking system"""
    
    def __init__(self, deduplication_threshold: float = 0.8):
        self.deduplication_threshold = deduplication_threshold
        self.aggregated_answers = []
        
    def aggregate_answers(self, evaluations: List[Dict]) -> List[AggregatedAnswer]:
        """Aggregate answers with 70%+ relevance and rank them"""
        
        # Extract high-relevance evaluations (7/10 or higher)
        high_relevance_evaluations = [
            eval for eval in evaluations 
            if eval.get('relevance', {}).get('meets_threshold', False)
        ]
        
        if not high_relevance_evaluations:
            logger.warning("No answers with 70%+ relevance found")
            return []
        
        # Convert evaluations to AggregatedAnswer objects
        candidates = []
        for eval in high_relevance_evaluations:
            content = eval.get('content', 'High-relevance content found')
            score = eval.get('relevance', {}).get('score', 0)
            url = eval.get('url', '')
            
            aggregated = AggregatedAnswer(
                content=content[:500],  # Limit content length
                relevance_score=score,
                source_urls=[url] if url else [],
                confidence_level=self._calculate_confidence_level(score),
                extraction_time=datetime.utcnow(),
                is_deduplicated=False
            )
            candidates.append(aggregated)
        
        # Remove duplicates and rank by relevance
        deduplicated_answers = self.deduplicate_content(candidates)
        ranked_answers = self.rank_by_relevance(deduplicated_answers)
        
        self.aggregated_answers = ranked_answers
        logger.info(f"Answer aggregation completed: {len(evaluations)} → {len(high_relevance_evaluations)} → {len(ranked_answers)}")
        
        return ranked_answers
    
    def deduplicate_content(self, answers: List[AggregatedAnswer]) -> List[AggregatedAnswer]:
        """Remove duplicate content"""
        if len(answers) <= 1:
            return answers
        
        deduplicated = []
        processed_indices = set()
        
        for i, answer in enumerate(answers):
            if i in processed_indices:
                continue
            
            similar_answers = [answer]
            similar_indices = {i}
            
            for j, other_answer in enumerate(answers[i+1:], i+1):
                if j in processed_indices:
                    continue
                
                similarity = self._calculate_similarity(answer, other_answer)
                if similarity >= self.deduplication_threshold:
                    similar_answers.append(other_answer)
                    similar_indices.add(j)
            
            merged_answer = self._merge_similar_answers(similar_answers)
            merged_answer.is_deduplicated = len(similar_answers) > 1
            
            deduplicated.append(merged_answer)
            processed_indices.update(similar_indices)
        
        return deduplicated
    
    def _calculate_similarity(self, answer1: AggregatedAnswer, answer2: AggregatedAnswer) -> float:
        """Calculate similarity between two answers"""
        # Check URL overlap
        common_urls = set(answer1.source_urls) & set(answer2.source_urls)
        if common_urls:
            return 1.0
        
        # Simple content similarity
        words1 = set(answer1.content.lower().split())
        words2 = set(answer2.content.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _merge_similar_answers(self, similar_answers: List[AggregatedAnswer]) -> AggregatedAnswer:
        """Merge similar answers"""
        best_answer = max(similar_answers, key=lambda x: x.relevance_score)
        
        all_urls = []
        for answer in similar_answers:
            all_urls.extend(answer.source_urls)
        unique_urls = list(set(all_urls))
        
        return AggregatedAnswer(
            content=best_answer.content,
            relevance_score=best_answer.relevance_score,
            source_urls=unique_urls,
            confidence_level=self._calculate_confidence_level(best_answer.relevance_score),
            extraction_time=best_answer.extraction_time,
            is_deduplicated=True
        )
    
    def rank_by_relevance(self, answers: List[AggregatedAnswer]) -> List[AggregatedAnswer]:
        """Rank answers by relevance score"""
        sorted_answers = sorted(answers, key=lambda x: x.relevance_score, reverse=True)
        
        for i, answer in enumerate(sorted_answers, 1):
            answer.rank = i
        
        return sorted_answers
    
    def _calculate_confidence_level(self, relevance_score: float) -> str:
        """Calculate confidence level based on relevance score"""
        if relevance_score >= 9.0:
            return "very_high"
        elif relevance_score >= 8.0:
            return "high"
        elif relevance_score >= 7.0:
            return "medium"
        else:
            return "low"
    
    def get_top_answer(self) -> Optional[AggregatedAnswer]:
        """Get the highest relevance answer"""
        return self.aggregated_answers[0] if self.aggregated_answers else None


@dataclass
class SummaryResult:
    """Summary result data class"""
    original_question: str
    summary_text: str
    relevance_score: float
    source_urls: List[str]
    confidence_metrics: Dict[str, float]
    generation_time: datetime
    token_usage: int


class SummaryGenerator:
    """Summary generation system for highest relevance answers"""
    
    def __init__(self, api_client: AsyncOpenAI):
        self.api_client = api_client
        
        self.summary_prompt_template = """
Question: {question}

Highest relevance content (Score: {relevance_score}/10):
{content}

Source URLs: {source_urls}

Instructions:
1. **Summary**: Provide a direct, concise answer to the original question
2. **Relevance**: Clearly show how this information relates to the original question
3. **Reliability**: Comment on the reliability of the information sources
4. **Structure**: Present in an organized, readable format

Create a summary in the following format:
【Summary】
[Direct answer to the original question in 3-5 sentences]

【Relevance】
[Explain how this information relates to the original question]

【Sources】
[Comment on the source URLs and their reliability]
"""
    
    async def generate_summary(self, question: str, best_answer: AggregatedAnswer) -> SummaryResult:
        """Generate summary for the highest relevance answer"""
        start_time = time.time()
        
        logger.info(f"Generating summary for answer with score {best_answer.relevance_score}/10")
        
        try:
            prompt = self.create_summary_prompt(question, best_answer)
            
            response = await self.api_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are an expert in information summarization. Create concise and accurate summaries based on the given information."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            summary_text = response.choices[0].message.content
            token_usage = response.usage.total_tokens if response.usage else 0
            
            confidence_metrics = self._calculate_confidence_metrics(
                best_answer.relevance_score,
                len(best_answer.source_urls)
            )
            
            summary_result = SummaryResult(
                original_question=question,
                summary_text=summary_text,
                relevance_score=best_answer.relevance_score,
                source_urls=best_answer.source_urls,
                confidence_metrics=confidence_metrics,
                generation_time=datetime.utcnow(),
                token_usage=token_usage
            )
            
            processing_time = time.time() - start_time
            logger.info(f"Summary generated in {processing_time:.2f}s, tokens: {token_usage}")
            
            return summary_result
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            
            return SummaryResult(
                original_question=question,
                summary_text=f"Summary generation failed: {str(e)}\n\nOriginal answer (Score: {best_answer.relevance_score}/10) available for reference.",
                relevance_score=best_answer.relevance_score,
                source_urls=best_answer.source_urls,
                confidence_metrics={"overall_confidence": 0.0, "error": True},
                generation_time=datetime.utcnow(),
                token_usage=0
            )
    
    def create_summary_prompt(self, question: str, best_answer: AggregatedAnswer) -> str:
        """Create summary generation prompt"""
        return self.summary_prompt_template.format(
            question=question,
            relevance_score=best_answer.relevance_score,
            content=best_answer.content,
            source_urls=", ".join(best_answer.source_urls)
        )
    
    def _calculate_confidence_metrics(self, relevance_score: float, source_count: int) -> Dict[str, float]:
        """Calculate confidence metrics"""
        base_confidence = relevance_score / 10.0
        source_bonus = min(source_count * 0.05, 0.2)  # Max 20% bonus for sources
        
        overall_confidence = min(base_confidence + source_bonus, 1.0)
        
        return {
            "overall_confidence": overall_confidence,
            "relevance_confidence": base_confidence,
            "source_diversity_bonus": source_bonus,
            "source_count": float(source_count)
        }


@dataclass
class FormattedResult:
    """Formatted result data class"""
    title: str
    content: str
    format_type: str
    metadata: Dict[str, Any]
    timestamp: datetime


class ResultFormatter:
    """Result formatting system for structured display"""
    
    def __init__(self):
        self.summary_template = """# Research Results: {title}

## Summary
{summary}

## Relevance Score
**{relevance_score:.1f}/10** - {confidence_level}

## Key Sources
{sources}

## Confidence Metrics
- Overall Confidence: {overall_confidence:.1%}
- Sources Used: {source_count}
- Generated: {timestamp}

---
*Generated by Enhanced DeepSeek Research*
"""
    
    def format_final_result(self, summary_result: SummaryResult) -> FormattedResult:
        """Format final research result for display"""
        
        # Format sources list
        sources_formatted = []
        for i, url in enumerate(summary_result.source_urls, 1):
            sources_formatted.append(f"{i}. {url}")
        sources_text = "\n".join(sources_formatted) if sources_formatted else "No sources available"
        
        # Determine confidence level
        overall_confidence = summary_result.confidence_metrics.get('overall_confidence', 0.0)
        if overall_confidence >= 0.8:
            confidence_level = "High Confidence"
        elif overall_confidence >= 0.6:
            confidence_level = "Medium Confidence"
        else:
            confidence_level = "Low Confidence"
        
        # Generate formatted content
        formatted_content = self.summary_template.format(
            title=summary_result.original_question,
            summary=summary_result.summary_text,
            relevance_score=summary_result.relevance_score,
            confidence_level=confidence_level,
            sources=sources_text,
            overall_confidence=overall_confidence,
            source_count=len(summary_result.source_urls),
            timestamp=summary_result.generation_time.strftime("%Y-%m-%d %H:%M:%S UTC")
        )
        
        return FormattedResult(
            title=f"Research: {summary_result.original_question}",
            content=formatted_content,
            format_type="research_summary",
            metadata={
                "relevance_score": summary_result.relevance_score,
                "source_count": len(summary_result.source_urls),
                "confidence_metrics": summary_result.confidence_metrics,
                "token_usage": summary_result.token_usage
            },
            timestamp=summary_result.generation_time
        )
    
    def format_error_result(self, question: str, error_message: str) -> FormattedResult:
        """Format error result for display"""
        
        error_content = f"""# Research Error

**Question:** {question}

**Error:** {error_message}

Please try rephrasing your question or check your network connection.

---
*Enhanced DeepSeek Research*
"""
        
        return FormattedResult(
            title=f"Error: {question}",
            content=error_content,
            format_type="error",
            metadata={"error": error_message},
            timestamp=datetime.utcnow()
        )