#!/usr/bin/env python3
"""
Enhanced Web Search Service
Advanced Google search with filtering, source diversification, and quality assessment
"""

import logging
import requests
import time
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from .interfaces import IWebSearchService, SearchResult
from .config import get_config_manager
from .metrics import get_metrics_collector

logger = logging.getLogger(__name__)


class EnhancedGoogleWebSearchService(IWebSearchService):
    """Enhanced Google web search service with filtering and source diversification"""
    
    def __init__(self):
        self.config = get_config_manager()
        self.metrics = get_metrics_collector()
        
        # Get Google API credentials
        google_creds = self.config.get_api_credentials('google')
        self.api_key = google_creds.get('api_key', '')
        self.cse_id = google_creds.get('cse_id', '')
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        
        # Search cache to avoid duplicate searches
        self.search_cache = {}
        
        # Research settings
        research_settings = self.config.get_research_settings()
        self.request_delay = research_settings['request_delay_seconds']
        self.max_retries = research_settings['max_retries']
        
        if not self.api_key:
            logger.warning("⚠️ GOOGLE_API_KEY not set. Web search functionality will be limited.")
        if not self.cse_id:
            logger.warning("⚠️ GOOGLE_CSE_ID not set. Web search functionality will be limited.")
    
    async def search(self, query: str, num_results: int = 5, **kwargs) -> List[SearchResult]:
        """Perform a basic web search"""
        return await self.search_with_filters(query, num_results, **kwargs)
    
    async def search_with_filters(self, query: str, num_results: int = 5, 
                                 exclude_domains: List[str] = None,
                                 prefer_domains: List[str] = None) -> List[SearchResult]:
        """Perform a web search with domain filtering and quality assessment"""
        if not self.api_key or not self.cse_id:
            logger.error("❌ Google API credentials not configured")
            self.metrics.increment_counter("search_errors", tags={"error_type": "missing_credentials"})
            return []
        
        timing_id = self.metrics.start_timing("google_search", {
            "query_length": str(len(query)),
            "num_results": str(num_results)
        })
        
        # Check cache first
        cache_key = f"{query}_{num_results}_{hash(str(exclude_domains))}"
        if cache_key in self.search_cache:
            logger.info(f"🔄 Using cached search results for: {query}")
            self.metrics.increment_counter("search_cache_hits")
            self.metrics.end_timing(timing_id)
            return self.search_cache[cache_key]
        
        try:
            # Build query with domain filters
            modified_query = query
            if exclude_domains:
                for domain in exclude_domains:
                    modified_query += f" -site:{domain}"
            
            # Add preferred domains boost (if specified)
            if prefer_domains:
                domain_boost = " OR ".join([f"site:{domain}" for domain in prefer_domains])
                modified_query = f"({modified_query}) OR ({domain_boost})"
            
            params = {
                'key': self.api_key,
                'cx': self.cse_id,
                'q': modified_query,
                'num': min(num_results, 10),
                'safe': 'active',
                'sort': 'date'  # Prefer recent content
            }
            
            logger.info(f"🔍 Google search: {modified_query}")
            
            # Perform search with retry logic
            response = await self._perform_search_with_retry(params)
            
            if not response:
                self.metrics.increment_counter("search_failures")
                return []
            
            data = response.json()
            results = []
            
            if 'items' in data:
                for item in data['items']:
                    search_result = SearchResult(
                        title=item.get('title', ''),
                        url=item.get('link', ''),
                        snippet=item.get('snippet', ''),
                        display_link=item.get('displayLink', ''),
                        search_query=query
                    )
                    
                    # Assess result quality
                    quality_score = self._assess_result_quality(search_result)
                    search_result.relevance_score = quality_score
                    
                    results.append(search_result)
            
            # Sort by quality score if available
            results.sort(key=lambda x: x.relevance_score or 0, reverse=True)
            
            # Cache results
            self.search_cache[cache_key] = results
            
            # Record metrics
            self.metrics.increment_counter("search_requests_successful")
            self.metrics.record_histogram_value("search_results_count", len(results))
            
            logger.info(f"✅ Found {len(results)} search results")
            return results
            
        except Exception as e:
            logger.error(f"❌ Google search failed: {e}")
            self.metrics.increment_counter("search_errors", tags={"error_type": "request_failed"})
            return []
        finally:
            self.metrics.end_timing(timing_id)
    
    async def _perform_search_with_retry(self, params: Dict[str, Any]) -> Optional[requests.Response]:
        """Perform search request with retry logic"""
        for attempt in range(self.max_retries):
            try:
                response = requests.get(self.base_url, params=params, timeout=15)
                response.raise_for_status()
                
                # Add delay to respect rate limits
                if self.request_delay > 0:
                    time.sleep(self.request_delay)
                
                return response
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Search attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    wait_time = (2 ** attempt) * self.request_delay
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {self.max_retries} search attempts failed")
                    return None
        
        return None
    
    def _assess_result_quality(self, result: SearchResult) -> float:
        """Assess the quality of a search result"""
        try:
            domain = urlparse(result.url).netloc.lower()
            quality_score = 5.0  # Default medium quality
            
            # High-quality business/news domains
            high_quality_indicators = [
                'reuters.com', 'bloomberg.com', 'wsj.com', 'ft.com', 'economist.com',
                'forbes.com', 'fortune.com', 'businessweek.com', 'harvard.edu',
                'mckinsey.com', 'bcg.com', 'deloitte.com', 'pwc.com', 'kpmg.com',
                'gartner.com', 'forrester.com', 'idc.com', 'statista.com'
            ]
            
            # Medium-quality sources
            medium_quality_indicators = [
                'techcrunch.com', 'venturebeat.com', 'wired.com', 'arstechnica.com',
                'zdnet.com', 'computerworld.com', 'infoworld.com', 'cio.com',
                'wikipedia.org', 'investopedia.com'
            ]
            
            # Low-quality or problematic sources
            low_quality_indicators = [
                'reddit.com', 'quora.com', 'yahoo.com/answers', 'stackoverflow.com',
                'medium.com', 'linkedin.com/pulse', 'facebook.com', 'twitter.com'
            ]
            
            # Assess quality based on domain
            if any(indicator in domain for indicator in high_quality_indicators):
                quality_score = 9.0
            elif any(indicator in domain for indicator in medium_quality_indicators):
                quality_score = 7.0
            elif any(indicator in domain for indicator in low_quality_indicators):
                quality_score = 3.0
            elif domain.endswith('.edu') or domain.endswith('.gov'):
                quality_score = 8.0
            elif domain.endswith('.org'):
                quality_score = 6.0
            elif domain.endswith('.com') and len(domain.split('.')) == 2:
                # Corporate websites
                quality_score = 6.0
            
            # Adjust based on content indicators
            title_lower = result.title.lower()
            snippet_lower = result.snippet.lower()
            
            # Positive indicators
            positive_indicators = [
                'report', 'analysis', 'research', 'study', 'data', 'statistics',
                'official', 'press release', 'whitepaper', 'survey'
            ]
            
            # Negative indicators
            negative_indicators = [
                'blog', 'opinion', 'personal', 'forum', 'discussion', 'comment',
                'social', 'user-generated', 'wiki'
            ]
            
            for indicator in positive_indicators:
                if indicator in title_lower or indicator in snippet_lower:
                    quality_score += 0.5
            
            for indicator in negative_indicators:
                if indicator in title_lower or indicator in snippet_lower:
                    quality_score -= 0.5
            
            # Ensure score is within bounds
            quality_score = max(1.0, min(10.0, quality_score))
            
            return quality_score
            
        except Exception as e:
            logger.warning(f"Failed to assess result quality for {result.url}: {e}")
            return 5.0  # Default score
    
    async def search_multiple_queries(self, queries: List[str], 
                                    max_results_per_query: int = 5) -> List[SearchResult]:
        """Perform searches for multiple queries and deduplicate results"""
        timing_id = self.metrics.start_timing("multi_query_search", {
            "query_count": str(len(queries))
        })
        
        all_results = []
        seen_urls = set()
        
        logger.info(f"🔍 Performing comprehensive search with {len(queries)} queries")
        
        # Domains to exclude for business research (low quality sources)
        exclude_domains = [
            'reddit.com', 'quora.com', 'yahoo.com', 'facebook.com', 
            'twitter.com', 'instagram.com', 'tiktok.com'
        ]
        
        for i, query in enumerate(queries, 1):
            logger.info(f"🔍 Search {i}/{len(queries)}: {query}")
            
            query_start = time.time()
            results = await self.search_with_filters(
                query, 
                num_results=max_results_per_query,
                exclude_domains=exclude_domains
            )
            query_time = time.time() - query_start
            
            # Filter out duplicate URLs
            new_results = []
            for result in results:
                if result.url not in seen_urls:
                    seen_urls.add(result.url)
                    new_results.append(result)
            
            all_results.extend(new_results)
            logger.info(f"  ✅ Query {i} completed: {len(new_results)} new results in {query_time:.2f}s")
            
            # Add delay between searches to be respectful
            if i < len(queries) and self.request_delay > 0:
                time.sleep(self.request_delay)
        
        # Sort all results by quality score
        all_results.sort(key=lambda x: x.relevance_score or 0, reverse=True)
        
        self.metrics.record_histogram_value("multi_search_total_results", len(all_results))
        self.metrics.record_histogram_value("multi_search_unique_results", len(seen_urls))
        
        logger.info(f"✅ Comprehensive search completed: {len(all_results)} unique results")
        
        self.metrics.end_timing(timing_id)
        return all_results
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search service statistics"""
        return {
            'cache_size': len(self.search_cache),
            'api_configured': bool(self.api_key and self.cse_id),
            'request_delay': self.request_delay,
            'max_retries': self.max_retries
        }
    
    def clear_cache(self):
        """Clear the search cache"""
        self.search_cache.clear()
        logger.info("Search cache cleared")


# Factory function for creating web search service
def create_web_search_service() -> IWebSearchService:
    """Create and return a web search service instance"""
    config = get_config_manager()
    
    if config.is_service_configured('google'):
        return EnhancedGoogleWebSearchService()
    else:
        logger.warning("Google search not configured, returning no-op service")
        return NoOpWebSearchService()


class NoOpWebSearchService(IWebSearchService):
    """No-operation web search service for when Google search is not configured"""
    
    async def search(self, query: str, num_results: int = 5, **kwargs) -> List[SearchResult]:
        logger.warning("Web search not available - Google API not configured")
        return []
    
    async def search_with_filters(self, query: str, num_results: int = 5, 
                                 exclude_domains: List[str] = None,
                                 prefer_domains: List[str] = None) -> List[SearchResult]:
        logger.warning("Web search not available - Google API not configured")
        return []
