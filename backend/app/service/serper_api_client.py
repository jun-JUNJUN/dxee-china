#!/usr/bin/env python3
"""
Serper API client for direct HTTP integration with deep-think functionality
"""

import os
import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import random
from collections import deque
import time

logger = logging.getLogger(__name__)


class SerperAPIError(Exception):
    """Base exception for Serper API errors"""
    pass


class SerperRateLimitError(SerperAPIError):
    """Rate limit exceeded error"""
    pass


class SerperNetworkError(SerperAPIError):
    """Network/connection error"""
    pass


class SerperAPIClient:
    """
    Client for Serper API with direct HTTP integration
    Supports both search and scraping functionality with retry logic and rate limiting
    """
    
    def __init__(self, api_key: Optional[str] = None, timeout: int = 30, 
                 requests_per_second: float = 2.0, max_concurrent_requests: int = 5,
                 html_cache_service=None):
        self.api_key = api_key or os.environ.get('SERPER_API_KEY')
        self.timeout = timeout
        self.search_url = os.environ.get('SERPER_SEARCH_URL', 'https://google.serper.dev/search')
        self.scrape_url = os.environ.get('SERPER_SCRAPE_URL', 'https://scrape.serper.dev/scrape')
        
        if not self.api_key:
            logger.warning("Serper API key not found. Set SERPER_API_KEY environment variable.")
        
        # HTML cache service for caching content
        self.html_cache_service = html_cache_service
        
        # Rate limiting and retry configuration
        self.max_retries = 3
        self.base_delay = 1.0
        self.max_delay = 60.0
        
        # Enhanced rate limiting
        self.requests_per_second = requests_per_second
        self.max_concurrent_requests = max_concurrent_requests
        self.request_interval = 1.0 / requests_per_second
        
        # Request tracking for monitoring
        self.request_count = 0
        self.error_count = 0
        self.last_request_time = None
        self.response_times = deque(maxlen=100)  # Keep last 100 response times
        
        # Rate limiting state
        self.last_request_timestamp = 0.0
        self.request_timestamps = deque(maxlen=100)  # Rolling window for rate limiting
        self.active_requests = 0
        self.request_queue = asyncio.Queue()
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        # Background queue processor
        self._queue_processor_task = None
        self._queue_processor_running = False
        
        # HTTP session for connection pooling
        connector = aiohttp.TCPConnector(
            limit=max_concurrent_requests * 2,
            limit_per_host=max_concurrent_requests,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        timeout_config = aiohttp.ClientTimeout(total=timeout, connect=10)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout_config,
            headers=self._get_headers()
        )
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests"""
        return {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json',
            'User-Agent': 'DeepThink-Research/1.0'
        }
    
    async def _wait_for_rate_limit(self):
        """Wait if necessary to respect rate limits"""
        current_time = time.time()
        
        # Clean old timestamps (older than 1 second)
        while self.request_timestamps and current_time - self.request_timestamps[0] > 1.0:
            self.request_timestamps.popleft()
        
        # Check if we need to wait
        if len(self.request_timestamps) >= self.requests_per_second:
            # Calculate how long to wait
            oldest_request = self.request_timestamps[0]
            wait_time = 1.0 - (current_time - oldest_request)
            if wait_time > 0:
                logger.debug(f"Rate limit wait: {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
        
        # Update timestamps
        self.request_timestamps.append(current_time)
    
    async def _acquire_request_slot(self):
        """Acquire a slot for making a request (respects concurrency limits)"""
        await self.semaphore.acquire()
        self.active_requests += 1
        await self._wait_for_rate_limit()
    
    def _release_request_slot(self):
        """Release a request slot"""
        self.active_requests -= 1
        self.semaphore.release()
    
    async def start_queue_processor(self):
        """Start the background queue processor"""
        if not self._queue_processor_running:
            self._queue_processor_running = True
            self._queue_processor_task = asyncio.create_task(self._process_request_queue())
            logger.info("Request queue processor started")
    
    async def stop_queue_processor(self):
        """Stop the background queue processor"""
        if self._queue_processor_running:
            self._queue_processor_running = False
            if self._queue_processor_task:
                self._queue_processor_task.cancel()
                try:
                    await self._queue_processor_task
                except asyncio.CancelledError:
                    pass
            logger.info("Request queue processor stopped")
    
    async def _process_request_queue(self):
        """Background processor for request queue"""
        while self._queue_processor_running:
            try:
                # Get request from queue (wait up to 1 second)
                try:
                    request_item = await asyncio.wait_for(self.request_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Process the request
                try:
                    result = await self._execute_queued_request(request_item)
                    request_item['future'].set_result(result)
                except Exception as e:
                    request_item['future'].set_exception(e)
                finally:
                    self.request_queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in queue processor: {e}")
    
    async def _execute_queued_request(self, request_item: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a queued request"""
        await self._acquire_request_slot()
        try:
            request_type = request_item['type']
            args = request_item['args']
            kwargs = request_item['kwargs']
            
            if request_type == 'search':
                return await self._direct_search(*args, **kwargs)
            elif request_type == 'scrape':
                return await self._direct_scrape(*args, **kwargs)
            else:
                raise ValueError(f"Unknown request type: {request_type}")
        finally:
            self._release_request_slot()
    
    async def queue_request(self, request_type: str, *args, **kwargs) -> Dict[str, Any]:
        """Queue a request for processing"""
        if not self._queue_processor_running:
            await self.start_queue_processor()
        
        future = asyncio.Future()
        request_item = {
            'type': request_type,
            'args': args,
            'kwargs': kwargs,
            'future': future,
            'queued_at': time.time()
        }
        
        await self.request_queue.put(request_item)
        logger.debug(f"Request queued: {request_type}")
        
        return await future
    
    async def _make_request(self, url: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic and error handling
        
        Args:
            url: API endpoint URL
            data: Request payload
            
        Returns:
            API response data
            
        Raises:
            SerperAPIError: For various API errors
        """
        if not self.api_key:
            raise SerperAPIError("Serper API key not configured")
        
        headers = self._get_headers()
        retry_count = 0
        
        while retry_count <= self.max_retries:
            try:
                # Track request timing
                start_time = time.time()
                self.request_count += 1
                self.last_request_time = datetime.utcnow()
                
                async with self.session.post(url, json=data, headers=headers) as response:
                        response_data = await response.json()
                        
                        # Handle successful response
                        if response.status == 200:
                            # Track response time
                            response_time = time.time() - start_time
                            self.response_times.append(response_time)
                            
                            logger.debug(f"Serper API request successful: {url} ({response_time:.2f}s)")
                            return response_data
                        
                        # Handle rate limiting
                        elif response.status == 429:
                            if retry_count < self.max_retries:
                                delay = self._calculate_backoff_delay(retry_count)
                                logger.warning(f"Rate limit hit, retrying in {delay}s (attempt {retry_count + 1})")
                                await asyncio.sleep(delay)
                                retry_count += 1
                                continue
                            else:
                                raise SerperRateLimitError(f"Rate limit exceeded: {response_data}")
                        
                        # Handle other HTTP errors
                        else:
                            error_msg = f"HTTP {response.status}: {response_data}"
                            if retry_count < self.max_retries and response.status >= 500:
                                # Retry on server errors
                                delay = self._calculate_backoff_delay(retry_count)
                                logger.warning(f"Server error, retrying in {delay}s: {error_msg}")
                                await asyncio.sleep(delay)
                                retry_count += 1
                                continue
                            else:
                                raise SerperAPIError(error_msg)
            
            except aiohttp.ClientError as e:
                if retry_count < self.max_retries:
                    delay = self._calculate_backoff_delay(retry_count)
                    logger.warning(f"Network error, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                    retry_count += 1
                    continue
                else:
                    raise SerperNetworkError(f"Network error after {self.max_retries} retries: {e}")
            
            except asyncio.TimeoutError:
                if retry_count < self.max_retries:
                    delay = self._calculate_backoff_delay(retry_count)
                    logger.warning(f"Request timeout, retrying in {delay}s")
                    await asyncio.sleep(delay)
                    retry_count += 1
                    continue
                else:
                    raise SerperNetworkError(f"Request timeout after {self.max_retries} retries")
        
        # This should never be reached due to the loop logic
        raise SerperAPIError("Unexpected error in request handling")
    
    async def _direct_search(self, query: str, num_results: int = 10, country: str = 'us', 
                            location: str = '', language: str = 'en') -> Dict[str, Any]:
        """Direct search method for queue processing"""
        if not query.strip():
            raise ValueError("Search query cannot be empty")
        
        if not 1 <= num_results <= 100:
            raise ValueError("Number of results must be between 1 and 100")
        
        payload = {
            'q': query.strip(),
            'num': num_results,
            'gl': country.lower(),
            'hl': language.lower()
        }
        
        if location:
            payload['location'] = location
        
        logger.info(f"Performing direct Serper search: '{query}' (num={num_results})")
        response = await self._make_request(self.search_url, payload)
        
        # Validate response structure
        if not isinstance(response, dict):
            raise SerperAPIError("Invalid response format")
        
        # Extract key metrics for logging
        organic_count = len(response.get('organic', []))
        news_count = len(response.get('news', []))
        logger.info(f"Search completed: {organic_count} organic results, {news_count} news results")
        
        return response
    
    async def _direct_scrape(self, url: str, extract_text: bool = True, 
                            extract_markdown: bool = True) -> Dict[str, Any]:
        """Direct scrape method for queue processing"""
        if not url or not url.strip():
            raise ValueError("URL cannot be empty")
        
        # Basic URL validation
        url = url.strip()
        if not (url.startswith('http://') or url.startswith('https://')):
            raise ValueError("URL must start with http:// or https://")
        
        payload = {'url': url}
        
        # Configure extraction options
        if extract_text:
            payload['extractText'] = True
        if extract_markdown:
            payload['extractMarkdown'] = True
        
        logger.info(f"Performing direct scraping: {url}")
        response = await self._make_request(self.scrape_url, payload)
        
        # Validate response structure
        if not isinstance(response, dict):
            raise SerperAPIError("Invalid response format")
        
        # Log extraction results
        text_length = len(response.get('text', ''))
        markdown_length = len(response.get('markdown', ''))
        title = response.get('title', 'Unknown')
        
        logger.info(f"Scraping completed for '{title}': {text_length} chars text, {markdown_length} chars markdown")
        
        return response
    
    def _calculate_backoff_delay(self, retry_count: int) -> float:
        """
        Calculate exponential backoff delay with jitter
        
        Args:
            retry_count: Current retry attempt (0-based)
            
        Returns:
            Delay in seconds
        """
        # Exponential backoff: base_delay * (2 ^ retry_count)
        delay = self.base_delay * (2 ** retry_count)
        
        # Add jitter to prevent thundering herd
        jitter = random.uniform(0.1, 0.5) * delay
        delay += jitter
        
        # Cap at max_delay
        return min(delay, self.max_delay)
    
    async def search(self, query: str, num_results: int = 10, country: str = 'us', 
                    location: str = '', language: str = 'en', use_queue: bool = True) -> Dict[str, Any]:
        """
        Perform web search using Serper API
        
        Args:
            query: Search query string
            num_results: Number of results to return (1-100)
            country: Country code for search (e.g., 'us', 'uk')
            location: Location for localized search
            language: Language code (e.g., 'en', 'es')
            use_queue: Whether to use request queue for rate limiting (default: True)
            
        Returns:
            Search results data
            
        Raises:
            SerperAPIError: For API errors
        """
        try:
            if use_queue:
                # Use queued execution with rate limiting
                return await self.queue_request('search', query, num_results, country, location, language)
            else:
                # Direct execution with manual rate limiting
                await self._acquire_request_slot()
                try:
                    return await self._direct_search(query, num_results, country, location, language)
                finally:
                    self._release_request_slot()
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Search failed for query '{query}': {e}")
            raise
    
    async def scrape(self, url: str, extract_text: bool = True, 
                    extract_markdown: bool = True, use_queue: bool = True) -> Dict[str, Any]:
        """
        Scrape content from URL using Serper API
        
        Args:
            url: URL to scrape
            extract_text: Whether to extract plain text
            extract_markdown: Whether to extract markdown content
            use_queue: Whether to use request queue for rate limiting (default: True)
            
        Returns:
            Scraped content data
            
        Raises:
            SerperAPIError: For API errors
        """
        try:
            if use_queue:
                # Use queued execution with rate limiting
                return await self.queue_request('scrape', url, extract_text, extract_markdown)
            else:
                # Direct execution with manual rate limiting
                await self._acquire_request_slot()
                try:
                    return await self._direct_scrape(url, extract_text, extract_markdown)
                finally:
                    self._release_request_slot()
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Scraping failed for URL '{url}': {e}")
            raise
    
    async def scrape_with_cache(self, url: str, extract_text: bool = True, 
                               extract_markdown: bool = True, use_queue: bool = True,
                               cache_expiry_days: Optional[int] = None) -> Tuple[Dict[str, Any], bool]:
        """
        Scrape content from URL with HTML caching support
        
        Args:
            url: URL to scrape
            extract_text: Whether to extract plain text
            extract_markdown: Whether to extract markdown content
            use_queue: Whether to use request queue for rate limiting
            cache_expiry_days: Override cache expiry days
            
        Returns:
            Tuple of (scraped content data, from_cache: bool)
            
        Raises:
            SerperAPIError: For API errors
        """
        if not self.html_cache_service:
            # No cache service available, fall back to regular scraping
            logger.info(f"No cache service available, performing direct scrape for: {url}")
            content = await self.scrape(url, extract_text, extract_markdown, use_queue)
            return content, False
        
        try:
            # Define fetch callback for cache service
            async def fetch_callback(url: str) -> str:
                """Fetch content using Serper API and return as HTML-like string"""
                content_data = await self.scrape(url, extract_text, extract_markdown, use_queue)
                
                # Convert the scraped data to HTML-like content for caching
                # We'll use markdown content if available, otherwise text
                html_content = content_data.get('markdown', '') or content_data.get('text', '')
                
                if not html_content:
                    logger.warning(f"No content extracted for caching from URL: {url}")
                    # Return a minimal representation with available data
                    return json.dumps({
                        'title': content_data.get('title', ''),
                        'url': url,
                        'content': content_data.get('text', ''),
                        'metadata': {k: v for k, v in content_data.items() if k not in ['text', 'markdown']}
                    })
                
                # Return the content for caching
                return html_content
            
            # Use cache service to get or fetch content
            cached_content, from_cache = await self.html_cache_service.get_or_fetch_content(
                url, fetch_callback, cache_expiry_days
            )
            
            # Convert cached content back to Serper API format
            if from_cache:
                logger.info(f"Using cached content for URL: {url} (access count: {cached_content.access_count})")
                
                # Try to parse as JSON first (for structured cached content)
                try:
                    parsed_content = json.loads(cached_content.html_content)
                    if isinstance(parsed_content, dict):
                        return parsed_content, True
                except (json.JSONDecodeError, TypeError):
                    pass
                
                # Return as markdown/text content
                return {
                    'url': url,
                    'title': f'Cached content for {url}',
                    'text': cached_content.html_content,
                    'markdown': cached_content.html_content,
                    'cached': True,
                    'cache_metadata': {
                        'access_count': cached_content.access_count,
                        'last_accessed': cached_content.last_accessed.isoformat(),
                        'retrieval_timestamp': cached_content.retrieval_timestamp.isoformat()
                    }
                }, True
            else:
                logger.info(f"Fetched new content for URL: {url}")
                # Content was freshly fetched, parse from the fetch callback result
                try:
                    parsed_content = json.loads(cached_content.html_content)
                    if isinstance(parsed_content, dict):
                        return parsed_content, False
                except (json.JSONDecodeError, TypeError):
                    pass
                
                # Return as text content
                return {
                    'url': url,
                    'title': f'Fresh content for {url}',
                    'text': cached_content.html_content,
                    'markdown': cached_content.html_content,
                    'cached': False
                }, False
                
        except Exception as e:
            logger.error(f"Cache-aware scraping failed for URL '{url}': {e}")
            logger.info(f"Falling back to direct scraping for URL: {url}")
            
            # Fallback to regular scraping
            try:
                content = await self.scrape(url, extract_text, extract_markdown, use_queue)
                return content, False
            except Exception as fallback_e:
                logger.error(f"Fallback scraping also failed for URL '{url}': {fallback_e}")
                raise SerperAPIError(f"Both cache-aware and fallback scraping failed: {fallback_e}")
    
    async def batch_search(self, search_requests: List[Dict[str, Any]],
                          delay_between_requests: float = 1.0) -> List[Dict[str, Any]]:
        """
        Perform batch search operations
        
        Args:
            search_requests: List of request dictionaries with format:
                {'q': 'query', 'type': 'search'/'scrape', 'engine': 'google', 'url': '...' (for scrape)}
            delay_between_requests: Delay between requests in seconds
                
        Returns:
            List of results corresponding to each request
        """
        if not search_requests:
            return []
        
        results = []
        
        for i, request in enumerate(search_requests):
            try:
                request_type = request.get('type', 'search')
                
                if request_type == 'search':
                    query = request.get('q', '')
                    if not query:
                        logger.warning(f"Empty query in batch request: {request}")
                        results.append({'success': False, 'error': 'Empty query'})
                        continue
                    
                    # Perform search
                    search_result = await self.search(
                        query=query,
                        num_results=request.get('num', 10),
                        country=request.get('gl', 'us'),
                        location=request.get('location', ''),
                        language=request.get('hl', 'en')
                    )
                    
                    results.append({
                        'success': True,
                        'data': search_result,
                        'request': request
                    })
                    
                elif request_type == 'scrape':
                    url = request.get('url', '')
                    if not url:
                        logger.warning(f"Empty URL in batch scrape request: {request}")
                        results.append({'success': False, 'error': 'Empty URL'})
                        continue
                    
                    # Perform scraping
                    scrape_result = await self.scrape(
                        url=url,
                        extract_text=request.get('extractText', True),
                        extract_markdown=request.get('extractMarkdown', True)
                    )
                    
                    results.append({
                        'success': True,
                        'data': scrape_result,
                        'request': request
                    })
                    
                else:
                    logger.warning(f"Unknown request type in batch: {request_type}")
                    results.append({'success': False, 'error': f'Unknown request type: {request_type}'})
                
                # Add delay between requests (except after the last one)
                if i < len(search_requests) - 1:
                    await asyncio.sleep(delay_between_requests)
                    
            except Exception as e:
                logger.error(f"Batch search failed for request '{request}': {e}")
                results.append({
                    'success': False,
                    'error': str(e),
                    'request': request
                })
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive client usage and performance statistics
        
        Returns:
            Dictionary with detailed usage stats
        """
        # Calculate response time statistics
        avg_response_time = 0.0
        min_response_time = 0.0
        max_response_time = 0.0
        
        if self.response_times:
            avg_response_time = sum(self.response_times) / len(self.response_times)
            min_response_time = min(self.response_times)
            max_response_time = max(self.response_times)
        
        # Calculate request rate (requests per second)
        current_time = time.time()
        recent_requests = [ts for ts in self.request_timestamps if current_time - ts <= 60.0]  # Last minute
        requests_per_minute = len(recent_requests)
        
        return {
            # Basic metrics
            'request_count': self.request_count,
            'error_count': self.error_count,
            'error_rate': round(self.error_count / max(self.request_count, 1), 3),
            'last_request_time': self.last_request_time.isoformat() if self.last_request_time else None,
            'api_key_configured': bool(self.api_key),
            
            # Performance metrics
            'avg_response_time_seconds': round(avg_response_time, 3),
            'min_response_time_seconds': round(min_response_time, 3),
            'max_response_time_seconds': round(max_response_time, 3),
            'response_time_samples': len(self.response_times),
            
            # Rate limiting metrics
            'requests_per_second_limit': self.requests_per_second,
            'requests_last_minute': requests_per_minute,
            'current_request_rate': round(requests_per_minute / 60.0, 2),
            
            # Concurrency metrics
            'max_concurrent_requests': self.max_concurrent_requests,
            'active_requests': self.active_requests,
            'queue_size': self.request_queue.qsize(),
            'queue_processor_running': self._queue_processor_running,
            
            # System health
            'within_rate_limits': len(recent_requests) <= (self.requests_per_second * 60),
            'system_healthy': (self.error_count / max(self.request_count, 1)) < 0.1
        }
    
    def reset_stats(self):
        """Reset usage statistics"""
        self.request_count = 0
        self.error_count = 0
        self.last_request_time = None
