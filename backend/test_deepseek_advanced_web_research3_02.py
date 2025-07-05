#!/usr/bin/env python3
"""
Enhanced DeepSeek Web Research with MongoDB Caching and Multi-Query Strategy v3.02
This script implements an advanced research workflow with MongoDB caching:
1. Multi-angle search query generation
2. MongoDB caching for scraped web content
3. Content deduplication and smart caching
4. Enhanced content filtering and source diversification
5. Iterative query refinement based on gaps
6. Comprehensive logging and performance analysis

New Features in v3.02:
- MongoDB integration for caching scraped web content
- Smart URL matching to avoid duplicate scraping
- Keywords tracking for better cache management
- Content freshness checking

Usage:
    python test_deepseek_advanced_web_research3_02.py

Environment variables:
    DEEPSEEK_API_KEY: Your DeepSeek API key
    DEEPSEEK_API_URL: The base URL for the DeepSeek API (default: https://api.deepseek.com)
    GOOGLE_API_KEY: Your Google Custom Search API key
    GOOGLE_CSE_ID: Your Google Custom Search Engine ID
    MONGODB_URI: MongoDB connection string (default: mongodb://localhost:27017)
"""

import os
import sys
import json
import asyncio
import logging
import requests
import re
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import AsyncOpenAI
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass
from collections import defaultdict

# Web scraping libraries
try:
    from bs4 import BeautifulSoup
    import newspaper
    from newspaper import Article
    from readability import Document
except ImportError as e:
    print(f"❌ Missing required libraries: {e}")
    print("📦 Please install: pip install beautifulsoup4 lxml newspaper3k readability-lxml")
    sys.exit(1)

# MongoDB libraries
try:
    from pymongo import MongoClient
    from motor.motor_asyncio import AsyncIOMotorClient
except ImportError as e:
    print(f"❌ Missing MongoDB libraries: {e}")
    print("📦 Please install: pip install pymongo motor")
    sys.exit(1)

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='deepseek_enhanced_research_v302.log',
    filemode='a'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger('').addHandler(console)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@dataclass
class TimingMetrics:
    """Track timing metrics for performance analysis"""
    start_time: float
    end_time: Optional[float] = None
    phase_times: Dict[str, float] = None
    
    def __post_init__(self):
        if self.phase_times is None:
            self.phase_times = {}
    
    def start_phase(self, phase_name: str):
        """Start timing a phase"""
        self.phase_times[f"{phase_name}_start"] = time.time()
        logger.info(f"⏱️ Starting phase: {phase_name}")
    
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
    """Track search and analysis metrics"""
    total_queries: int = 0
    total_results: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    source_types: Dict[str, int] = None
    relevance_scores: List[float] = None
    
    def __post_init__(self):
        if self.source_types is None:
            self.source_types = defaultdict(int)
        if self.relevance_scores is None:
            self.relevance_scores = []

class MongoDBCacheService:
    """MongoDB service for caching scraped web content"""
    
    def __init__(self, mongodb_uri: str = None):
        self.mongodb_uri = mongodb_uri or os.environ.get('MONGODB_URI', 'mongodb://localhost:27017')
        self.client = None
        self.db = None
        self.collection = None
        self.is_connected = False
        
    async def connect(self):
        """Connect to MongoDB"""
        try:
            self.client = AsyncIOMotorClient(self.mongodb_uri)
            self.db = self.client.web_research_cache
            self.collection = self.db.scraped_content
            
            # Test connection
            await self.client.admin.command('ping')
            self.is_connected = True
            
            # Create indexes for better performance
            await self.collection.create_index("url", unique=True)
            await self.collection.create_index("keywords")
            await self.collection.create_index("accessed_date")
            
            logger.info("✅ Connected to MongoDB cache")
            
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            self.is_connected = False
    
    async def search_cached_content(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Search for cached content by keywords"""
        if not self.is_connected:
            return []
        
        try:
            # Build search query for keywords
            query = {
                "$or": [
                    {"keywords": {"$in": keywords}},
                    {"title": {"$regex": "|".join(keywords), "$options": "i"}},
                    {"content": {"$regex": "|".join(keywords), "$options": "i"}}
                ]
            }
            
            cursor = self.collection.find(query)
            cached_results = []
            
            async for doc in cursor:
                # Check if content is still fresh (within 7 days)
                accessed_date = doc.get('accessed_date')
                if isinstance(accessed_date, datetime):
                    days_old = (datetime.utcnow() - accessed_date).days
                    if days_old <= 7:  # Content is still fresh
                        cached_results.append(doc)
                        logger.info(f"🔍 Found cached content: {doc['url']} ({days_old} days old)")
            
            logger.info(f"📊 Found {len(cached_results)} cached results for keywords: {keywords}")
            return cached_results
            
        except Exception as e:
            logger.error(f"❌ Cache search failed: {e}")
            return []
    
    async def get_cached_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Get cached content by URL"""
        if not self.is_connected:
            return None
        
        try:
            result = await self.collection.find_one({"url": url})
            if result:
                # Check freshness (within 7 days)
                accessed_date = result.get('accessed_date')
                if isinstance(accessed_date, datetime):
                    days_old = (datetime.utcnow() - accessed_date).days
                    if days_old <= 7:
                        logger.info(f"💾 Cache hit: {url} ({days_old} days old)")
                        return result
                    else:
                        logger.info(f"⏰ Cache expired: {url} ({days_old} days old)")
                        # Remove expired cache
                        await self.collection.delete_one({"url": url})
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Cache retrieval failed for {url}: {e}")
            return None
    
    async def save_content(self, url: str, title: str, content: str, keywords: List[str], 
                          method: str, word_count: int, domain_info: Dict[str, Any]) -> bool:
        """Save scraped content to cache"""
        if not self.is_connected:
            return False
        
        try:
            document = {
                "url": url,
                "title": title,
                "content": content,
                "keywords": keywords,
                "accessed_date": datetime.utcnow(),
                "method": method,
                "word_count": word_count,
                "domain_info": domain_info,
                "created_at": datetime.utcnow()
            }
            
            # Use upsert to handle duplicates
            await self.collection.replace_one(
                {"url": url}, 
                document, 
                upsert=True
            )
            
            logger.info(f"💾 Cached content: {url} ({word_count} words)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Cache save failed for {url}: {e}")
            return False
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.is_connected:
            return {"total_entries": 0, "connected": False}
        
        try:
            total_entries = await self.collection.count_documents({})
            fresh_entries = await self.collection.count_documents({
                "accessed_date": {"$gte": datetime.utcnow() - timedelta(days=7)}
            })
            
            return {
                "total_entries": total_entries,
                "fresh_entries": fresh_entries,
                "connected": True
            }
            
        except Exception as e:
            logger.error(f"❌ Cache stats failed: {e}")
            return {"total_entries": 0, "connected": False, "error": str(e)}
    
    async def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("🔌 MongoDB connection closed")

class EnhancedWebContentExtractor:
    """Enhanced service for extracting content from web pages with MongoDB caching"""
    
    def __init__(self, cache_service: MongoDBCacheService):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.domain_quality_cache = {}
        self.cache_service = cache_service
    
    def assess_domain_quality(self, url: str) -> Dict[str, Any]:
        """Assess the quality and type of a domain for content extraction"""
        try:
            domain = urlparse(url).netloc.lower()
            
            # Check cache first
            if domain in self.domain_quality_cache:
                return self.domain_quality_cache[domain]
            
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
            
            quality_score = 5  # Default medium quality
            source_type = "general"
            
            # Assess quality
            if any(indicator in domain for indicator in high_quality_indicators):
                quality_score = 9
                source_type = "premium"
            elif any(indicator in domain for indicator in medium_quality_indicators):
                quality_score = 7
                source_type = "reliable"
            elif any(indicator in domain for indicator in low_quality_indicators):
                quality_score = 3
                source_type = "social"
            elif domain.endswith('.edu') or domain.endswith('.gov'):
                quality_score = 8
                source_type = "academic"
            elif domain.endswith('.org'):
                quality_score = 6
                source_type = "organization"
            
            # Company websites (ending in .com with short domains)
            if domain.endswith('.com') and len(domain.split('.')) == 2 and len(domain.split('.')[0]) < 15:
                quality_score = max(quality_score, 6)
                source_type = "corporate"
            
            result = {
                'domain': domain,
                'quality_score': quality_score,
                'source_type': source_type,
                'recommended': quality_score >= 6
            }
            
            self.domain_quality_cache[domain] = result
            logger.info(f"🔍 Domain assessment: {domain} - Quality: {quality_score}/10, Type: {source_type}")
            return result
            
        except Exception as e:
            logger.warning(f"⚠️ Domain assessment failed for {url}: {e}")
            return {'domain': 'unknown', 'quality_score': 5, 'source_type': 'unknown', 'recommended': True}
    
    async def extract_article_content(self, url: str, keywords: List[str] = None) -> Dict[str, Any]:
        """Extract article content with MongoDB caching support"""
        start_time = time.time()
        
        try:
            logger.info(f"🔄 Extracting content from: {url}")
            
            # Check cache first
            cached_content = await self.cache_service.get_cached_content(url)
            if cached_content:
                extraction_time = time.time() - start_time
                logger.info(f"💾 Using cached content: {len(cached_content.get('content', ''))} chars in {extraction_time:.2f}s")
                
                return {
                    'url': url,
                    'title': cached_content.get('title', 'No title'),
                    'content': cached_content.get('content', ''),
                    'method': f"cache_{cached_content.get('method', 'unknown')}",
                    'word_count': cached_content.get('word_count', 0),
                    'extraction_time': extraction_time,
                    'domain_info': cached_content.get('domain_info', {}),
                    'success': True,
                    'from_cache': True,
                    'cache_date': cached_content.get('accessed_date')
                }
            
            # Assess domain quality
            domain_info = self.assess_domain_quality(url)
            
            # Extract content using various methods
            extraction_result = await self._extract_content_methods(url, domain_info)
            
            # Save to cache if extraction was successful
            if extraction_result['success'] and keywords:
                await self.cache_service.save_content(
                    url=url,
                    title=extraction_result['title'],
                    content=extraction_result['content'],
                    keywords=keywords,
                    method=extraction_result['method'],
                    word_count=extraction_result['word_count'],
                    domain_info=domain_info
                )
            
            extraction_result['from_cache'] = False
            return extraction_result
            
        except Exception as e:
            extraction_time = time.time() - start_time
            logger.error(f"❌ Content extraction failed for {url}: {e} in {extraction_time:.2f}s")
            return {
                'url': url,
                'title': 'Error',
                'content': f'Error extracting content: {str(e)}',
                'method': 'error',
                'word_count': 0,
                'extraction_time': extraction_time,
                'domain_info': {'quality_score': 0, 'source_type': 'error'},
                'success': False,
                'error': str(e),
                'from_cache': False
            }
    
    async def _extract_content_methods(self, url: str, domain_info: Dict[str, Any]) -> Dict[str, Any]:
        """Try different content extraction methods"""
        start_time = time.time()
        
        # Method 1: Try newspaper3k first (best for articles)
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            if article.text and len(article.text.strip()) > 100:
                extraction_time = time.time() - start_time
                logger.info(f"✅ Successful extraction via newspaper3k: {len(article.text)} chars in {extraction_time:.2f}s")
                
                return {
                    'url': url,
                    'title': article.title or 'No title',
                    'content': article.text,
                    'method': 'newspaper3k',
                    'word_count': len(article.text.split()),
                    'extraction_time': extraction_time,
                    'domain_info': domain_info,
                    'success': True
                }
        except Exception as e:
            logger.warning(f"⚠️ Newspaper3k failed for {url}: {e}")
        
        # Method 2: Try readability + BeautifulSoup
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            doc = Document(response.text)
            soup = BeautifulSoup(doc.content(), 'html.parser')
            
            content = soup.get_text(separator=' ', strip=True)
            title = doc.title() or soup.find('title')
            title = title.get_text() if hasattr(title, 'get_text') else str(title) if title else 'No title'
            
            if content and len(content.strip()) > 100:
                extraction_time = time.time() - start_time
                logger.info(f"✅ Successful extraction via readability: {len(content)} chars in {extraction_time:.2f}s")
                
                return {
                    'url': url,
                    'title': title,
                    'content': content,
                    'method': 'readability+bs4',
                    'word_count': len(content.split()),
                    'extraction_time': extraction_time,
                    'domain_info': domain_info,
                    'success': True
                }
        except Exception as e:
            logger.warning(f"⚠️ Readability method failed for {url}: {e}")
        
        # Method 3: Basic BeautifulSoup fallback
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for script in soup(["script", "style"]):
                script.decompose()
            
            content_selectors = [
                'article', 'main', '.content', '.article-content', 
                '.post-content', '.entry-content', '#content'
            ]
            
            content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content = ' '.join([elem.get_text(separator=' ', strip=True) for elem in elements])
                    break
            
            if not content:
                body = soup.find('body')
                if body:
                    content = body.get_text(separator=' ', strip=True)
            
            title = soup.find('title')
            title = title.get_text() if title else 'No title'
            
            if content and len(content.strip()) > 50:
                extraction_time = time.time() - start_time
                logger.info(f"✅ Successful extraction via BeautifulSoup: {len(content)} chars in {extraction_time:.2f}s")
                
                return {
                    'url': url,
                    'title': title,
                    'content': content[:5000],  # Limit content length
                    'method': 'beautifulsoup',
                    'word_count': len(content.split()),
                    'extraction_time': extraction_time,
                    'domain_info': domain_info,
                    'success': True
                }
        except Exception as e:
            logger.warning(f"⚠️ BeautifulSoup method failed for {url}: {e}")
        
        # All methods failed
        extraction_time = time.time() - start_time
        logger.error(f"❌ All extraction methods failed for {url} in {extraction_time:.2f}s")
        
        return {
            'url': url,
            'title': 'Extraction failed',
            'content': 'Could not extract content from this URL',
            'method': 'none',
            'word_count': 0,
            'extraction_time': extraction_time,
            'domain_info': domain_info,
            'success': False,
            'error': 'All extraction methods failed'
        }

class EnhancedGoogleWebSearchService:
    """Enhanced Google web search service with filtering and source diversification"""
    
    def __init__(self):
        self.api_key = os.environ.get('GOOGLE_API_KEY', '')
        self.cse_id = os.environ.get('GOOGLE_CSE_ID', '')
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        self.search_cache = {}  # Cache to avoid duplicate searches
        
        if not self.api_key:
            logger.warning("⚠️ GOOGLE_API_KEY not set. Web search functionality will be limited.")
        if not self.cse_id:
            logger.warning("⚠️ GOOGLE_CSE_ID not set. Web search functionality will be limited.")
    
    async def search_with_filters(self, query: str, num_results: int = 10, 
                                 exclude_domains: List[str] = None,
                                 prefer_domains: List[str] = None) -> List[Dict[str, Any]]:
        """Enhanced search with domain filtering"""
        if not self.api_key or not self.cse_id:
            logger.error("❌ Google API credentials not configured")
            return []
        
        # Check cache
        cache_key = f"{query}_{num_results}"
        if cache_key in self.search_cache:
            logger.info(f"🔄 Using cached results for: {query}")
            return self.search_cache[cache_key]
        
        try:
            # Build query with domain filters
            modified_query = query
            if exclude_domains:
                for domain in exclude_domains:
                    modified_query += f" -site:{domain}"
            
            params = {
                'key': self.api_key,
                'cx': self.cse_id,
                'q': modified_query,
                'num': min(num_results, 10),
                'safe': 'active'
            }
            
            logger.info(f"🔍 Google search: {modified_query}")
            start_time = time.time()
            
            response = requests.get(self.base_url, params=params, timeout=15)
            response.raise_for_status()
            
            search_time = time.time() - start_time
            data = response.json()
            results = []
            
            if 'items' in data:
                for item in data['items']:
                    result = {
                        'title': item.get('title', ''),
                        'link': item.get('link', ''),
                        'snippet': item.get('snippet', ''),
                        'displayLink': item.get('displayLink', ''),
                        'search_query': query
                    }
                    results.append(result)
            
            # Cache results
            self.search_cache[cache_key] = results
            
            logger.info(f"✅ Found {len(results)} search results in {search_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"❌ Google search failed: {e}")
            return []

class EnhancedDeepSeekResearchService:
    """Enhanced research service with MongoDB caching and multi-query strategy"""
    
    def __init__(self):
        self.api_key = os.environ.get('DEEPSEEK_API_KEY', '')
        self.api_url = os.environ.get('DEEPSEEK_API_URL', 'https://api.deepseek.com')
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.api_url)
        self.web_search = EnhancedGoogleWebSearchService()
        self.cache_service = MongoDBCacheService()
        self.content_extractor = EnhancedWebContentExtractor(self.cache_service)
        self.metrics = SearchMetrics()
        
        if not self.api_key:
            logger.error("❌ DEEPSEEK_API_KEY not set")
            raise ValueError("DEEPSEEK_API_KEY environment variable is required")
    
    async def initialize(self):
        """Initialize MongoDB connection"""
        await self.cache_service.connect()
        
        # Display cache stats
        cache_stats = await self.cache_service.get_cache_stats()
        logger.info(f"📊 Cache stats: {cache_stats['total_entries']} total entries, {cache_stats.get('fresh_entries', 0)} fresh")
    
    async def search_existing_cache(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Search for existing cached content based on queries"""
        logger.info(f"🔍 Searching cache for {len(queries)} queries")
        
        # Extract keywords from all queries
        all_keywords = []
        for query in queries:
            # Simple keyword extraction - split and clean
            keywords = [word.strip().lower() for word in re.split(r'[^\w]+', query) if len(word.strip()) > 2]
            all_keywords.extend(keywords)
        
        # Remove duplicates while preserving order
        unique_keywords = list(dict.fromkeys(all_keywords))
        
        cached_results = await self.cache_service.search_cached_content(unique_keywords)
        
        if cached_results:
            logger.info(f"💾 Found {len(cached_results)} cached results")
            self.metrics.cache_hits += len(cached_results)
            
            # Convert cached results to extraction format
            extracted_contents = []
            for cached in cached_results:
                content = {
                    'url': cached['url'],
                    'title': cached.get('title', 'No title'),
                    'content': cached.get('content', ''),
                    'method': f"cache_{cached.get('method', 'unknown')}",
                    'word_count': cached.get('word_count', 0),
                    'extraction_time': 0.0,  # Instant from cache
                    'domain_info': cached.get('domain_info', {}),
                    'success': True,
                    'from_cache': True,
                    'cache_date': cached.get('accessed_date'),
                    'search_result': {
                        'title': cached.get('title', 'No title'),
                        'link': cached['url'],
                        'snippet': cached.get('content', '')[:200] + '...',
                        'displayLink': urlparse(cached['url']).netloc,
                        'search_query': 'cached_result'
                    }
                }
                extracted_contents.append(content)
            
            return extracted_contents
        else:
            logger.info("📭 No cached results found")
            return []
    
    async def generate_multi_angle_queries(self, original_question: str) -> List[str]:
        """Generate multiple search queries from different angles"""
        start_time = time.time()
        
        try:
            logger.info("🎯 Generating multi-angle search queries")
            
            system_message = """You are an expert research strategist. Generate 4-5 different search queries to comprehensively research a topic from multiple angles.

For business research questions, consider these angles:
1. Company/product names and direct information
2. Market analysis and industry reports  
3. Financial data and revenue information
4. Competitive analysis and rankings
5. Regional/geographic specific information

Instructions:
1. Analyze the question to identify key aspects
2. Generate 4-5 distinct search queries covering different angles
3. Make queries specific and targeted
4. Include relevant industry terms and modifiers
5. Format response as: Query1="...", Query2="...", Query3="...", etc.
6. Ensure queries complement each other without too much overlap"""

            user_prompt = f"""Original Question: {original_question}

Generate 4-5 comprehensive search queries that approach this question from different angles. Focus on finding authoritative, data-rich sources."""

            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                stream=False,
                timeout=30.0
            )
            
            response_text = response.choices[0].message.content
            generation_time = time.time() - start_time
            
            # Extract queries from response
            queries = []
            query_pattern = r'Query\d*="([^"]+)"'
            matches = re.findall(query_pattern, response_text)
            
            if matches:
                queries = matches
                logger.info(f"✅ Generated {len(queries)} search queries in {generation_time:.2f}s")
                for i, query in enumerate(queries, 1):
                    logger.info(f"  {i}. {query}")
            else:
                # Fallback: use original question
                queries = [original_question]
                logger.warning("⚠️ Could not extract queries, using original question")
            
            self.metrics.total_queries += len(queries)
            return queries
                
        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(f"❌ Query generation failed in {generation_time:.2f}s: {e}")
            return [original_question]
    
    async def perform_comprehensive_search(self, queries: List[str], max_results_per_query: int = 5) -> List[Dict[str, Any]]:
        """Perform comprehensive search across multiple queries"""
        start_time = time.time()
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
            results = await self.web_search.search_with_filters(
                query, 
                num_results=max_results_per_query,
                exclude_domains=exclude_domains
            )
            query_time = time.time() - query_start
            
            # Filter out duplicate URLs
            new_results = []
            for result in results:
                if result['link'] not in seen_urls:
                    seen_urls.add(result['link'])
                    new_results.append(result)
            
            all_results.extend(new_results)
            logger.info(f"  ✅ Query {i} completed: {len(new_results)} new results in {query_time:.2f}s")
            
            # Add delay between searches
            await asyncio.sleep(1)
        
        total_time = time.time() - start_time
        self.metrics.total_results = len(all_results)
        
        logger.info(f"✅ Comprehensive search completed: {len(all_results)} unique results in {total_time:.2f}s")
        return all_results
    
    async def extract_and_analyze_content(self, search_results: List[Dict[str, Any]], queries: List[str]) -> List[Dict[str, Any]]:
        """Extract content with MongoDB caching and quality filtering"""
        start_time = time.time()
        
        logger.info(f"📄 Extracting content from {len(search_results)} sources")
        
        # Extract keywords for caching
        keywords = []
        for query in queries:
            query_keywords = [word.strip().lower() for word in re.split(r'[^\w]+', query) if len(word.strip()) > 2]
            keywords.extend(query_keywords)
        keywords = list(dict.fromkeys(keywords))  # Remove duplicates
        
        # Sort by domain quality for better extraction order
        def get_quality_score(result):
            domain_info = self.content_extractor.assess_domain_quality(result['link'])
            return domain_info['quality_score']
        
        sorted_results = sorted(search_results, key=get_quality_score, reverse=True)
        
        extracted_contents = []
        source_type_counts = defaultdict(int)
        
        for i, result in enumerate(sorted_results):
            logger.info(f"📄 Extracting {i+1}/{len(sorted_results)}: {result['displayLink']}")
            
            content = await self.content_extractor.extract_article_content(result['link'], keywords)
            content['search_result'] = result
            
            # Track cache hits/misses
            if content.get('from_cache', False):
                self.metrics.cache_hits += 1
            else:
                self.metrics.cache_misses += 1
            
            # Track source types for diversification
            source_type = content.get('domain_info', {}).get('source_type', 'unknown')
            source_type_counts[source_type] += 1
            self.metrics.source_types[source_type] += 1
            
            if content['success']:
                self.metrics.successful_extractions += 1
                cache_status = "💾 Cache" if content.get('from_cache') else "🌐 Live"
                logger.info(f"  ✅ Success ({cache_status}): {content['word_count']} words, quality: {content.get('domain_info', {}).get('quality_score', 0)}/10")
            else:
                self.metrics.failed_extractions += 1
                logger.warning(f"  ❌ Failed: {content.get('error', 'Unknown error')}")
            
            extracted_contents.append(content)
            
            # Add delay only for live scraping (not cached content)
            if not content.get('from_cache', False):
                await asyncio.sleep(1)
        
        extraction_time = time.time() - start_time
        logger.info(f"✅ Content extraction completed in {extraction_time:.2f}s")
        logger.info(f"💾 Cache performance: {self.metrics.cache_hits} hits, {self.metrics.cache_misses} misses")
        logger.info(f"📊 Source distribution: {dict(source_type_counts)}")
        
        return extracted_contents
    
    async def analyze_content_with_gaps(self, original_question: str, extracted_contents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhanced analysis with gap identification and relevance scoring"""
        start_time = time.time()
        
        logger.info("🧠 Starting comprehensive content analysis")
        
        try:
            # Prepare content for analysis with quality indicators
            content_summaries = []
            for i, content in enumerate(extracted_contents):
                if content['success']:
                    domain_info = content.get('domain_info', {})
                    quality_score = domain_info.get('quality_score', 5)
                    source_type = domain_info.get('source_type', 'unknown')
                    cache_status = "CACHED" if content.get('from_cache') else "LIVE"
                    
                    summary = f"""
Source {i+1}: {content['title']} [{cache_status}]
URL: {content['url']}
Quality Score: {quality_score}/10
Source Type: {source_type}
Extraction Method: {content['method']}
Content Preview: {content['content'][:1500]}...
Word Count: {content['word_count']}
"""
                else:
                    summary = f"""
Source {i+1}: {content['title']} (Extraction Failed)
URL: {content['url']}
Error: {content.get('error', 'Unknown error')}
"""
                content_summaries.append(summary)
            
            system_message = """You are an expert research analyst with advanced reasoning capabilities. Analyze extracted web content for relevance and identify any gaps in information.

Your Analysis Tasks:
1. RELEVANCE ASSESSMENT: For each source, provide:
   - Individual relevance score (1-10)
   - Key insights that answer the original question
   - Data quality assessment (factual, recent, authoritative)
   - Any limitations or concerns

2. SYNTHESIS: Combine information across sources to:
   - Answer the original question comprehensively
   - Identify patterns and trends
   - Resolve conflicts between sources
   - Highlight the most reliable findings

3. GAP ANALYSIS: Identify what information is missing:
   - Key aspects of the question not fully addressed
   - Data that would strengthen the analysis
   - Sources that would provide better coverage

4. QUALITY ASSESSMENT: Evaluate the overall research quality:
   - Source diversity and authority
   - Data completeness and accuracy
   - Timeliness of information

CRITICAL: End your response with "OVERALL_RELEVANCE_SCORE: X" where X (1-10) represents how well all sources combined answer the original question.
- Score 9-10: Comprehensive answer with authoritative sources
- Score 7-8: Good answer with solid sources, minor gaps
- Score 5-6: Partial answer, significant gaps or quality issues  
- Score 3-4: Limited answer, major gaps or unreliable sources
- Score 1-2: Poor answer, mostly irrelevant or unreliable information"""

            user_prompt = f"""Original Research Question: {original_question}

Extracted Content from {len(extracted_contents)} Web Sources:
{''.join(content_summaries)}

Please provide a comprehensive analysis including:
1. Individual source relevance scores and assessment
2. Synthesized answer to the original question
3. Gap analysis - what's missing
4. Overall quality assessment
5. Your final overall relevance score (1-10)"""

            print(f"\n🧠 Starting DeepSeek Reasoning Analysis...")
            print("🔄 [REASONING] Analyzing content relevance and gaps...", flush=True)
            
            analysis_start = time.time()
            
            response = await self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                stream=True,
                timeout=90.0
            )
            
            # Process streaming response with enhanced logging
            reasoning_content = ""
            analysis_content = ""
            reasoning_buffer = ""
            analysis_buffer = ""
            
            async for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    delta = choice.delta
                    
                    # Handle reasoning content streaming
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        reasoning_chunk = delta.reasoning_content
                        reasoning_content += reasoning_chunk
                        reasoning_buffer += reasoning_chunk
                        
                        if len(reasoning_buffer) > 100:
                            print(f"\n🔄 [REASONING] ...{reasoning_buffer[-50:]}", end="", flush=True)
                            reasoning_buffer = ""
                    
                    # Handle regular content streaming
                    if hasattr(delta, 'content') and delta.content:
                        content_chunk = delta.content
                        analysis_content += content_chunk
                        analysis_buffer += content_chunk
                        
                        if len(analysis_buffer) > 100:
                            print(f"\n🔄 [ANALYSIS] ...{analysis_buffer[-50:]}", end="", flush=True)
                            analysis_buffer = ""
            
            # Display remaining content
            if reasoning_buffer:
                print(f"\n🔄 [REASONING] ...{reasoning_buffer}", end="", flush=True)
            if analysis_buffer:
                print(f"\n🔄 [ANALYSIS] ...{analysis_buffer}", end="", flush=True)
            
            analysis_time = time.time() - analysis_start
            total_time = time.time() - start_time
            
            print(f"\n✅ Analysis completed in {analysis_time:.2f}s")
            
            # Extract overall relevance score
            overall_relevance_score = 0
            score_match = re.search(r'OVERALL_RELEVANCE_SCORE:\s*(\d+)', analysis_content)
            if score_match:
                overall_relevance_score = int(score_match.group(1))
                logger.info(f"📊 Extracted relevance score: {overall_relevance_score}/10")
                print(f"📊 Final relevance score: {overall_relevance_score}/10")
            else:
                logger.warning("⚠️ Could not extract overall relevance score")
                print("⚠️ Warning: Could not extract relevance score")
            
            self.metrics.relevance_scores.append(overall_relevance_score)
            
            return {
                'original_question': original_question,
                'analysis_content': analysis_content,
                'reasoning_content': reasoning_content,
                'overall_relevance_score': overall_relevance_score,
                'sources_analyzed': len(extracted_contents),
                'successful_extractions': self.metrics.successful_extractions,
                'cache_hits': self.metrics.cache_hits,
                'cache_misses': self.metrics.cache_misses,
                'analysis_time': analysis_time,
                'total_analysis_time': total_time,
                'model': 'deepseek-reasoner',
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            analysis_time = time.time() - start_time
            logger.error(f"❌ Analysis failed in {analysis_time:.2f}s: {e}")
            return {
                'original_question': original_question,
                'error': str(e),
                'analysis_time': analysis_time,
                'cache_hits': self.metrics.cache_hits,
                'cache_misses': self.metrics.cache_misses,
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def conduct_enhanced_research(self, original_question: str, target_relevance: int = 7, max_iterations: int = 3) -> Dict[str, Any]:
        """Conduct comprehensive research with MongoDB caching and iterative improvement"""
        timing = TimingMetrics(start_time=time.time())
        
        logger.info(f"🚀 Starting enhanced research with MongoDB caching for: {original_question}")
        logger.info(f"🎯 Target relevance score: {target_relevance}/10")
        
        results = {
            'original_question': original_question,
            'timestamp': datetime.utcnow().isoformat(),
            'research_type': 'enhanced_multi_query_with_mongodb',
            'target_relevance': target_relevance,
            'iterations': [],
            'final_metrics': {}
        }
        
        current_relevance = 0
        iteration = 0
        all_extracted_contents = []
        
        try:
            while current_relevance < target_relevance and iteration < max_iterations:
                iteration += 1
                logger.info(f"🔄 Starting iteration {iteration}/{max_iterations}")
                
                iteration_results = {
                    'iteration': iteration,
                    'steps': {},
                    'relevance_achieved': 0
                }
                
                # Step 1: Generate multi-angle queries
                timing.start_phase(f'query_generation_iter{iteration}')
                if iteration == 1:
                    queries = await self.generate_multi_angle_queries(original_question)
                else:
                    # Generate follow-up queries based on gaps identified in previous iteration
                    queries = await self.generate_followup_queries(original_question, results['iterations'][-1])
                timing.end_phase(f'query_generation_iter{iteration}')
                
                iteration_results['steps']['step1'] = {
                    'description': f'Generate queries for iteration {iteration}',
                    'queries': queries,
                    'query_count': len(queries),
                    'time_taken': timing.phase_times.get(f'query_generation_iter{iteration}_duration', 0),
                    'success': True
                }
                
                # Step 1.5: Check MongoDB cache first (only in first iteration)
                cached_content = []
                if iteration == 1:
                    timing.start_phase(f'cache_search_iter{iteration}')
                    cached_content = await self.search_existing_cache(queries)
                    timing.end_phase(f'cache_search_iter{iteration}')
                    
                    iteration_results['steps']['step1_5'] = {
                        'description': f'Search MongoDB cache iteration {iteration}',
                        'cached_results': len(cached_content),
                        'time_taken': timing.phase_times.get(f'cache_search_iter{iteration}_duration', 0),
                        'success': True
                    }
                    
                    if cached_content:
                        logger.info(f"💾 Using {len(cached_content)} cached results from MongoDB")
                        all_extracted_contents.extend(cached_content)
                
                # Step 2: Comprehensive search (if needed)
                timing.start_phase(f'comprehensive_search_iter{iteration}')
                search_results = await self.perform_comprehensive_search(queries, max_results_per_query=6)
                timing.end_phase(f'comprehensive_search_iter{iteration}')
                
                iteration_results['steps']['step2'] = {
                    'description': f'Comprehensive search iteration {iteration}',
                    'search_results': search_results,
                    'total_results': len(search_results),
                    'queries_used': len(queries),
                    'time_taken': timing.phase_times.get(f'comprehensive_search_iter{iteration}_duration', 0),
                    'success': len(search_results) > 0
                }
                
                if not search_results and not cached_content:
                    logger.warning(f"⚠️ No search results or cached content found in iteration {iteration}")
                    iteration_results['warning'] = 'No search results or cached content found'
                    results['iterations'].append(iteration_results)
                    continue
                
                # Step 3: Content extraction and analysis (skip if all from cache)
                new_extracted_contents = []
                if search_results:
                    timing.start_phase(f'content_extraction_iter{iteration}')
                    new_extracted_contents = await self.extract_and_analyze_content(search_results, queries)
                    timing.end_phase(f'content_extraction_iter{iteration}')
                    
                    # Combine with previous contents for comprehensive analysis
                    all_extracted_contents.extend(new_extracted_contents)
                
                iteration_results['steps']['step3'] = {
                    'description': f'Content extraction iteration {iteration}',
                    'new_extractions': len(new_extracted_contents),
                    'cached_extractions': len(cached_content) if iteration == 1 else 0,
                    'total_extractions': len(all_extracted_contents),
                    'successful_new': sum(1 for c in new_extracted_contents if c['success']),
                    'cache_hits': self.metrics.cache_hits,
                    'cache_misses': self.metrics.cache_misses,
                    'time_taken': timing.phase_times.get(f'content_extraction_iter{iteration}_duration', 0),
                    'success': sum(1 for c in new_extracted_contents if c['success']) > 0 or len(cached_content) > 0
                }
                
                # Step 4: Enhanced analysis of all content
                timing.start_phase(f'content_analysis_iter{iteration}')
                analysis = await self.analyze_content_with_gaps(original_question, all_extracted_contents)
                timing.end_phase(f'content_analysis_iter{iteration}')
                
                current_relevance = analysis.get('overall_relevance_score', 0)
                
                iteration_results['steps']['step4'] = {
                    'description': f'Comprehensive analysis iteration {iteration}',
                    'analysis': analysis,
                    'relevance_score': current_relevance,
                    'sources_analyzed': len(all_extracted_contents),
                    'time_taken': timing.phase_times.get(f'content_analysis_iter{iteration}_duration', 0),
                    'success': 'error' not in analysis
                }
                
                iteration_results['relevance_achieved'] = current_relevance
                results['iterations'].append(iteration_results)
                
                logger.info(f"✅ Iteration {iteration} completed: Relevance score {current_relevance}/10")
                
                if current_relevance >= target_relevance:
                    logger.info(f"🎉 Target relevance {target_relevance} achieved with score {current_relevance}!")
                    break
                elif iteration < max_iterations:
                    logger.info(f"🔄 Target not met ({current_relevance} < {target_relevance}), continuing to iteration {iteration + 1}")
            
            # Finalize timing
            timing.end_time = time.time()
            total_duration = timing.get_total_duration()
            
            # Final comprehensive metrics
            final_analysis = results['iterations'][-1]['steps']['step4']['analysis'] if results['iterations'] else {}
            
            # Get final cache stats
            cache_stats = await self.cache_service.get_cache_stats()
            
            results['final_metrics'] = {
                'total_duration': total_duration,
                'iterations_completed': iteration,
                'target_achieved': current_relevance >= target_relevance,
                'final_relevance_score': current_relevance,
                'phase_durations': timing.get_phase_summary(),
                'search_metrics': {
                    'total_queries': self.metrics.total_queries,
                    'total_results': self.metrics.total_results,
                    'total_extractions': len(all_extracted_contents),
                    'successful_extractions': self.metrics.successful_extractions,
                    'failed_extractions': self.metrics.failed_extractions,
                    'extraction_success_rate': (self.metrics.successful_extractions / 
                                               max(1, self.metrics.successful_extractions + self.metrics.failed_extractions)) * 100,
                    'source_distribution': dict(self.metrics.source_types),
                    'relevance_progression': [iter_data['relevance_achieved'] for iter_data in results['iterations']],
                    'cache_performance': {
                        'cache_hits': self.metrics.cache_hits,
                        'cache_misses': self.metrics.cache_misses,
                        'cache_hit_rate': (self.metrics.cache_hits / max(1, self.metrics.cache_hits + self.metrics.cache_misses)) * 100,
                        'total_cache_entries': cache_stats.get('total_entries', 0),
                        'fresh_cache_entries': cache_stats.get('fresh_entries', 0)
                    }
                },
                'performance_analysis': {
                    'avg_time_per_iteration': total_duration / max(1, iteration),
                    'relevance_improvement_rate': (current_relevance - results['iterations'][0]['relevance_achieved']) / max(1, iteration - 1) if iteration > 1 else 0,
                    'final_analysis_summary': final_analysis.get('analysis_content', '')[:500] + '...' if final_analysis.get('analysis_content') else 'No analysis available'
                }
            }
            
            results['success'] = True
            
            # Enhanced logging summary
            logger.info(f"✅ Enhanced research with MongoDB caching completed!")
            logger.info(f"📊 Final relevance score: {current_relevance}/10 (Target: {target_relevance})")
            logger.info(f"🔄 Iterations completed: {iteration}/{max_iterations}")
            logger.info(f"⏱️ Total time: {total_duration:.2f}s")
            logger.info(f"💾 Cache performance: {self.metrics.cache_hits} hits, {self.metrics.cache_misses} misses")
            logger.info(f"🎯 Target achieved: {'Yes' if current_relevance >= target_relevance else 'No'}")
            
        except Exception as e:
            timing.end_time = time.time()
            total_duration = timing.get_total_duration()
            
            logger.error(f"❌ Enhanced research failed after {total_duration:.2f}s: {e}")
            results['error'] = str(e)
            results['success'] = False
            results['final_metrics'] = {
                'total_duration': total_duration,
                'iterations_completed': iteration,
                'phase_durations': timing.get_phase_summary(),
                'cache_performance': {
                    'cache_hits': self.metrics.cache_hits,
                    'cache_misses': self.metrics.cache_misses
                }
            }
        
        return results
    
    async def generate_followup_queries(self, original_question: str, previous_iteration: Dict[str, Any]) -> List[str]:
        """Generate follow-up queries based on gaps identified in previous iteration"""
        start_time = time.time()
        
        try:
            logger.info("🎯 Generating follow-up queries based on identified gaps")
            
            previous_analysis = previous_iteration['steps']['step4']['analysis']
            previous_score = previous_iteration['relevance_achieved']
            
            system_message = """You are an expert research strategist. Based on a previous research iteration and its gaps, generate targeted follow-up search queries to fill the missing information.

Instructions:
1. Analyze the previous analysis to identify specific gaps and weaknesses
2. Generate 3-4 targeted search queries that address these gaps
3. Focus on missing data types, unexplored angles, or contradictory information
4. Make queries specific and different from previous searches
5. Format response as: Query1="...", Query2="...", Query3="...", etc."""

            user_prompt = f"""Original Question: {original_question}

Previous Iteration Results:
- Relevance Score: {previous_score}/10
- Analysis Summary: {previous_analysis.get('analysis_content', '')[:1000]}...

Generate 3-4 targeted follow-up search queries to address the gaps and improve relevance."""

            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                stream=False,
                timeout=30.0
            )
            
            response_text = response.choices[0].message.content
            generation_time = time.time() - start_time
            
            # Extract queries from response
            queries = []
            query_pattern = r'Query\d*="([^"]+)"'
            matches = re.findall(query_pattern, response_text)
            
            if matches:
                queries = matches
                logger.info(f"✅ Generated {len(queries)} follow-up queries in {generation_time:.2f}s")
                for i, query in enumerate(queries, 1):
                    logger.info(f"  {i}. {query}")
            else:
                # Fallback: modify original question
                queries = [f"{original_question} latest data", f"{original_question} market analysis", f"{original_question} revenue statistics"]
                logger.warning("⚠️ Could not extract follow-up queries, using modified versions")
            
            self.metrics.total_queries += len(queries)
            return queries
                
        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(f"❌ Follow-up query generation failed in {generation_time:.2f}s: {e}")
            return [f"{original_question} additional information"]
    
    async def cleanup(self):
        """Clean up resources"""
        await self.cache_service.close()

def print_separator(char="=", length=80):
    """Print a separator line"""
    print("\n" + char * length + "\n")

def print_step_header(step_num: int, description: str):
    """Print a step header"""
    print(f"\n🎯 STEP {step_num}: {description}")
    print("-" * 60)

async def test_enhanced_research():
    """Test the iterative enhanced research process with MongoDB caching"""
    print_separator()
    print("🚀 DEEPSEEK ENHANCED WEB RESEARCH v3.02 - WITH MONGODB CACHING")
    print("Enhanced Multi-Query Research Process with MongoDB Integration:")
    print("1. Initialize MongoDB connection and check cache statistics")
    print("2. Search existing cached content based on generated queries")
    print("3. Generate multiple search queries from different angles")
    print("4. Perform comprehensive search with smart URL deduplication")
    print("5. Extract content with MongoDB caching (avoid duplicate scraping)")
    print("6. Analyze with gap identification and relevance scoring")
    print("7. If target relevance not met, generate follow-up queries and repeat")
    print("8. Comprehensive performance tracking including cache metrics")
    print_separator()
    
    # Initialize service
    try:
        service = EnhancedDeepSeekResearchService()
        await service.initialize()
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    # Display initial cache statistics
    cache_stats = await service.cache_service.get_cache_stats()
    print(f"💾 MongoDB Cache Status:")
    print(f"   📊 Total Entries: {cache_stats.get('total_entries', 0)}")
    print(f"   🔥 Fresh Entries (≤7 days): {cache_stats.get('fresh_entries', 0)}")
    print(f"   🔌 Connected: {'Yes' if cache_stats.get('connected') else 'No'}")
    
    # Test question
    original_question = "Find the CRM/SFA software available in Japan and make the rank by their revenues"
    target_relevance = 7
    print(f"\n🎯 Research Question: {original_question}")
    print(f"📊 Target Relevance Score: {target_relevance}/10")
    
    # Conduct iterative research with MongoDB caching
    start_time = time.time()
    results = await service.conduct_enhanced_research(original_question, target_relevance=target_relevance, max_iterations=3)
    total_time = time.time() - start_time
    
    if not results.get('success'):
        print(f"❌ Research failed: {results.get('error', 'Unknown error')}")
        await service.cleanup()
        return
    
    # Display iteration results
    print_separator("=", 100)
    print("📊 ITERATION RESULTS WITH MONGODB CACHING")
    
    for iteration_data in results['iterations']:
        iteration_num = iteration_data['iteration']
        relevance = iteration_data['relevance_achieved']
        
        print(f"\n🔄 ITERATION {iteration_num}:")
        print(f"   📊 Relevance Score: {relevance}/10")
        
        # Show key metrics for each step
        for step_key, step_data in iteration_data['steps'].items():
            step_name = step_data['description']
            time_taken = step_data.get('time_taken', 0)
            success = "✅" if step_data.get('success', False) else "❌"
            
            if 'query_count' in step_data:
                print(f"   {success} {step_name}: {step_data['query_count']} queries in {time_taken:.2f}s")
            elif 'cached_results' in step_data:
                print(f"   {success} {step_name}: {step_data['cached_results']} cached results in {time_taken:.2f}s")
            elif 'total_results' in step_data:
                print(f"   {success} {step_name}: {step_data['total_results']} results in {time_taken:.2f}s")
            elif 'new_extractions' in step_data:
                cache_hits = step_data.get('cache_hits', 0)
                cache_misses = step_data.get('cache_misses', 0)
                cached_extractions = step_data.get('cached_extractions', 0)
                print(f"   {success} {step_name}: {step_data['successful_new']}/{step_data['new_extractions']} successful + {cached_extractions} cached in {time_taken:.2f}s")
                if cache_hits + cache_misses > 0:
                    print(f"      💾 Cache: {cache_hits} hits, {cache_misses} misses")
            elif 'sources_analyzed' in step_data:
                print(f"   {success} {step_name}: {step_data['sources_analyzed']} sources in {time_taken:.2f}s")
    
    # Final comprehensive results with cache metrics
    final_metrics = results.get('final_metrics', {})
    final_score = final_metrics.get('final_relevance_score', 0)
    target_achieved = final_metrics.get('target_achieved', False)
    cache_performance = final_metrics.get('search_metrics', {}).get('cache_performance', {})
    
    print_separator("=", 100) 
    print("🎉 FINAL RESULTS & MONGODB CACHE PERFORMANCE")
    
    print(f"🎯 Target Achievement: {'SUCCESS' if target_achieved else 'PARTIAL'}")
    print(f"📊 Final Relevance Score: {final_score}/10 (Target: {target_relevance}/10)")
    print(f"🔄 Iterations Completed: {final_metrics.get('iterations_completed', 0)}/3")
    print(f"⏱️ Total Research Time: {final_metrics.get('total_duration', 0):.2f}s")
    
    # Cache performance breakdown
    print(f"\n💾 MONGODB CACHE PERFORMANCE:")
    print(f"   🎯 Cache Hits: {cache_performance.get('cache_hits', 0)}")
    print(f"   🌐 Cache Misses: {cache_performance.get('cache_misses', 0)}")
    print(f"   📊 Cache Hit Rate: {cache_performance.get('cache_hit_rate', 0):.1f}%")
    print(f"   📚 Total Cache Entries: {cache_performance.get('total_cache_entries', 0)}")
    print(f"   🔥 Fresh Cache Entries: {cache_performance.get('fresh_cache_entries', 0)}")
    
    # Performance breakdown
    search_metrics = final_metrics.get('search_metrics', {})
    print(f"\n📈 PERFORMANCE BREAKDOWN:")
    print(f"   🔍 Total Queries: {search_metrics.get('total_queries', 0)}")
    print(f"   📄 Total Sources Found: {search_metrics.get('total_results', 0)}")
    print(f"   ✅ Successful Extractions: {search_metrics.get('successful_extractions', 0)}")
    print(f"   📊 Extraction Success Rate: {search_metrics.get('extraction_success_rate', 0):.1f}%")
    
    # Relevance progression
    relevance_progression = search_metrics.get('relevance_progression', [])
    if relevance_progression:
        print(f"   📈 Relevance Progression: {' → '.join(map(str, relevance_progression))}")
    
    # Source distribution
    source_dist = search_metrics.get('source_distribution', {})
    if source_dist:
        print(f"   🏢 Source Types: {dict(source_dist)}")
    
    # Display final analysis
    if results['iterations']:
        final_iteration = results['iterations'][-1]
        final_analysis = final_iteration['steps']['step4']['analysis']
        
        print_separator("-", 80)
        print("📋 FINAL COMPREHENSIVE ANALYSIS:")
        analysis_content = final_analysis.get('analysis_content', 'No analysis available')
        print(analysis_content[:1500] + "..." if len(analysis_content) > 1500 else analysis_content)
    
    print_separator("=", 100)
    
    # Success/failure assessment
    if target_achieved:
        print("🎉 SUCCESS: Target relevance score achieved with MongoDB caching!")
        print(f"✅ Research completed with {final_score}/10 relevance (≥{target_relevance} required)")
        if cache_performance.get('cache_hits', 0) > 0:
            print(f"💾 Cache optimization: {cache_performance.get('cache_hits', 0)} URLs served from cache")
    else:
        print("⚠️ PARTIAL SUCCESS: Target relevance not fully achieved")
        print(f"📊 Final score: {final_score}/10 (Target: {target_relevance}/10)")
        print("💡 Consider running additional iterations or refining the research question")
    
    # Save comprehensive results with cache info
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'enhanced_research_v3_02_results_{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {filename}")
    
    # Get final cache statistics
    final_cache_stats = await service.cache_service.get_cache_stats()
    print(f"📊 Final cache statistics: {final_cache_stats.get('total_entries', 0)} total entries")
    
    # Cleanup
    await service.cleanup()
    print("🚀 Enhanced research with MongoDB caching completed!")

def check_environment():
    """Check if required environment variables are set"""
    print("🔧 Checking environment variables...")
    
    required_vars = {
        'DEEPSEEK_API_KEY': os.environ.get('DEEPSEEK_API_KEY'),
        'GOOGLE_API_KEY': os.environ.get('GOOGLE_API_KEY'),
        'GOOGLE_CSE_ID': os.environ.get('GOOGLE_CSE_ID'),
        'MONGODB_URI': os.environ.get('MONGODB_URI', 'mongodb://localhost:27017')
    }
    
    missing_vars = []
    for var, value in required_vars.items():
        if var == 'MONGODB_URI':
            print(f"✅ {var}: {value}")  # MongoDB URI is OK to show
        elif value and value != f"your_{var.lower()}_here":
            masked_value = f"{value[:5]}...{value[-5:]}" if len(value) > 10 else "***"
            print(f"✅ {var}: {masked_value}")
        else:
            print(f"❌ {var}: Not set")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n⚠️ Missing environment variables: {', '.join(missing_vars)}")
        print("\nPlease configure your .env file with the required API keys.")
        if 'MONGODB_URI' not in missing_vars:
            print("💡 MongoDB URI is set - cache functionality will be available")
        return False
    
    return True

async def main():
    """Main test function"""
    print("🚀 DEEPSEEK ENHANCED WEB RESEARCH v3.02 - WITH MONGODB CACHING")
    print("=" * 70)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment setup incomplete. Please configure required variables.")
        sys.exit(1)
    
    try:
        # Run the enhanced iterative research test with MongoDB caching
        await test_enhanced_research()
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())