#!/usr/bin/env python3
"""
Enhanced DeepSeek Web Research with Multi-Query Strategy and Comprehensive Logging v3.01
This script implements an advanced research workflow with improved relevance scoring:
1. Multi-angle search query generation
2. Enhanced content filtering and source diversification
3. Iterative query refinement based on gaps
4. Comprehensive logging and performance analysis
5. Time tracking for overall evaluation

Usage:
    python test_deepseek_advanced_web_research3_01.py

Environment variables:
    DEEPSEEK_API_KEY: Your DeepSeek API key
    DEEPSEEK_API_URL: The base URL for the DeepSeek API (default: https://api.deepseek.com)
    GOOGLE_API_KEY: Your Google Custom Search API key
    GOOGLE_CSE_ID: Your Google Custom Search Engine ID
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

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='deepseek_enhanced_research.log',
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
    source_types: Dict[str, int] = None
    relevance_scores: List[float] = None
    
    def __post_init__(self):
        if self.source_types is None:
            self.source_types = defaultdict(int)
        if self.relevance_scores is None:
            self.relevance_scores = []

class EnhancedWebContentExtractor:
    """Enhanced service for extracting content from web pages with quality assessment"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.domain_quality_cache = {}
    
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
    
    async def extract_article_content(self, url: str) -> Dict[str, Any]:
        """Extract article content with enhanced quality assessment"""
        start_time = time.time()
        
        try:
            logger.info(f"🔄 Extracting content from: {url}")
            
            # Assess domain quality first
            domain_info = self.assess_domain_quality(url)
            
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
                'error': str(e)
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
    """Enhanced research service with multi-query strategy and comprehensive logging"""
    
    def __init__(self):
        self.api_key = os.environ.get('DEEPSEEK_API_KEY', '')
        self.api_url = os.environ.get('DEEPSEEK_API_URL', 'https://api.deepseek.com')
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.api_url)
        self.web_search = EnhancedGoogleWebSearchService()
        self.content_extractor = EnhancedWebContentExtractor()
        self.metrics = SearchMetrics()
        
        if not self.api_key:
            logger.error("❌ DEEPSEEK_API_KEY not set")
            raise ValueError("DEEPSEEK_API_KEY environment variable is required")
    
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
    
    async def extract_and_analyze_content(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract content with quality filtering and source diversification"""
        start_time = time.time()
        
        logger.info(f"📄 Extracting content from {len(search_results)} sources")
        
        # Sort by domain quality for better extraction order
        def get_quality_score(result):
            domain_info = self.content_extractor.assess_domain_quality(result['link'])
            return domain_info['quality_score']
        
        sorted_results = sorted(search_results, key=get_quality_score, reverse=True)
        
        extracted_contents = []
        source_type_counts = defaultdict(int)
        
        for i, result in enumerate(sorted_results):
            logger.info(f"📄 Extracting {i+1}/{len(sorted_results)}: {result['displayLink']}")
            
            content = await self.content_extractor.extract_article_content(result['link'])
            content['search_result'] = result
            
            # Track source types for diversification
            source_type = content.get('domain_info', {}).get('source_type', 'unknown')
            source_type_counts[source_type] += 1
            self.metrics.source_types[source_type] += 1
            
            if content['success']:
                self.metrics.successful_extractions += 1
                logger.info(f"  ✅ Success: {content['word_count']} words, quality: {content.get('domain_info', {}).get('quality_score', 0)}/10")
            else:
                self.metrics.failed_extractions += 1
                logger.warning(f"  ❌ Failed: {content.get('error', 'Unknown error')}")
            
            extracted_contents.append(content)
            
            # Add delay to be respectful to websites
            await asyncio.sleep(1)
        
        extraction_time = time.time() - start_time
        logger.info(f"✅ Content extraction completed in {extraction_time:.2f}s")
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
                    
                    summary = f"""
Source {i+1}: {content['title']}
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
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def conduct_enhanced_research(self, original_question: str, target_relevance: int = 7, max_iterations: int = 3) -> Dict[str, Any]:
        """Conduct comprehensive research with iterative improvement until target relevance is achieved"""
        timing = TimingMetrics(start_time=time.time())
        
        logger.info(f"🚀 Starting enhanced research for: {original_question}")
        logger.info(f"🎯 Target relevance score: {target_relevance}/10")
        
        results = {
            'original_question': original_question,
            'timestamp': datetime.utcnow().isoformat(),
            'research_type': 'enhanced_multi_query',
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
                
                # Step 2: Comprehensive search
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
                
                if not search_results:
                    logger.warning(f"⚠️ No search results found in iteration {iteration}")
                    iteration_results['warning'] = 'No search results found'
                    results['iterations'].append(iteration_results)
                    continue
                
                # Step 3: Content extraction and analysis
                timing.start_phase(f'content_extraction_iter{iteration}')
                new_extracted_contents = await self.extract_and_analyze_content(search_results)
                timing.end_phase(f'content_extraction_iter{iteration}')
                
                # Combine with previous contents for comprehensive analysis
                all_extracted_contents.extend(new_extracted_contents)
                
                iteration_results['steps']['step3'] = {
                    'description': f'Content extraction iteration {iteration}',
                    'new_extractions': len(new_extracted_contents),
                    'total_extractions': len(all_extracted_contents),
                    'successful_new': sum(1 for c in new_extracted_contents if c['success']),
                    'time_taken': timing.phase_times.get(f'content_extraction_iter{iteration}_duration', 0),
                    'success': sum(1 for c in new_extracted_contents if c['success']) > 0
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
                    'relevance_progression': [iter_data['relevance_achieved'] for iter_data in results['iterations']]
                },
                'performance_analysis': {
                    'avg_time_per_iteration': total_duration / max(1, iteration),
                    'relevance_improvement_rate': (current_relevance - results['iterations'][0]['relevance_achieved']) / max(1, iteration - 1) if iteration > 1 else 0,
                    'final_analysis_summary': final_analysis.get('analysis_content', '')[:500] + '...' if final_analysis.get('analysis_content') else 'No analysis available'
                }
            }
            
            results['success'] = True
            
            # Enhanced logging summary
            logger.info(f"✅ Enhanced research completed!")
            logger.info(f"📊 Final relevance score: {current_relevance}/10 (Target: {target_relevance})")
            logger.info(f"🔄 Iterations completed: {iteration}/{max_iterations}")
            logger.info(f"⏱️ Total time: {total_duration:.2f}s")
            logger.info(f"🎯 Target achieved: {'Yes' if current_relevance >= target_relevance else 'No'}")
            
            # Detailed performance summary
            print(f"\n📈 FINAL PERFORMANCE SUMMARY:")
            print(f"   🎯 Target Relevance: {target_relevance}/10")
            print(f"   📊 Final Score: {current_relevance}/10")
            print(f"   ✅ Target Achieved: {'Yes' if current_relevance >= target_relevance else 'No'}")
            print(f"   🔄 Iterations: {iteration}/{max_iterations}")
            print(f"   ⏱️ Total Duration: {total_duration:.2f}s")
            print(f"   🔍 Total Queries: {self.metrics.total_queries}")
            print(f"   📄 Total Sources: {len(all_extracted_contents)}")
            print(f"   ✅ Successful Extractions: {self.metrics.successful_extractions}")
            
        except Exception as e:
            timing.end_time = time.time()
            total_duration = timing.get_total_duration()
            
            logger.error(f"❌ Enhanced research failed after {total_duration:.2f}s: {e}")
            results['error'] = str(e)
            results['success'] = False
            results['final_metrics'] = {
                'total_duration': total_duration,
                'iterations_completed': iteration,
                'phase_durations': timing.get_phase_summary()
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

def print_separator(char="=", length=80):
    """Print a separator line"""
    print("\n" + char * length + "\n")

def print_step_header(step_num: int, description: str):
    """Print a step header"""
    print(f"\n🎯 STEP {step_num}: {description}")
    print("-" * 60)

async def test_enhanced_research():
    """Test the iterative enhanced research process"""
    print_separator()
    print("🚀 DEEPSEEK ENHANCED WEB RESEARCH v3.01 - ITERATIVE MODE")
    print("Enhanced Multi-Query Research Process with Iterative Improvement:")
    print("1. Generate multiple search queries from different angles")
    print("2. Perform comprehensive search with quality filtering")
    print("3. Extract content with source diversification")
    print("4. Analyze with gap identification and relevance scoring")
    print("5. If target relevance not met, generate follow-up queries and repeat")
    print("6. Comprehensive performance tracking and evaluation")
    print_separator()
    
    # Initialize service
    try:
        service = EnhancedDeepSeekResearchService()
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    # Test question
    original_question = "Find the CRM/SFA software available in Japan and make the rank by their revenues"
    target_relevance = 7
    print(f"🎯 Research Question: {original_question}")
    print(f"📊 Target Relevance Score: {target_relevance}/10")
    
    # Conduct iterative research
    start_time = time.time()
    results = await service.conduct_enhanced_research(original_question, target_relevance=target_relevance, max_iterations=3)
    total_time = time.time() - start_time
    
    if not results.get('success'):
        print(f"❌ Research failed: {results.get('error', 'Unknown error')}")
        return
    
    # Display iteration results
    print_separator("=", 100)
    print("📊 ITERATION RESULTS SUMMARY")
    
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
            elif 'total_results' in step_data:
                print(f"   {success} {step_name}: {step_data['total_results']} results in {time_taken:.2f}s")
            elif 'new_extractions' in step_data:
                print(f"   {success} {step_name}: {step_data['successful_new']}/{step_data['new_extractions']} successful in {time_taken:.2f}s")
            elif 'sources_analyzed' in step_data:
                print(f"   {success} {step_name}: {step_data['sources_analyzed']} sources in {time_taken:.2f}s")
    
    # Final comprehensive results
    final_metrics = results.get('final_metrics', {})
    final_score = final_metrics.get('final_relevance_score', 0)
    target_achieved = final_metrics.get('target_achieved', False)
    
    print_separator("=", 100) 
    print("🎉 FINAL RESULTS & EVALUATION")
    
    print(f"🎯 Target Achievement: {'SUCCESS' if target_achieved else 'PARTIAL'}")
    print(f"📊 Final Relevance Score: {final_score}/10 (Target: {target_relevance}/10)")
    print(f"🔄 Iterations Completed: {final_metrics.get('iterations_completed', 0)}/3")
    print(f"⏱️ Total Research Time: {final_metrics.get('total_duration', 0):.2f}s")
    
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
        print("🎉 SUCCESS: Target relevance score achieved!")
        print(f"✅ Research completed with {final_score}/10 relevance (≥{target_relevance} required)")
    else:
        print("⚠️ PARTIAL SUCCESS: Target relevance not fully achieved")
        print(f"📊 Final score: {final_score}/10 (Target: {target_relevance}/10)")
        print("💡 Consider running additional iterations or refining the research question")
    
    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'enhanced_research_v3_01_results_{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {filename}")
    print("🚀 Enhanced iterative research process completed!")

def check_environment():
    """Check if required environment variables are set"""
    print("🔧 Checking environment variables...")
    
    required_vars = {
        'DEEPSEEK_API_KEY': os.environ.get('DEEPSEEK_API_KEY'),
        'GOOGLE_API_KEY': os.environ.get('GOOGLE_API_KEY'),
        'GOOGLE_CSE_ID': os.environ.get('GOOGLE_CSE_ID')
    }
    
    missing_vars = []
    for var, value in required_vars.items():
        if value and value != f"your_{var.lower()}_here":
            masked_value = f"{value[:5]}...{value[-5:]}" if len(value) > 10 else "***"
            print(f"✅ {var}: {masked_value}")
        else:
            print(f"❌ {var}: Not set")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n⚠️ Missing environment variables: {', '.join(missing_vars)}")
        print("\nPlease configure your .env file with the required API keys.")
        return False
    
    return True

async def main():
    """Main test function"""
    print("🚀 DEEPSEEK ENHANCED WEB RESEARCH v3.01")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment setup incomplete. Please configure required variables.")
        sys.exit(1)
    
    try:
        # Run the enhanced iterative research test
        await test_enhanced_research()
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())