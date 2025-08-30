#!/usr/bin/env python3
"""
DeepSeek Advanced Web Research with Serper MCP Integration v4.01

This script implements an advanced research system that combines:
- DeepSeek LLM for reasoning and analysis via OpenAI-compatible API
- Serper API for professional web search and content scraping
- Deep-thinking algorithms inspired by the 'jan' project
- MCP (Model Context Protocol) patterns from 'mcp-server-serper'

Key Features:
- Multi-perspective query generation using deep-thinking patterns
- Professional web search via Serper API with advanced operators
- Content scraping with markdown and metadata extraction
- Relevance evaluation with 70% threshold filtering
- Progressive answer building with confidence tracking
- Optional MongoDB caching for performance optimization
- Statistical data extraction and ranking
- 10-minute research timeout with graceful degradation

Usage:
    python test_deepseek_advanced_web_research4_01.py

Environment variables:
    DEEPSEEK_API_KEY: Your DeepSeek API key
    DEEPSEEK_API_URL: DeepSeek API endpoint (default: https://api.deepseek.com)
    SERPER_API_KEY: Your Serper API key for search and scraping
    MONGODB_URI: MongoDB connection string (optional, for caching)
    CACHE_EXPIRY_DAYS: Cache expiry in days (default: 30)
    MAX_RESEARCH_TIME: Maximum research time in seconds (default: 600)
"""

import os
import sys
import json
import asyncio
import logging
import time
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
from urllib.parse import urlparse, urljoin

# Environment and configuration
from dotenv import load_dotenv

# DeepSeek API via OpenAI interface
try:
    from openai import AsyncOpenAI
except ImportError:
    print("❌ Missing OpenAI library")
    print("📦 Please install: pip install openai")
    sys.exit(1)

# Async HTTP client
try:
    import aiohttp
except ImportError:
    print("❌ Missing aiohttp library")
    print("📦 Please install: pip install aiohttp")
    sys.exit(1)

# Token counting
try:
    import tiktoken
except ImportError:
    print("❌ Missing tiktoken library")
    print("📦 Please install: pip install tiktoken")
    sys.exit(1)

# Data validation (optional but recommended)
try:
    from pydantic import BaseModel, Field, validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    print("⚠️ Pydantic not available, using dataclasses only")

# MongoDB libraries (optional)
try:
    from motor.motor_asyncio import AsyncIOMotorClient
    from pymongo import MongoClient
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    print("ℹ️ MongoDB libraries not available, caching disabled")

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deepseek_serper_research_v401.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration Constants
# =============================================================================

# API Configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Research Configuration
MAX_RESEARCH_TIME = int(os.getenv("MAX_RESEARCH_TIME", "600"))  # 10 minutes
MAX_QUERIES_PER_RESEARCH = 10
MAX_RESULTS_PER_QUERY = 10
RELEVANCE_THRESHOLD = 0.7  # 70% relevance threshold

# Token Management
MAX_TOTAL_TOKENS = 50000
MAX_CONTENT_LENGTH = 2000
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_TEMPERATURE = 0.7
DEEPSEEK_MAX_TOKENS = 4000

# Serper Configuration
SERPER_BASE_URL = "https://google.serper.dev"
SERPER_SCRAPE_URL = "https://scrape.serper.dev"
SERPER_DEFAULT_REGION = "us"
SERPER_DEFAULT_LANGUAGE = "en"

# MongoDB Configuration (optional)
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
CACHE_EXPIRY_DAYS = int(os.getenv("CACHE_EXPIRY_DAYS", "30"))

# Validate required configuration
if not DEEPSEEK_API_KEY:
    logger.error("❌ DEEPSEEK_API_KEY environment variable is required")
    sys.exit(1)

if not SERPER_API_KEY:
    logger.error("❌ SERPER_API_KEY environment variable is required")
    sys.exit(1)

# =============================================================================
# Data Models
# =============================================================================

class SearchType(Enum):
    """Types of searches supported"""
    GENERAL = "general"
    NEWS = "news"
    ACADEMIC = "academic"
    TECHNICAL = "technical"
    BUSINESS = "business"

@dataclass
class SearchQuery:
    """Enhanced search query with operators"""
    text: str
    priority: int = 1
    search_type: SearchType = SearchType.GENERAL
    region: str = SERPER_DEFAULT_REGION
    language: str = SERPER_DEFAULT_LANGUAGE
    time_filter: Optional[str] = None  # qdr:h, qdr:d, qdr:w, qdr:m, qdr:y
    num_results: int = MAX_RESULTS_PER_QUERY
    page: int = 1
    
    # Advanced operators
    site: Optional[str] = None
    filetype: Optional[str] = None
    intitle: Optional[str] = None
    inurl: Optional[str] = None
    exact_phrase: Optional[str] = None
    exclude_terms: List[str] = field(default_factory=list)
    or_terms: List[str] = field(default_factory=list)
    date_before: Optional[str] = None
    date_after: Optional[str] = None
    
    def to_serper_params(self) -> Dict[str, Any]:
        """Convert to Serper API parameters"""
        return {
            "q": self.text,
            "gl": self.region,
            "hl": self.language,
            "num": self.num_results,
            "page": self.page,
            "tbs": self.time_filter
        }

@dataclass
class SearchResult:
    """Search result from Serper"""
    url: str
    title: str
    snippet: str
    position: int
    domain: str
    cached_url: Optional[str] = None
    
@dataclass
class ScrapeResult:
    """Content scraped from webpage"""
    url: str
    text: str
    markdown: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    json_ld: Optional[Dict] = None
    extraction_time: float = 0.0
    
@dataclass
class ScoredContent:
    """Content with relevance score"""
    url: str
    title: str
    content: str
    relevance_score: float  # 0-1 scale
    confidence: float
    source_quality: int  # 1-10 scale
    extraction_method: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class Answer:
    """Synthesized answer"""
    content: str
    confidence: float
    sources: List[str]
    statistics: Optional[Dict] = None
    gaps: List[str] = field(default_factory=list)
    versions: List[Dict] = field(default_factory=list)
    generation_time: float = 0.0

@dataclass
class ResearchResult:
    """Complete research output"""
    question: str
    answer: Answer
    queries_generated: int
    sources_analyzed: int
    cache_hits: int = 0
    total_duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

# =============================================================================
# Token Management
# =============================================================================

class TokenManager:
    """Manage token counting and optimization"""
    
    def __init__(self, max_tokens: int = MAX_TOTAL_TOKENS):
        self.max_tokens = max_tokens
        self.encoding = None
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except Exception as e:
            logger.warning(f"⚠️ Could not load tiktoken encoding: {e}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.encoding:
            try:
                return len(self.encoding.encode(text))
            except:
                pass
        # Fallback: rough estimation (1 token ≈ 4 characters)
        return len(text) // 4
    
    def optimize_content(self, content: str, max_length: int = MAX_CONTENT_LENGTH) -> str:
        """Optimize content to fit within token limits"""
        if len(content) <= max_length:
            return content
        
        # Extract key sentences
        sentences = content.split('. ')
        if len(sentences) <= 3:
            return content[:max_length] + "..."
        
        # Take first, middle, and last parts
        first_part = '. '.join(sentences[:2])
        middle_idx = len(sentences) // 2
        middle_part = sentences[middle_idx]
        last_part = sentences[-1]
        
        summarized = f"{first_part}. ... {middle_part}. ... {last_part}"
        
        if len(summarized) > max_length:
            return content[:max_length] + "..."
        
        return summarized
    
    def estimate_cost(self, input_tokens: int, output_tokens: int, model: str = "deepseek-chat") -> float:
        """Estimate cost in USD based on token usage"""
        # DeepSeek pricing (as of 2024)
        # Input: $0.14 per 1M tokens, Output: $0.28 per 1M tokens
        DEEPSEEK_INPUT_COST_PER_1M = 0.14
        DEEPSEEK_OUTPUT_COST_PER_1M = 0.28
        
        if model.startswith("deepseek"):
            input_cost = (input_tokens / 1_000_000) * DEEPSEEK_INPUT_COST_PER_1M
            output_cost = (output_tokens / 1_000_000) * DEEPSEEK_OUTPUT_COST_PER_1M
            return input_cost + output_cost
        
        # Fallback generic pricing
        return (input_tokens + output_tokens) / 1_000_000 * 0.20
    
    def create_batches(self, content_list: List[str], max_tokens_per_batch: int = 15000) -> List[List[str]]:
        """Create batches of content that fit within token limits"""
        batches = []
        current_batch = []
        current_tokens = 0
        
        for content in content_list:
            content_tokens = self.count_tokens(content)
            
            # If single content exceeds limit, optimize it
            if content_tokens > max_tokens_per_batch:
                # Process current batch if not empty
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
                
                # Optimize large content and add as single batch
                optimized = self.optimize_content(content, max_tokens_per_batch * 4)  # 4 chars per token
                batches.append([optimized])
                continue
            
            # Check if adding this content would exceed limit
            if current_tokens + content_tokens > max_tokens_per_batch:
                # Save current batch and start new one
                if current_batch:
                    batches.append(current_batch)
                current_batch = [content]
                current_tokens = content_tokens
            else:
                # Add to current batch
                current_batch.append(content)
                current_tokens += content_tokens
        
        # Add final batch if not empty
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def calculate_batch_stats(self, batches: List[List[str]]) -> Dict[str, Any]:
        """Calculate statistics for batching"""
        total_items = sum(len(batch) for batch in batches)
        total_tokens = sum(
            sum(self.count_tokens(content) for content in batch)
            for batch in batches
        )
        
        batch_sizes = [len(batch) for batch in batches]
        batch_tokens = [
            sum(self.count_tokens(content) for content in batch)
            for batch in batches
        ]
        
        return {
            "total_batches": len(batches),
            "total_items": total_items,
            "total_tokens": total_tokens,
            "avg_batch_size": sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0,
            "avg_tokens_per_batch": sum(batch_tokens) / len(batch_tokens) if batch_tokens else 0,
            "min_batch_size": min(batch_sizes) if batch_sizes else 0,
            "max_batch_size": max(batch_sizes) if batch_sizes else 0,
            "estimated_cost": self.estimate_cost(total_tokens, total_tokens // 4)  # Assume 1:4 input:output ratio
        }
    
    def optimize_for_cost(self, content_list: List[str], target_cost: float = 1.0) -> List[str]:
        """Optimize content list to stay within target cost"""
        optimized = []
        current_cost = 0.0
        
        for content in content_list:
            tokens = self.count_tokens(content)
            estimated_cost = self.estimate_cost(tokens, tokens // 4)
            
            if current_cost + estimated_cost > target_cost:
                # Try to optimize content to fit budget
                remaining_budget = target_cost - current_cost
                if remaining_budget > 0.01:  # At least 1 cent
                    # Calculate max tokens for remaining budget
                    max_tokens = int((remaining_budget / 0.20) * 1_000_000)  # Conservative estimate
                    optimized_content = self.optimize_content(content, max_tokens * 4)
                    optimized.append(optimized_content)
                break
            else:
                optimized.append(content)
                current_cost += estimated_cost
        
        return optimized
    
    def get_usage_report(self, token_usage_history: List[Dict[str, int]]) -> Dict[str, Any]:
        """Generate usage and cost report"""
        if not token_usage_history:
            return {"error": "No usage history provided"}
        
        total_input_tokens = sum(usage.get("input_tokens", 0) for usage in token_usage_history)
        total_output_tokens = sum(usage.get("output_tokens", 0) for usage in token_usage_history)
        total_cost = self.estimate_cost(total_input_tokens, total_output_tokens)
        
        return {
            "total_requests": len(token_usage_history),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "total_cost_usd": round(total_cost, 4),
            "avg_tokens_per_request": (total_input_tokens + total_output_tokens) / len(token_usage_history),
            "avg_cost_per_request": round(total_cost / len(token_usage_history), 6),
            "cost_breakdown": {
                "input_cost": round(self.estimate_cost(total_input_tokens, 0), 4),
                "output_cost": round(self.estimate_cost(0, total_output_tokens), 4)
            }
        }

# =============================================================================
# Serper API Client
# =============================================================================

class SerperClient:
    """Serper API client with MCP patterns"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = SERPER_BASE_URL
        self.scrape_url = SERPER_SCRAPE_URL
        self.session = None
        self.request_count = 0
        self.last_request_time = 0
        self.rate_limit_delay = 1.0  # Minimum seconds between requests
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _rate_limit(self):
        """Apply rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    def build_advanced_query(self, query: SearchQuery) -> str:
        """Build query string with advanced operators"""
        q = query.text
        
        # Apply search operators
        if query.site:
            q += f" site:{query.site}"
        if query.filetype:
            q += f" filetype:{query.filetype}"
        if query.intitle:
            q += f" intitle:{query.intitle}"
        if query.inurl:
            q += f" inurl:{query.inurl}"
        if query.exact_phrase:
            q += f' "{query.exact_phrase}"'
        if query.exclude_terms:
            for term in query.exclude_terms:
                q += f" -{term}"
        if query.or_terms:
            q += f" ({' OR '.join(query.or_terms)})"
        if query.date_before:
            q += f" before:{query.date_before}"
        if query.date_after:
            q += f" after:{query.date_after}"
        
        return q.strip()
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Execute web search"""
        await self._rate_limit()
        
        # Build query with operators
        search_query = self.build_advanced_query(query)
        
        # Prepare request parameters
        params = {
            "q": search_query,
            "gl": query.region,
            "hl": query.language,
            "num": query.num_results,
            "page": query.page
        }
        
        if query.time_filter:
            params["tbs"] = query.time_filter
        
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/search",
                json=params,
                headers=headers
            ) as response:
                self.request_count += 1
                
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Serper API error: {response.status} - {error_text}")
                    return []
                
                data = await response.json()
                
                # Parse organic results
                results = []
                for idx, item in enumerate(data.get("organic", [])):
                    result = SearchResult(
                        url=item.get("link", ""),
                        title=item.get("title", ""),
                        snippet=item.get("snippet", ""),
                        position=idx + 1,
                        domain=urlparse(item.get("link", "")).netloc,
                        cached_url=item.get("cachedPageLink")
                    )
                    results.append(result)
                
                logger.info(f"🔍 Serper search completed: {len(results)} results")
                return results
                
        except Exception as e:
            logger.error(f"❌ Serper search failed: {e}")
            return []
    
    async def scrape(self, url: str, include_markdown: bool = True) -> Optional[ScrapeResult]:
        """Scrape webpage content"""
        await self._rate_limit()
        
        params = {
            "url": url,
            "includeMarkdown": include_markdown
        }
        
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        
        start_time = time.time()
        
        try:
            async with self.session.post(
                self.scrape_url,
                json=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                self.request_count += 1
                
                if response.status != 200:
                    error_text = await response.text()
                    logger.warning(f"Scrape failed for {url}: {response.status}")
                    return None
                
                data = await response.json()
                
                result = ScrapeResult(
                    url=url,
                    text=data.get("text", ""),
                    markdown=data.get("markdown"),
                    metadata=data.get("metadata", {}),
                    json_ld=data.get("jsonLd"),
                    extraction_time=time.time() - start_time
                )
                
                logger.info(f"✅ Scraped {url} in {result.extraction_time:.2f}s")
                return result
                
        except asyncio.TimeoutError:
            logger.warning(f"⏱️ Scrape timeout for {url}")
            return None
        except Exception as e:
            logger.error(f"❌ Scrape error for {url}: {e}")
            return None

# =============================================================================
# Deep-Thinking Query Generation
# =============================================================================

class QueryPatterns:
    """Repository of query generation patterns"""
    
    FACTUAL_PATTERNS = [
        "what is {topic}",
        "define {topic}",
        "{topic} explanation",
        "{topic} overview"
    ]
    
    COMPARATIVE_PATTERNS = [
        "{topic} vs {alternative}",
        "compare {topic} with {alternative}",
        "difference between {topic} and {alternative}",
        "{topic} comparison"
    ]
    
    TEMPORAL_PATTERNS = [
        "{topic} in {year}",
        "latest {topic}",
        "{topic} trends",
        "future of {topic}",
        "{topic} 2024",
        "recent developments {topic}"
    ]
    
    CAUSAL_PATTERNS = [
        "why {topic}",
        "{topic} causes",
        "{topic} effects",
        "impact of {topic}",
        "benefits of {topic}",
        "problems with {topic}"
    ]
    
    STATISTICAL_PATTERNS = [
        "{topic} statistics",
        "{topic} numbers",
        "{topic} market size",
        "{topic} growth rate",
        "{topic} data"
    ]
    
    @classmethod
    def apply_pattern(cls, pattern: str, entities: Dict[str, str]) -> str:
        """Apply pattern with entity substitution"""
        result = pattern
        for key, value in entities.items():
            result = result.replace(f"{{{key}}}", value)
        return result

class DeepThinkingEngine:
    """Query generation using deep-thinking patterns inspired by 'jan'"""
    
    def __init__(self, llm_client: AsyncOpenAI):
        self.llm = llm_client
        self.patterns = QueryPatterns()
        self.generated_queries = set()  # Track to avoid duplicates
    
    async def generate_queries(self, question: str, max_queries: int = MAX_QUERIES_PER_RESEARCH) -> List[SearchQuery]:
        """Generate multi-perspective search queries"""
        logger.info("🧠 Starting deep-thinking query generation")
        
        # Phase 1: Analyze the question
        analysis = await self.analyze_question(question)
        
        # Phase 2: Generate queries from different perspectives
        queries = []
        
        # Factual queries
        queries.extend(await self.generate_factual_queries(analysis))
        
        # Comparative queries
        queries.extend(await self.generate_comparative_queries(analysis))
        
        # Temporal queries
        queries.extend(await self.generate_temporal_queries(analysis))
        
        # Statistical queries
        queries.extend(await self.generate_statistical_queries(analysis))
        
        # Expert perspective queries
        queries.extend(await self.generate_expert_queries(analysis))
        
        # Deduplicate and prioritize
        unique_queries = self.deduplicate_queries(queries)
        prioritized = self.prioritize_queries(unique_queries)
        
        # Return top queries
        final_queries = prioritized[:max_queries]
        logger.info(f"📋 Generated {len(final_queries)} unique queries")
        
        return final_queries
    
    async def analyze_question(self, question: str) -> Dict[str, Any]:
        """Deep analysis of the research question"""
        prompt = f"""Analyze this research question and extract key information:

Question: {question}

Provide a JSON response with:
1. main_topic: The primary subject
2. subtopics: List of related subtopics
3. entities: Key entities mentioned (people, companies, technologies)
4. intent: What the user wants to know (definition, comparison, statistics, etc.)
5. scope: Temporal scope (current, historical, future)
6. domain: Field of knowledge (technology, business, science, etc.)
"""
        
        try:
            response = await self.llm.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            content = response.choices[0].message.content
            # Parse JSON from response
            analysis = self.parse_json_response(content)
            logger.info(f"📊 Question analysis complete: {analysis.get('main_topic', 'Unknown')}")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Question analysis failed: {e}")
            # Fallback analysis
            return {
                "main_topic": question,
                "subtopics": [],
                "entities": [],
                "intent": "general",
                "scope": "current",
                "domain": "general"
            }
    
    def parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON from LLM response"""
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Fallback
        return {}
    
    async def generate_factual_queries(self, analysis: Dict) -> List[SearchQuery]:
        """Generate factual information queries"""
        queries = []
        topic = analysis.get("main_topic", "")
        
        if not topic:
            return queries
        
        # Apply factual patterns
        for pattern in self.patterns.FACTUAL_PATTERNS[:3]:
            query_text = pattern.format(topic=topic)
            if query_text not in self.generated_queries:
                self.generated_queries.add(query_text)
                queries.append(SearchQuery(
                    text=query_text,
                    priority=8,
                    search_type=SearchType.GENERAL
                ))
        
        return queries
    
    async def generate_comparative_queries(self, analysis: Dict) -> List[SearchQuery]:
        """Generate comparative analysis queries"""
        queries = []
        topic = analysis.get("main_topic", "")
        
        # Look for alternatives to compare
        if "vs" in topic or "compare" in topic.lower():
            queries.append(SearchQuery(
                text=f"{topic} comparison analysis",
                priority=9,
                search_type=SearchType.GENERAL
            ))
        
        return queries
    
    async def generate_temporal_queries(self, analysis: Dict) -> List[SearchQuery]:
        """Generate time-based queries"""
        queries = []
        topic = analysis.get("main_topic", "")
        scope = analysis.get("scope", "current")
        
        if scope in ["current", "future"]:
            # Recent developments
            queries.append(SearchQuery(
                text=f"latest {topic} 2024",
                priority=7,
                search_type=SearchType.NEWS,
                time_filter="qdr:m"  # Past month
            ))
            
            # Trends
            queries.append(SearchQuery(
                text=f"{topic} trends forecast",
                priority=6,
                search_type=SearchType.GENERAL
            ))
        
        return queries
    
    async def generate_statistical_queries(self, analysis: Dict) -> List[SearchQuery]:
        """Generate queries for statistical data"""
        queries = []
        topic = analysis.get("main_topic", "")
        
        # Statistical patterns
        queries.append(SearchQuery(
            text=f"{topic} statistics data numbers",
            priority=8,
            search_type=SearchType.GENERAL,
            intitle="statistics"
        ))
        
        # Look for PDFs with data
        queries.append(SearchQuery(
            text=f"{topic} report",
            priority=7,
            search_type=SearchType.GENERAL,
            filetype="pdf"
        ))
        
        return queries
    
    async def generate_expert_queries(self, analysis: Dict) -> List[SearchQuery]:
        """Generate expert-level queries"""
        queries = []
        topic = analysis.get("main_topic", "")
        domain = analysis.get("domain", "general")
        
        # Academic sources
        if domain in ["technology", "science"]:
            queries.append(SearchQuery(
                text=f"{topic} research paper",
                priority=6,
                search_type=SearchType.ACADEMIC,
                site="scholar.google.com"
            ))
        
        # Industry sources
        if domain in ["business", "technology"]:
            queries.append(SearchQuery(
                text=f"{topic} industry analysis",
                priority=7,
                search_type=SearchType.BUSINESS
            ))
        
        return queries
    
    def deduplicate_queries(self, queries: List[SearchQuery]) -> List[SearchQuery]:
        """Remove duplicate queries"""
        seen = set()
        unique = []
        
        for query in queries:
            key = query.text.lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(query)
        
        return unique
    
    def prioritize_queries(self, queries: List[SearchQuery]) -> List[SearchQuery]:
        """Sort queries by priority"""
        return sorted(queries, key=lambda q: q.priority, reverse=True)

# =============================================================================
# DeepSeek Integration
# =============================================================================

class DeepSeekClient:
    """DeepSeek API client using OpenAI interface"""
    
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=DEEPSEEK_API_URL
        )
        self.token_manager = TokenManager()
    
    async def evaluate_relevance(self, question: str, content: str, url: str) -> float:
        """Evaluate content relevance on 0-1 scale"""
        # Optimize content for token limits
        optimized_content = self.token_manager.optimize_content(content)
        
        prompt = f"""Rate the relevance of this content to the research question on a scale of 0-10.

Research Question: {question}

Content from {url}:
{optimized_content}

Provide only a number between 0 and 10, where:
0 = Completely irrelevant
5 = Somewhat relevant
7 = Relevant (meets threshold)
10 = Perfectly relevant

Rating:"""
        
        try:
            response = await self.client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10
            )
            
            # Extract rating
            rating_text = response.choices[0].message.content.strip()
            rating = float(re.search(r'\d+\.?\d*', rating_text).group()) / 10.0
            
            return min(max(rating, 0.0), 1.0)  # Ensure 0-1 range
            
        except Exception as e:
            logger.warning(f"⚠️ Relevance evaluation failed: {e}")
            return 0.5  # Default middle score
    
    async def synthesize_answer(self, question: str, contents: List[ScoredContent]) -> str:
        """Synthesize comprehensive answer from sources"""
        # Prepare source summaries
        source_texts = []
        for idx, content in enumerate(contents[:10], 1):  # Limit to top 10
            optimized = self.token_manager.optimize_content(content.content, 500)
            source_texts.append(f"Source {idx} ({content.relevance_score:.1%} relevant):\n{optimized}")
        
        sources_combined = "\n\n".join(source_texts)
        
        prompt = f"""Based on the following sources, provide a comprehensive answer to the research question.

Research Question: {question}

Sources:
{sources_combined}

Instructions:
1. Synthesize information from multiple sources
2. Provide a clear, structured answer
3. Include specific data and statistics when available
4. Cite source numbers [1], [2], etc.
5. Identify any gaps in the available information

Answer:"""
        
        try:
            response = await self.client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=DEEPSEEK_TEMPERATURE,
                max_tokens=DEEPSEEK_MAX_TOKENS
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"❌ Answer synthesis failed: {e}")
            return "Unable to synthesize answer due to an error."
    
    async def extract_statistics(self, contents: List[ScoredContent]) -> Dict[str, Any]:
        """Extract statistical data from contents"""
        stats = {
            "numbers_found": [],
            "percentages": [],
            "dates": [],
            "metrics": {}
        }
        
        for content in contents:
            # Extract numbers
            numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|thousand|K|M|B))?\b', 
                                content.content, re.IGNORECASE)
            stats["numbers_found"].extend(numbers[:5])  # Limit per source
            
            # Extract percentages
            percentages = re.findall(r'\b\d+(?:\.\d+)?%', content.content)
            stats["percentages"].extend(percentages[:3])
            
            # Extract years
            years = re.findall(r'\b20\d{2}\b', content.content)
            stats["dates"].extend(years[:3])
        
        # Deduplicate
        stats["numbers_found"] = list(set(stats["numbers_found"]))[:10]
        stats["percentages"] = list(set(stats["percentages"]))[:10]
        stats["dates"] = sorted(list(set(stats["dates"])))[:5]
        
        return stats
    
    async def reason(self, prompt: str, max_tokens: int = 1000) -> str:
        """General reasoning method for any prompt"""
        try:
            response = await self.client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"❌ Reasoning failed: {e}")
            return f"Error: Unable to generate reasoning - {str(e)}"
    
    async def reason_stream(self, prompt: str, max_tokens: int = 1000):
        """Streaming version of reasoning method"""
        try:
            stream = await self.client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=max_tokens,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"❌ Streaming reasoning failed: {e}")
            yield f"Error: Unable to generate streaming reasoning - {str(e)}"
    
    async def analyze_with_streaming(self, question: str):
        """Analyze question with streaming response"""
        prompt = f"""Analyze this research question and extract key information:

Question: {question}

Provide a detailed analysis including:
1. Main topic identification
2. Key concepts and entities
3. Question type and complexity
4. Suggested research angles
5. Expected information types

Analysis:"""

        try:
            stream = await self.client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    yield content
                
        except Exception as e:
            logger.error(f"❌ Streaming analysis failed: {e}")
            yield f"Error: Unable to generate streaming analysis - {str(e)}"
    
    async def analyze_question_complete(self, question: str) -> dict:
        """Complete analysis method that returns full result"""
        prompt = f"""Analyze this research question and extract key information:

Question: {question}

Provide a JSON response with:
- main_topic: primary topic
- key_concepts: list of key concepts
- question_type: type of question
- complexity: complexity level
- research_angles: suggested research approaches

Response:"""

        try:
            response = await self.client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800
            )
            
            content = response.choices[0].message.content.strip()
            
            # Try to parse as JSON
            try:
                import json
                return json.loads(content)
            except:
                return {"analysis": content, "raw_response": True}
                
        except Exception as e:
            logger.error(f"❌ Complete analysis failed: {e}")
            return {"error": f"Unable to generate analysis - {str(e)}"}

# =============================================================================
# Result Processing
# =============================================================================

class ResultProcessor:
    """Process and analyze search results"""
    
    def __init__(self, deepseek_client: DeepSeekClient, serper_client: SerperClient):
        self.deepseek = deepseek_client
        self.serper = serper_client
        self.token_manager = TokenManager()
    
    async def process_search_results(self, 
                                    results: List[SearchResult], 
                                    question: str) -> List[ScoredContent]:
        """Process search results and extract content"""
        scored_contents = []
        
        for result in results:
            # Try to scrape full content
            scraped = await self.serper.scrape(result.url)
            
            if scraped and scraped.text:
                content_text = scraped.text
                extraction_method = "scrape"
            else:
                # Fallback to snippet
                content_text = result.snippet
                extraction_method = "snippet"
            
            # Evaluate relevance
            relevance = await self.deepseek.evaluate_relevance(
                question, content_text, result.url
            )
            
            # Create scored content
            scored = ScoredContent(
                url=result.url,
                title=result.title,
                content=content_text,
                relevance_score=relevance,
                confidence=0.8 if extraction_method == "scrape" else 0.5,
                source_quality=self._assess_source_quality(result.domain),
                extraction_method=extraction_method
            )
            
            scored_contents.append(scored)
            
            # Log progress
            if relevance >= RELEVANCE_THRESHOLD:
                logger.info(f"✅ Relevant content found: {result.title} ({relevance:.1%})")
            else:
                logger.debug(f"❌ Low relevance: {result.title} ({relevance:.1%})")
        
        return scored_contents
    
    def _assess_source_quality(self, domain: str) -> int:
        """Assess source quality based on domain"""
        # High-quality domains
        high_quality = [
            "wikipedia.org", "nature.com", "science.org", 
            "ieee.org", "acm.org", "arxiv.org",
            "harvard.edu", "mit.edu", "stanford.edu"
        ]
        
        # Medium-quality domains
        medium_quality = [
            "medium.com", "github.com", "stackoverflow.com",
            "reddit.com", "quora.com"
        ]
        
        domain_lower = domain.lower()
        
        # Check for high-quality domains
        for hq in high_quality:
            if hq in domain_lower:
                return 9
        
        # Check for edu/gov/org domains
        if domain_lower.endswith(".edu") or domain_lower.endswith(".gov"):
            return 8
        if domain_lower.endswith(".org"):
            return 7
        
        # Check for medium-quality domains
        for mq in medium_quality:
            if mq in domain_lower:
                return 6
        
        # Default
        return 5
    
    def filter_by_relevance(self, contents: List[ScoredContent]) -> List[ScoredContent]:
        """Filter contents meeting relevance threshold"""
        filtered = [c for c in contents if c.relevance_score >= RELEVANCE_THRESHOLD]
        logger.info(f"📊 Filtered: {len(filtered)}/{len(contents)} meet {RELEVANCE_THRESHOLD:.0%} threshold")
        return filtered
    
    def deduplicate_contents(self, contents: List[ScoredContent]) -> List[ScoredContent]:
        """Remove duplicate content based on similarity"""
        if len(contents) <= 1:
            return contents
        
        unique = []
        seen_urls = set()
        seen_content_hashes = set()
        
        for content in contents:
            # Check URL
            if content.url in seen_urls:
                continue
            
            # Simple content hash (first 100 chars)
            content_hash = hash(content.content[:100] if len(content.content) > 100 else content.content)
            if content_hash in seen_content_hashes:
                continue
            
            unique.append(content)
            seen_urls.add(content.url)
            seen_content_hashes.add(content_hash)
        
        logger.info(f"🔄 Deduplicated: {len(contents)} -> {len(unique)} unique sources")
        return unique

# =============================================================================
# MongoDB Cache (Optional)
# =============================================================================

class MongoDBCache:
    """MongoDB cache for web content"""
    
    def __init__(self, uri: str = MONGODB_URI, expiry_days: int = CACHE_EXPIRY_DAYS):
        self.uri = uri
        self.expiry_days = expiry_days
        self.client = None
        self.db = None
        self.collection = None
        self.enabled = MONGODB_AVAILABLE
    
    async def connect(self):
        """Connect to MongoDB"""
        if not self.enabled:
            return
        
        try:
            self.client = AsyncIOMotorClient(self.uri)
            self.db = self.client.research_cache
            self.collection = self.db.web_content
            
            # Create TTL index for automatic expiry
            await self.collection.create_index(
                "cached_at",
                expireAfterSeconds=self.expiry_days * 24 * 60 * 60
            )
            
            logger.info(f"✅ MongoDB cache connected (expiry: {self.expiry_days} days)")
            
        except Exception as e:
            logger.warning(f"⚠️ MongoDB connection failed: {e}")
            self.enabled = False
    
    async def get(self, url: str) -> Optional[Dict[str, Any]]:
        """Get cached content"""
        if not self.enabled:
            return None
        
        try:
            result = await self.collection.find_one({"url": url})
            if result:
                logger.debug(f"💾 Cache hit: {url}")
                return result.get("content")
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
        
        return None
    
    async def set(self, url: str, content: Dict[str, Any]):
        """Cache content"""
        if not self.enabled:
            return
        
        try:
            await self.collection.update_one(
                {"url": url},
                {
                    "$set": {
                        "url": url,
                        "content": content,
                        "cached_at": datetime.utcnow()
                    }
                },
                upsert=True
            )
            logger.debug(f"💾 Cached: {url}")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    async def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()

# =============================================================================
# Progressive Answer Builder
# =============================================================================

class ProgressiveAnswerBuilder:
    """Build answers progressively as research progresses"""
    
    def __init__(self):
        self.answer_versions = []
        self.current_answer = ""
        self.confidence = 0.0
        self.sources_used = []
        self.gaps = []
    
    def update(self, new_content: ScoredContent) -> bool:
        """Update answer with new content if valuable"""
        if new_content.relevance_score < RELEVANCE_THRESHOLD:
            return False
        
        # Check if content adds value
        if new_content.url in self.sources_used:
            return False
        
        # Update sources
        self.sources_used.append(new_content.url)
        
        # Update confidence
        self.confidence = min(
            0.95,  # Max confidence
            self.confidence + (new_content.relevance_score * 0.1)
        )
        
        # Save version
        self.answer_versions.append({
            "version": len(self.answer_versions) + 1,
            "confidence": self.confidence,
            "sources_count": len(self.sources_used),
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.info(f"📝 Answer updated: v{len(self.answer_versions)}, "
                   f"confidence {self.confidence:.1%}, "
                   f"{len(self.sources_used)} sources")
        
        return True
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current answer state"""
        return {
            "answer": self.current_answer,
            "confidence": self.confidence,
            "sources": self.sources_used,
            "versions": len(self.answer_versions),
            "gaps": self.gaps
        }

# =============================================================================
# Main Research Orchestrator
# =============================================================================

class DeepSeekResearcher:
    """Main research orchestrator"""
    
    def __init__(self, deepseek_api_key: str = DEEPSEEK_API_KEY, 
                 serper_api_key: str = SERPER_API_KEY):
        # Initialize clients
        self.deepseek_client = DeepSeekClient(deepseek_api_key)
        self.serper_client = SerperClient(serper_api_key)
        
        # Initialize components
        self.thinking_engine = DeepThinkingEngine(self.deepseek_client.client)
        self.result_processor = ResultProcessor(self.deepseek_client, self.serper_client)
        self.answer_builder = ProgressiveAnswerBuilder()
        
        # Optional cache
        self.cache = MongoDBCache() if MONGODB_AVAILABLE else None
        
        # Metrics
        self.start_time = None
        self.queries_generated = 0
        self.sources_analyzed = 0
        self.cache_hits = 0
    
    async def research(self, question: str) -> ResearchResult:
        """Conduct comprehensive research"""
        self.start_time = time.time()
        logger.info(f"🚀 Starting research: {question}")
        
        # Initialize Serper client session
        async with self.serper_client:
            # Connect cache if available
            if self.cache:
                await self.cache.connect()
            
            try:
                # Phase 1: Generate queries
                queries = await self.thinking_engine.generate_queries(question)
                self.queries_generated = len(queries)
                
                # Phase 2: Execute searches and process results
                all_contents = []
                
                for query in queries:
                    # Check timeout
                    if self._check_timeout():
                        logger.warning("⏰ Research timeout reached")
                        break
                    
                    # Execute search
                    logger.info(f"🔍 Searching: {query.text}")
                    results = await self.serper_client.search(query)
                    
                    # Process results
                    contents = await self.result_processor.process_search_results(
                        results, question
                    )
                    all_contents.extend(contents)
                    
                    # Update progress
                    for content in contents:
                        self.answer_builder.update(content)
                    
                    # Check if we have enough relevant content
                    relevant_contents = self.result_processor.filter_by_relevance(contents)
                    if len(relevant_contents) >= 10:
                        logger.info("✅ Sufficient relevant content found")
                        break
                
                # Phase 3: Filter and deduplicate
                all_contents = self.result_processor.filter_by_relevance(all_contents)
                all_contents = self.result_processor.deduplicate_contents(all_contents)
                
                # Sort by relevance
                all_contents.sort(key=lambda c: c.relevance_score, reverse=True)
                
                self.sources_analyzed = len(all_contents)
                
                # Phase 4: Synthesize answer
                if all_contents:
                    answer_text = await self.deepseek_client.synthesize_answer(
                        question, all_contents
                    )
                    
                    # Extract statistics
                    statistics = await self.deepseek_client.extract_statistics(all_contents)
                else:
                    answer_text = "No relevant information found for this query."
                    statistics = {}
                
                # Create final answer
                answer = Answer(
                    content=answer_text,
                    confidence=self.answer_builder.confidence,
                    sources=[c.url for c in all_contents[:10]],
                    statistics=statistics,
                    gaps=self.answer_builder.gaps,
                    versions=self.answer_builder.answer_versions,
                    generation_time=time.time() - self.start_time
                )
                
                # Create result
                result = ResearchResult(
                    question=question,
                    answer=answer,
                    queries_generated=self.queries_generated,
                    sources_analyzed=self.sources_analyzed,
                    cache_hits=self.cache_hits,
                    total_duration=time.time() - self.start_time,
                    metadata={
                        "relevance_threshold": RELEVANCE_THRESHOLD,
                        "timeout_reached": self._check_timeout(),
                        "serper_requests": self.serper_client.request_count
                    }
                )
                
                # Log summary
                self._log_summary(result)
                
                return result
                
            finally:
                # Cleanup
                if self.cache:
                    await self.cache.close()
    
    def _check_timeout(self) -> bool:
        """Check if research timeout reached"""
        if self.start_time:
            return (time.time() - self.start_time) >= MAX_RESEARCH_TIME
        return False
    
    def _log_summary(self, result: ResearchResult):
        """Log research summary"""
        logger.info("=" * 50)
        logger.info("📊 RESEARCH SUMMARY")
        logger.info(f"❓ Question: {result.question}")
        logger.info(f"🔍 Queries generated: {result.queries_generated}")
        logger.info(f"📄 Sources analyzed: {result.sources_analyzed}")
        logger.info(f"💾 Cache hits: {result.cache_hits}")
        logger.info(f"🎯 Confidence: {result.answer.confidence:.1%}")
        logger.info(f"⏱️ Duration: {result.total_duration:.1f}s")
        logger.info(f"📈 Statistics found: {len(result.answer.statistics.get('numbers_found', []))}")
        logger.info("=" * 50)

# =============================================================================
# Main Execution
# =============================================================================

async def main():
    """Main execution function"""
    # Example research questions
    example_questions = [
        "What are the latest advances in quantum computing?",
        "Compare renewable energy adoption in Europe vs Asia",
        "What are the top emerging cybersecurity threats in 2024?",
        "Explain the impact of AI on healthcare diagnostics"
    ]
    
    # Get question from user or use example
    print("\n" + "="*60)
    print("🔬 DeepSeek Advanced Web Research with Serper MCP v4.01")
    print("="*60)
    
    print("\nExample questions:")
    for i, q in enumerate(example_questions, 1):
        print(f"{i}. {q}")
    
    choice = input("\nEnter question number (1-4) or type your own question: ").strip()
    
    if choice.isdigit() and 1 <= int(choice) <= len(example_questions):
        question = example_questions[int(choice) - 1]
    elif choice:
        question = choice
    else:
        question = example_questions[0]
    
    print(f"\n🔍 Researching: {question}\n")
    
    # Create researcher
    researcher = DeepSeekResearcher()
    
    # Conduct research
    try:
        result = await researcher.research(question)
        
        # Display results
        print("\n" + "="*60)
        print("📝 RESEARCH RESULTS")
        print("="*60)
        
        print(f"\n📌 Question: {result.question}")
        print(f"\n💡 Answer (Confidence: {result.answer.confidence:.1%}):\n")
        print(result.answer.content)
        
        if result.answer.statistics and result.answer.statistics.get("numbers_found"):
            print(f"\n📊 Key Statistics Found:")
            for num in result.answer.statistics["numbers_found"][:5]:
                print(f"  • {num}")
        
        if result.answer.sources:
            print(f"\n🔗 Top Sources:")
            for url in result.answer.sources[:5]:
                print(f"  • {url}")
        
        print(f"\n📈 Research Metrics:")
        print(f"  • Queries generated: {result.queries_generated}")
        print(f"  • Sources analyzed: {result.sources_analyzed}")
        print(f"  • Research duration: {result.total_duration:.1f}s")
        print(f"  • Answer versions: {len(result.answer.versions)}")
        
        # Save results to file
        output_file = f"research_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "question": result.question,
                "answer": result.answer.content,
                "confidence": result.answer.confidence,
                "sources": result.answer.sources,
                "statistics": result.answer.statistics,
                "metadata": result.metadata,
                "duration": result.total_duration
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"❌ Research failed: {e}")
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
