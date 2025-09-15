#!/usr/bin/env python3
"""
Serper API Integration Client
Extracted from test_deepseek_advanced_web_research4_01.py

This module provides web search and content scraping capabilities through the Serper API,
implementing advanced search operators and professional web research functionality.
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from urllib.parse import urlparse

# Import HTTP client
try:
    import aiohttp
except ImportError:
    aiohttp = None
    logging.error("aiohttp is required for SerperClient")

# Import query models
from app.service.deep_thinking_engine import SearchQuery, SearchType

logger = logging.getLogger(__name__)

# Configuration constants from the test algorithm
SERPER_BASE_URL = "https://google.serper.dev"
SERPER_SCRAPE_URL = "https://scrape.serper.dev"


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


class SerperClient:
    """Serper API client with MCP patterns"""

    def __init__(self, api_key: str, rate_limit_delay: float = 1.0):
        if not api_key:
            raise ValueError("Serper API key is required")

        self.api_key = api_key
        self.base_url = SERPER_BASE_URL
        self.scrape_url = SERPER_SCRAPE_URL
        self.session = None
        self.request_count = 0
        self.last_request_time = 0
        self.rate_limit_delay = rate_limit_delay  # Minimum seconds between requests

    async def __aenter__(self):
        """Async context manager entry"""
        if aiohttp is None:
            raise ImportError("aiohttp is required for SerperClient")
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

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            return urlparse(url).netloc
        except:
            return ""

    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Execute web search with advanced operators and rate limiting

        Performs a web search using the Serper API with professional search operators
        including site restrictions, file type filtering, and temporal constraints.

        Args:
            query: SearchQuery object containing search parameters and operators

        Returns:
            List of SearchResult objects with URL, title, snippet, and metadata

        Raises:
            No exceptions raised - returns empty list on API or network errors
        """
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
                        domain=self._extract_domain(item.get("link", "")),
                        cached_url=item.get("cachedPageLink")
                    )
                    results.append(result)

                logger.info(f"🔍 Serper search completed: {len(results)} results")
                return results

        except Exception as e:
            logger.error(f"❌ Serper search failed: {e}")
            self.request_count += 1  # Count failed requests too
            return []

    async def scrape(self, url: str, include_markdown: bool = True) -> Optional[ScrapeResult]:
        """Scrape webpage content with markdown extraction and metadata processing

        Extracts full page content from a URL using Serper's scraping service,
        providing both plain text and optionally formatted markdown output.

        Args:
            url: The webpage URL to scrape
            include_markdown: Whether to include markdown-formatted content

        Returns:
            ScrapeResult object with text, markdown, metadata and timing info,
            or None if scraping fails

        Raises:
            No exceptions raised - returns None on timeout, API errors, or network failures
        """
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
            timeout = aiohttp.ClientTimeout(total=30)
            async with self.session.post(
                self.scrape_url,
                json=params,
                headers=headers,
                timeout=timeout
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
            self.request_count += 1  # Count timed out requests too
            return None
        except Exception as e:
            logger.error(f"❌ Scrape error for {url}: {e}")
            self.request_count += 1  # Count failed requests too
            return None

    def get_request_count(self) -> int:
        """Get the current request count"""
        return self.request_count