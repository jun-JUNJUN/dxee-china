#!/usr/bin/env python3
"""
Content Extraction Pipeline
Orchestrates content scraping and extraction with caching, timeout handling, and quality assessment

This module provides a high-level pipeline for extracting content from web search results,
implementing graceful fallback strategies and comprehensive quality tracking.
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_EXTRACTION_TIMEOUT = 30.0  # seconds
MAX_CONCURRENT_EXTRACTIONS = 5


class ExtractionMethod(Enum):
    """Content extraction method"""
    SCRAPE = "scrape"
    CACHE = "cache"
    SNIPPET = "snippet"


class CacheStatus(Enum):
    """Cache interaction status"""
    HIT = "hit"
    MISS_STORED = "miss_stored"
    MISS_NOT_STORED = "miss_not_stored"
    DISABLED = "disabled"


@dataclass
class ExtractionResult:
    """Result of content extraction with metadata"""
    url: str
    text: str
    method: ExtractionMethod
    cache_status: CacheStatus = CacheStatus.DISABLED
    markdown: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: int = 5
    extraction_time: float = 0.0
    timeout_occurred: bool = False
    quality_factors: Dict[str, Any] = field(default_factory=dict)


class ContentExtractionPipeline:
    """High-level content extraction pipeline with caching and quality assessment"""

    def __init__(self, serper_client, cache_service=None, extraction_timeout=DEFAULT_EXTRACTION_TIMEOUT):
        self.serper_client = serper_client
        self.cache_service = cache_service
        self.extraction_timeout = extraction_timeout
        self.extraction_stats = {
            "total_attempts": 0,
            "successful_scrapes": 0,
            "cache_hits": 0,
            "snippet_fallbacks": 0,
            "timeout_occurred": 0
        }

    async def extract_content(self, search_result) -> Optional[ExtractionResult]:
        """Extract content from a single search result with comprehensive fallback strategies

        Implements a multi-stage extraction pipeline:
        1. Check cache for existing content (if cache service available)
        2. Attempt content scraping with configurable timeout
        3. Fallback to search result snippet if scraping fails
        4. Assess content quality and track extraction statistics

        Args:
            search_result: SearchResult object containing URL, title, snippet, and metadata

        Returns:
            ExtractionResult with extracted content, quality score, and extraction metadata,
            or None only in case of unexpected errors (graceful degradation ensures a result)

        Note:
            This method never returns None under normal circumstances as it implements
            comprehensive fallback strategies including snippet extraction.
        """
        start_time = time.time()
        self.extraction_stats["total_attempts"] += 1

        try:
            # Step 1: Check cache if available
            if self.cache_service:
                cached_content = await self.cache_service.get_cached_content(search_result.url)
                if cached_content:
                    self.extraction_stats["cache_hits"] += 1

                    result = ExtractionResult(
                        url=search_result.url,
                        text=cached_content.get("text", ""),
                        markdown=cached_content.get("markdown"),
                        method=ExtractionMethod.CACHE,
                        cache_status=CacheStatus.HIT,
                        metadata=cached_content.get("metadata", {}),
                        extraction_time=time.time() - start_time
                    )

                    result.quality_score = self._assess_content_quality(result)
                    return result

            # Step 2: Attempt content scraping with timeout
            timeout_occurred = False
            scraped_result = None

            try:
                scraped_result = await asyncio.wait_for(
                    self.serper_client.scrape(search_result.url, include_markdown=True),
                    timeout=self.extraction_timeout
                )
            except asyncio.TimeoutError:
                timeout_occurred = True
                self.extraction_stats["timeout_occurred"] += 1
                logger.warning(f"⏱️ Scraping timeout for {search_result.url}")

            # Step 3: Process successful scraping
            if scraped_result and scraped_result.text:
                self.extraction_stats["successful_scrapes"] += 1

                result = ExtractionResult(
                    url=search_result.url,
                    text=scraped_result.text,
                    markdown=scraped_result.markdown,
                    method=ExtractionMethod.SCRAPE,
                    metadata=scraped_result.metadata,
                    extraction_time=scraped_result.extraction_time,
                    timeout_occurred=timeout_occurred
                )

                # Store in cache if available
                if self.cache_service:
                    try:
                        await self.cache_service.store_content(
                            search_result.url,
                            {
                                "text": scraped_result.text,
                                "markdown": scraped_result.markdown,
                                "metadata": scraped_result.metadata
                            }
                        )
                        result.cache_status = CacheStatus.MISS_STORED
                    except Exception as e:
                        logger.warning(f"Failed to cache content for {search_result.url}: {e}")
                        result.cache_status = CacheStatus.MISS_NOT_STORED
                else:
                    result.cache_status = CacheStatus.DISABLED

                result.quality_score = self._assess_content_quality(result)
                return result

            # Step 4: Fallback to snippet
            self.extraction_stats["snippet_fallbacks"] += 1

            result = ExtractionResult(
                url=search_result.url,
                text=search_result.snippet,
                method=ExtractionMethod.SNIPPET,
                cache_status=CacheStatus.MISS_NOT_STORED if self.cache_service else CacheStatus.DISABLED,
                extraction_time=time.time() - start_time,
                timeout_occurred=timeout_occurred
            )

            result.quality_score = self._assess_content_quality(result)
            return result

        except Exception as e:
            logger.error(f"❌ Content extraction failed for {search_result.url}: {e}")

            # Emergency fallback to snippet
            result = ExtractionResult(
                url=search_result.url,
                text=search_result.snippet,
                method=ExtractionMethod.SNIPPET,
                cache_status=CacheStatus.MISS_NOT_STORED if self.cache_service else CacheStatus.DISABLED,
                extraction_time=time.time() - start_time
            )

            result.quality_score = self._assess_content_quality(result)
            return result

    async def extract_multiple_contents(self, search_results) -> List[ExtractionResult]:
        """Extract content from multiple search results concurrently"""
        # Create semaphore to limit concurrent extractions
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_EXTRACTIONS)

        async def extract_with_semaphore(search_result):
            async with semaphore:
                return await self.extract_content(search_result)

        # Execute extractions concurrently
        tasks = [extract_with_semaphore(result) for result in search_results]
        extraction_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and None results
        valid_results = []
        for result in extraction_results:
            if isinstance(result, ExtractionResult):
                valid_results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Extraction task failed: {result}")

        return valid_results

    def _assess_content_quality(self, extraction_result: ExtractionResult) -> int:
        """Assess content quality on 1-10 scale"""
        quality_factors = {}
        base_score = 5

        # Factor 1: Extraction method
        if extraction_result.method == ExtractionMethod.SCRAPE:
            base_score += 2
            quality_factors["extraction_method"] = "scrape_bonus"
        elif extraction_result.method == ExtractionMethod.CACHE:
            base_score += 1
            quality_factors["extraction_method"] = "cache_bonus"
        else:  # SNIPPET
            base_score -= 2
            quality_factors["extraction_method"] = "snippet_penalty"

        # Factor 2: Content length
        content_length = len(extraction_result.text)
        if content_length > 1000:
            base_score += 2
            quality_factors["content_length"] = "comprehensive"
        elif content_length > 500:
            base_score += 1
            quality_factors["content_length"] = "good"
        elif content_length > 50:
            quality_factors["content_length"] = "adequate"
        else:
            base_score -= 1
            quality_factors["content_length"] = "brief"

        # Factor 3: Metadata presence
        if extraction_result.metadata and len(extraction_result.metadata) > 0:
            base_score += 1
            quality_factors["has_metadata"] = True
        else:
            quality_factors["has_metadata"] = False

        # Factor 4: Markdown formatting
        if extraction_result.markdown:
            base_score += 1
            quality_factors["has_markdown"] = True
        else:
            quality_factors["has_markdown"] = False

        # Factor 5: Extraction speed (for scraping)
        if extraction_result.method == ExtractionMethod.SCRAPE:
            if extraction_result.extraction_time < 2.0:
                quality_factors["extraction_speed"] = "fast"
            elif extraction_result.extraction_time > 5.0:
                base_score -= 1
                quality_factors["extraction_speed"] = "slow"
            else:
                quality_factors["extraction_speed"] = "normal"

        # Store quality factors for debugging
        extraction_result.quality_factors = quality_factors

        # Ensure score is within valid range
        final_score = max(1, min(10, base_score))
        return final_score

    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get extraction statistics"""
        total = self.extraction_stats["total_attempts"]

        stats = dict(self.extraction_stats)

        if total > 0:
            stats["success_rate"] = self.extraction_stats["successful_scrapes"] / total
            stats["cache_hit_rate"] = self.extraction_stats["cache_hits"] / total
            stats["fallback_rate"] = self.extraction_stats["snippet_fallbacks"] / total
            stats["timeout_rate"] = self.extraction_stats["timeout_occurred"] / total
        else:
            stats["success_rate"] = 0.0
            stats["cache_hit_rate"] = 0.0
            stats["fallback_rate"] = 0.0
            stats["timeout_rate"] = 0.0

        return stats