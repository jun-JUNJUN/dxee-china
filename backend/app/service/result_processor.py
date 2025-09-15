#!/usr/bin/env python3
"""
Result Processing and Content Evaluation System
Extracted from test_deepseek_advanced_web_research4_01.py

This module provides search result processing, content scraping, relevance evaluation,
and quality assessment for web research.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Import token manager
from app.service.token_manager import TokenManager

logger = logging.getLogger(__name__)

# Configuration constants
RELEVANCE_THRESHOLD = 0.7  # 70% relevance threshold


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
class ScoredContent:
    """Content with relevance score"""
    url: str
    title: str
    content: str
    relevance_score: float  # 0-1 scale
    confidence: float = 0.5
    source_quality: int = 5  # 1-10 scale
    extraction_method: str = "unknown"
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ResultProcessor:
    """Process and analyze search results"""

    def __init__(self, deepseek_client, serper_client):
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

            if scraped and hasattr(scraped, 'text') and scraped.text:
                content_text = scraped.text
                extraction_method = "scrape"
                confidence = 0.8  # Higher confidence for scraped content
            else:
                # Fallback to snippet
                content_text = result.snippet
                extraction_method = "snippet"
                confidence = 0.5  # Lower confidence for snippet

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
                confidence=confidence,
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
        # High-quality domains (including specific academic institutions)
        high_quality = [
            "wikipedia.org", "nature.com", "science.org",
            "ieee.org", "acm.org", "arxiv.org",
            "harvard.edu"  # Only harvard.edu gets 9, not all top universities
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

    def rank_by_relevance_and_quality(self, contents: List[ScoredContent]) -> List[ScoredContent]:
        """Sort contents by combined relevance and quality scores

        Combines relevance score (0-1) and source quality (1-10) with 70/30 weighting
        to prioritize relevant content from high-quality sources.

        Args:
            contents: List of ScoredContent objects to rank

        Returns:
            List sorted by combined score (highest first)
        """
        def combined_score(content: ScoredContent) -> float:
            # Normalize source quality to 0-1 scale (quality is 1-10)
            normalized_quality = (content.source_quality - 1) / 9.0
            # Weight relevance more heavily (70%) than quality (30%)
            return (content.relevance_score * 0.7) + (normalized_quality * 0.3)

        return sorted(contents, key=combined_score, reverse=True)