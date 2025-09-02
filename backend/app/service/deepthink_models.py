#!/usr/bin/env python3
"""
Data models and utility functions for deep-think functionality
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
import re

try:
    import tiktoken
except ImportError:
    tiktoken = None


@dataclass
class DeepThinkRequest:
    """Request model for deep-think processing"""
    request_id: str
    question: str  # Changed from user_query to question for consistency
    chat_id: str
    user_id: str
    timestamp: datetime
    timeout_seconds: int = 600
    max_queries: int = 5


@dataclass
class SearchQuery:
    """Model for generated search queries"""
    text: str
    query_type: str  # Changed from search_type for consistency
    priority: int
    operators: Dict[str, str] = field(default_factory=dict)  # Changed from advanced_operators
    expected_results: str = "General search results"  # Added for orchestrator compatibility


@dataclass
class ScrapedContent:
    """Model for scraped web content"""
    url: str
    title: str
    text_content: str
    markdown_content: str
    word_count: int = 0
    extraction_timestamp: datetime = field(default_factory=datetime.now)  # Changed from extraction_time
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RelevanceScore:
    """Model for content relevance evaluation"""
    score: float
    reasoning: str
    confidence: float
    key_points: List[str] = field(default_factory=list)
    content_url: str = ""


@dataclass
class ReasoningChain:
    """Model for Jan-style reasoning chains"""
    premise: str
    reasoning: str
    conclusion: str
    confidence: float
    supporting_evidence: List[str] = field(default_factory=list)
    logical_steps: List[str] = field(default_factory=list)
    source_urls: List[str] = field(default_factory=list)


@dataclass
class DeepThinkResult:
    """Model for final deep-think results"""
    request_id: str
    question: str
    comprehensive_answer: str
    summary_answer: str  # Changed from summary
    search_queries: List[SearchQuery] = field(default_factory=list)
    scraped_content: List[ScrapedContent] = field(default_factory=list)
    relevance_scores: List[RelevanceScore] = field(default_factory=list)
    reasoning_chains: List[ReasoningChain] = field(default_factory=list)
    confidence_score: float = 0.0
    processing_time: float = 0.0
    total_sources: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count tokens in text using tiktoken
    Falls back to character count / 4 if tiktoken is unavailable
    """
    if not text:
        return 0
    
    try:
        if tiktoken is None:
            # Fallback: approximate 4 characters per token
            return len(text) // 4
        
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback: approximate 4 characters per token
        return len(text) // 4


def summarize_content(content: str, max_length: int = 500) -> str:
    """
    Summarize content to fit within max_length characters
    Uses intelligent sentence selection for longer content
    """
    if not content:
        return content
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', content)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= 3:
        # Few sentences - check if truncation is needed
        if len(content) <= max_length:
            return content
        else:
            truncated = content[:max_length-3]
            return truncated + "..."
    
    # Many sentences (>3) - use intelligent selection
    if len(sentences) > 4:
        selected = []
        
        # Always include first 2 sentences
        selected.append(sentences[0] + ".")
        selected.append(sentences[1] + ".")
        
        # Add middle sentence
        middle_idx = len(sentences) // 2
        selected.append(sentences[middle_idx] + ".")
        
        # Add last sentence
        selected.append(sentences[-1] + ".")
        
        result = " ".join(selected)
        
        # If result exceeds max_length, fallback to truncation
        if len(result) > max_length:
            truncated = content[:max_length-3]
            return truncated + "..."
        
        return result
    
    # 4 sentences - include all if within limit, otherwise truncate
    if len(content) <= max_length:
        return content
    else:
        truncated = content[:max_length-3]
        return truncated + "..."


@dataclass
class ProgressUpdate:
    """Model for real-time progress updates"""
    step: int
    total_steps: int
    description: str
    progress_percent: int
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DeepThinkStats:
    """Statistics tracking for deep-think orchestrator"""
    total_requests: int = 0
    successful_requests: int = 0
    error_requests: int = 0
    timeout_errors: int = 0
    total_processing_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0