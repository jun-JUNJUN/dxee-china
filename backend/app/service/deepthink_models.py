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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage"""
        return {
            'request_id': self.request_id,
            'question': self.question,
            'chat_id': self.chat_id,
            'user_id': self.user_id,
            'timestamp': self.timestamp,
            'timeout_seconds': self.timeout_seconds,
            'max_queries': self.max_queries
        }


@dataclass
class SearchQuery:
    """Model for generated search queries"""
    text: str
    query_type: str  # Changed from search_type for consistency
    priority: int
    operators: Dict[str, str] = field(default_factory=dict)  # Changed from advanced_operators
    expected_results: str = "General search results"  # Added for orchestrator compatibility
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage"""
        return {
            'text': self.text,
            'query_type': self.query_type,
            'priority': self.priority,
            'operators': self.operators,
            'expected_results': self.expected_results
        }


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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage"""
        return {
            'url': self.url,
            'title': self.title,
            'text_content': self.text_content,
            'markdown_content': self.markdown_content,
            'word_count': self.word_count,
            'extraction_timestamp': self.extraction_timestamp,
            'metadata': self.metadata
        }


@dataclass
class RelevanceScore:
    """Model for content relevance evaluation"""
    score: float
    reasoning: str
    confidence: float
    key_points: List[str] = field(default_factory=list)
    content_url: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage"""
        return {
            'score': self.score,
            'reasoning': self.reasoning,
            'confidence': self.confidence,
            'key_points': self.key_points,
            'content_url': self.content_url
        }


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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage"""
        return {
            'premise': self.premise,
            'reasoning': self.reasoning,
            'conclusion': self.conclusion,
            'confidence': self.confidence,
            'supporting_evidence': self.supporting_evidence,
            'logical_steps': self.logical_steps,
            'source_urls': self.source_urls
        }


@dataclass
class ProgressUpdate:
    """Model for streaming progress updates"""
    step: int
    total_steps: int
    description: str
    progress_percent: float
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'step': self.step,
            'total_steps': self.total_steps,
            'description': self.description,
            'progress_percent': self.progress_percent,
            'details': self.details,
            'timestamp': self.timestamp
        }


@dataclass
class DeepThinkStats:
    """Model for deep-think processing statistics"""
    total_requests: int = 0
    successful_requests: int = 0
    error_requests: int = 0
    timeout_errors: int = 0
    total_processing_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'error_requests': self.error_requests,
            'timeout_errors': self.timeout_errors,
            'total_processing_time': self.total_processing_time,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses
        }


@dataclass
class Answer:
    """Synthesized answer - matches test file structure exactly"""
    content: str
    confidence: float
    sources: List[str]
    statistics: Optional[Dict] = None
    gaps: List[str] = field(default_factory=list)
    versions: List[Dict] = field(default_factory=list)
    generation_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization"""
        return {
            'content': self.content,
            'confidence': self.confidence,
            'sources': self.sources,
            'statistics': self.statistics,
            'gaps': self.gaps,
            'versions': self.versions,
            'generation_time': self.generation_time
        }


@dataclass
class DeepThinkResult:
    """Model for final deep-think results - similar to ResearchResult from test file"""
    request_id: str
    question: str
    answer: Answer  # Structured answer object matching test file
    queries_generated: int
    sources_analyzed: int
    cache_hits: int = 0
    total_duration: float = 0.0
    # Keep additional fields for backward compatibility and extended functionality
    comprehensive_answer: str = ""  # Keep for backward compatibility
    summary_answer: str = ""  # Keep for backward compatibility
    search_queries: List[SearchQuery] = field(default_factory=list)
    scraped_content: List[ScrapedContent] = field(default_factory=list)
    relevance_scores: List[RelevanceScore] = field(default_factory=list)
    reasoning_chains: List[ReasoningChain] = field(default_factory=list)
    confidence_score: float = 0.0
    processing_time: float = 0.0
    total_sources: int = 0
    cache_misses: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage"""
        # Handle metadata properly - convert QuestionAnalysis if present
        metadata_dict = dict(self.metadata) if self.metadata else {}
        if 'question_analysis' in metadata_dict:
            analysis = metadata_dict['question_analysis']
            if hasattr(analysis, 'to_dict'):
                metadata_dict['question_analysis'] = analysis.to_dict()
            elif hasattr(analysis, '__dict__'):
                # Fallback for objects without to_dict method
                analysis_dict = dict(analysis.__dict__)
                # Convert enum values to strings
                if 'complexity' in analysis_dict and hasattr(analysis_dict['complexity'], 'value'):
                    analysis_dict['complexity'] = analysis_dict['complexity'].value
                metadata_dict['question_analysis'] = analysis_dict
        
        return {
            'request_id': self.request_id,
            'question': self.question,
            'answer': self.answer.to_dict() if self.answer else None,
            'queries_generated': self.queries_generated,
            'sources_analyzed': self.sources_analyzed,
            'cache_hits': self.cache_hits,
            'total_duration': self.total_duration,
            'comprehensive_answer': self.comprehensive_answer,
            'summary_answer': self.summary_answer,
            'search_queries': [q.to_dict() for q in self.search_queries],
            'scraped_content': [c.to_dict() for c in self.scraped_content],
            'relevance_scores': [r.to_dict() for r in self.relevance_scores],
            'reasoning_chains': [rc.to_dict() for rc in self.reasoning_chains],
            'confidence_score': self.confidence_score,
            'processing_time': self.processing_time,
            'total_sources': self.total_sources,
            'cache_misses': self.cache_misses,
            'timestamp': self.timestamp,
            'metadata': metadata_dict
        }


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
