#!/usr/bin/env python3
"""
Research Interfaces - Abstract base classes for research components
This module defines the interfaces that all research components must implement
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ResearchQuery:
    """Represents a research query with metadata"""
    question: str
    query_id: str
    timestamp: datetime
    context: Optional[Dict[str, Any]] = None
    search_mode: str = "standard"
    target_relevance: int = 7
    max_iterations: int = 3


@dataclass
class SearchResult:
    """Represents a search result"""
    title: str
    url: str
    snippet: str
    display_link: str
    search_query: str
    relevance_score: Optional[float] = None


@dataclass
class ExtractedContent:
    """Represents extracted content from a web page"""
    url: str
    title: str
    content: str
    method: str
    word_count: int
    success: bool
    extraction_time: float
    domain_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    from_cache: bool = False
    cache_date: Optional[datetime] = None


@dataclass
class AnalysisResult:
    """Represents analysis results from AI reasoning"""
    original_question: str
    analysis_content: str
    reasoning_content: Optional[str] = None
    relevance_score: Optional[int] = None
    sources_analyzed: int = 0
    model: str = ""
    timestamp: datetime = None
    error: Optional[str] = None


@dataclass
class ResearchResult:
    """Complete research result"""
    query: ResearchQuery
    research_type: str
    success: bool
    direct_answer: Optional[str] = None
    search_results: List[SearchResult] = None
    extracted_contents: List[ExtractedContent] = None
    analysis: Optional[AnalysisResult] = None
    iterations: List[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: datetime = None


class IWebSearchService(ABC):
    """Interface for web search services"""
    
    @abstractmethod
    async def search(self, query: str, num_results: int = 5, **kwargs) -> List[SearchResult]:
        """Perform a web search and return results"""
        pass
    
    @abstractmethod
    async def search_with_filters(self, query: str, num_results: int = 5, 
                                 exclude_domains: List[str] = None,
                                 prefer_domains: List[str] = None) -> List[SearchResult]:
        """Perform a web search with domain filtering"""
        pass


class IContentExtractor(ABC):
    """Interface for content extraction services"""
    
    @abstractmethod
    async def extract_content(self, url: str, keywords: List[str] = None) -> ExtractedContent:
        """Extract content from a URL"""
        pass
    
    @abstractmethod
    def assess_domain_quality(self, url: str) -> Dict[str, Any]:
        """Assess the quality and type of a domain"""
        pass


class ICacheService(ABC):
    """Interface for caching services"""
    
    @abstractmethod
    async def get_cached_content(self, url: str) -> Optional[ExtractedContent]:
        """Get cached content by URL"""
        pass
    
    @abstractmethod
    async def save_content(self, content: ExtractedContent, keywords: List[str]) -> bool:
        """Save content to cache"""
        pass
    
    @abstractmethod
    async def search_cached_content(self, keywords: List[str]) -> List[ExtractedContent]:
        """Search for cached content by keywords"""
        pass
    
    @abstractmethod
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        pass


class IAIReasoningService(ABC):
    """Interface for AI reasoning services"""
    
    @abstractmethod
    async def check_web_search_necessity(self, question: str) -> Dict[str, Any]:
        """Check if web search is necessary for answering the question"""
        pass
    
    @abstractmethod
    async def generate_search_queries(self, question: str, context: Dict[str, Any] = None) -> List[str]:
        """Generate optimized search queries"""
        pass
    
    @abstractmethod
    async def analyze_content_relevance(self, question: str, contents: List[ExtractedContent]) -> AnalysisResult:
        """Analyze content relevance and provide comprehensive answer"""
        pass
    
    @abstractmethod
    async def generate_followup_queries(self, question: str, previous_analysis: AnalysisResult) -> List[str]:
        """Generate follow-up queries based on gaps in previous analysis"""
        pass


class IResearchOrchestrator(ABC):
    """Interface for research orchestrators"""
    
    @abstractmethod
    async def conduct_research(self, query: ResearchQuery) -> ResearchResult:
        """Conduct complete research process"""
        pass
    
    @abstractmethod
    async def conduct_research_stream(self, query: ResearchQuery) -> AsyncGenerator[Dict[str, Any], None]:
        """Conduct research with streaming progress updates"""
        pass


class IProgressCallback(ABC):
    """Interface for progress callbacks"""
    
    @abstractmethod
    async def on_progress(self, step: str, data: Dict[str, Any]):
        """Called when progress is made in research"""
        pass
    
    @abstractmethod
    async def on_error(self, step: str, error: str, data: Dict[str, Any] = None):
        """Called when an error occurs"""
        pass
    
    @abstractmethod
    async def on_complete(self, result: ResearchResult):
        """Called when research is complete"""
        pass


class IMetricsCollector(ABC):
    """Interface for metrics collection"""
    
    @abstractmethod
    def start_timing(self, operation: str) -> str:
        """Start timing an operation, return timing ID"""
        pass
    
    @abstractmethod
    def end_timing(self, timing_id: str) -> float:
        """End timing an operation, return duration"""
        pass
    
    @abstractmethod
    def record_metric(self, name: str, value: Any, tags: Dict[str, str] = None):
        """Record a metric"""
        pass
    
    @abstractmethod
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all collected metrics"""
        pass


class IConfigurationManager(ABC):
    """Interface for configuration management"""
    
    @abstractmethod
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        pass
    
    @abstractmethod
    def get_api_credentials(self, service: str) -> Dict[str, str]:
        """Get API credentials for a service"""
        pass
    
    @abstractmethod
    def get_research_settings(self) -> Dict[str, Any]:
        """Get research-specific settings"""
        pass
