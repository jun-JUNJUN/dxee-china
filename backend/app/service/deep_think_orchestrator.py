#!/usr/bin/env python3
"""
Simplified Deep Think Orchestrator

Streamlined version of the advanced web research workflow from test_deepseek_advanced_web_research4_01.py
Focuses on essential features for production deployment with session management and HTML caching.

Key Features:
- Multi-query generation with DeepSeek LLM
- Serper API integration with HTML cache service
- Relevance evaluation with 70% threshold filtering
- Progressive answer synthesis with confidence tracking
- Session-based processing for frontend disconnection resilience
- SSE progress streaming for real-time updates

Architecture:
- Simplified from 10 phases to 6 core phases
- Session management for background processing
- HTML caching with access counters
- Statistical analysis and source attribution
- Graceful error handling and timeout management
"""

import asyncio
import logging
import json
import time
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum

# Import existing services
from .deepseek_service import DeepSeekService
from .serper_api_client import SerperAPIClient
from .html_cache_service import HTMLCacheService
from .mongodb_service import MongoDBService
from .session_manager_service import SessionManagerService

# Import session models
from ..models.session_models import (
    ProcessingSession, SessionStatus, SessionStep, 
    SessionProgress, SessionResult
)

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration Constants
# =============================================================================

# Research Configuration
MAX_RESEARCH_TIME = 600  # 10 minutes
MAX_QUERIES_PER_RESEARCH = 6  # Simplified from 10
MAX_RESULTS_PER_QUERY = 8   # Simplified from 10
RELEVANCE_THRESHOLD = 0.6  # 60% relevance threshold (more lenient than 70%)
MAX_CONTENT_LENGTH = 2000   # Token optimization

# Progress Steps for SSE Streaming
class ResearchStep(Enum):
    INITIALIZING = "Initializing session"
    ANALYZING_QUESTION = "Analyzing question"
    GENERATING_QUERIES = "Generating search queries" 
    SEARCHING_WEB = "Searching web sources"
    EXTRACTING_CONTENT = "Extracting content"
    EVALUATING_RELEVANCE = "Evaluating relevance"
    SYNTHESIZING_ANSWER = "Synthesizing final answer"
    COMPLETING = "Completing research"

# =============================================================================
# Data Models
# =============================================================================

@dataclass
class SearchQuery:
    """Simplified search query model"""
    text: str
    priority: int = 1
    num_results: int = MAX_RESULTS_PER_QUERY
    
@dataclass
class ScoredContent:
    """Content with relevance score and metadata"""
    url: str
    title: str
    content: str
    relevance_score: float  # 0-1 scale
    confidence: float
    source_quality: int  # 1-10 scale
    from_cache: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class ResearchAnswer:
    """Synthesized research answer"""
    content: str
    confidence: float
    sources: List[str]
    statistics: Dict[str, Any] = field(default_factory=dict)
    generation_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB serialization"""
        return {
            'content': self.content,
            'confidence': self.confidence,
            'sources': self.sources,
            'statistics': self.statistics,
            'generation_time': self.generation_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResearchAnswer':
        """Create from dictionary for MongoDB deserialization"""
        return cls(
            content=data.get('content', ''),
            confidence=data.get('confidence', 0.0),
            sources=data.get('sources', []),
            statistics=data.get('statistics', {}),
            generation_time=data.get('generation_time', 0.0)
        )

@dataclass
class ResearchResult:
    """Complete research output"""
    question: str
    answer: ResearchAnswer
    session_id: str
    queries_generated: int
    sources_analyzed: int
    cache_hits: int = 0
    total_duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProgressUpdate:
    """SSE progress update"""
    session_id: str
    step: ResearchStep
    progress: int  # 0-100
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)

# =============================================================================
# Simplified Deep Think Orchestrator
# =============================================================================

class DeepThinkOrchestrator:
    """
    Simplified research orchestrator based on test_deepseek_advanced_web_research4_01.py
    
    Streamlines the 10-phase workflow into 6 core phases:
    1. Question Analysis & Query Generation
    2. Web Search with Caching
    3. Content Extraction & Relevance Evaluation  
    4. Statistical Analysis & Source Attribution
    5. Answer Synthesis
    6. Result Processing & Storage
    """
    
    def __init__(self, 
                 deepseek_service: DeepSeekService,
                 serper_client: SerperAPIClient,
                 html_cache_service: HTMLCacheService,
                 mongodb_service: MongoDBService,
                 session_manager: Optional[SessionManagerService] = None):
        self.deepseek = deepseek_service
        self.serper = serper_client
        self.cache = html_cache_service
        self.mongodb = mongodb_service
        
        # Session manager for persistent session tracking
        self.session_manager = session_manager
        
        # Fallback in-memory tracking if no session manager
        self.active_sessions = {}
        
        # Performance metrics
        self.total_sessions = 0
        self.successful_sessions = 0
        self.cache_hit_rate = 0.0
        
        logger.info("DeepThinkOrchestrator initialized with session management support")
    
    async def start_research_session(self, question: str, session_id: str, 
                                   chat_id: str = None, user_id: str = None) -> AsyncGenerator[ProgressUpdate, None]:
        """
        Start research session with progress streaming and session management
        
        Args:
            question: Research question to investigate
            session_id: Unique session identifier
            chat_id: Associated chat ID (optional)
            user_id: User ID for session tracking (optional)
            
        Yields:
            ProgressUpdate: Real-time progress updates via SSE
        """
        start_time = time.time()
        self.total_sessions += 1
        
        # Track session (fallback to in-memory if no session manager)
        if self.session_manager:
            # Session should already be created by caller, just start it
            await self.session_manager.start_session(session_id)
        else:
            self.active_sessions[session_id] = {
                'question': question,
                'start_time': start_time,
                'status': 'running'
            }
        
        try:
            logger.info(f"Starting research session {session_id}: {question}")
            
            # Phase 1: Initialize and analyze question (0-15%)
            await self._update_session_progress(session_id, SessionStep.INITIALIZING, 5, 
                                              "Setting up research session")
            
            yield ProgressUpdate(
                session_id=session_id,
                step=ResearchStep.INITIALIZING,
                progress=5,
                description="Setting up research session"
            )
            
            await self._update_session_progress(session_id, SessionStep.ANALYZING_QUESTION, 10,
                                              "Analyzing question complexity")
            
            yield ProgressUpdate(
                session_id=session_id,
                step=ResearchStep.ANALYZING_QUESTION,
                progress=10,
                description="Analyzing question complexity"
            )
            
            analysis = await self._analyze_question(question)
            
            # Phase 2: Generate search queries (15-25%)
            yield ProgressUpdate(
                session_id=session_id,
                step=ResearchStep.GENERATING_QUERIES,
                progress=20,
                description=f"Generating queries for: {analysis.get('main_topic', 'topic')}"
            )
            
            queries = await self._generate_queries(question, analysis)
            
            # Phase 3: Execute searches with caching (25-60%)
            yield ProgressUpdate(
                session_id=session_id,
                step=ResearchStep.SEARCHING_WEB,
                progress=30,
                description=f"Searching {len(queries)} query variations"
            )
            
            all_contents = []
            cache_hits = 0
            
            # REPLICATE BACKEND PATTERN: Process query by query with stopping logic
            for i, query in enumerate(queries):
                # CRITICAL FIX 1: Check timeout (like backend line 1401)
                if self._check_timeout(start_time):
                    logger.warning(f"⏰ Research timeout reached for session {session_id}")
                    break
                
                # Update progress
                search_progress = 30 + int((i / len(queries)) * 30)  # 30-60%
                yield ProgressUpdate(
                    session_id=session_id,
                    step=ResearchStep.SEARCHING_WEB,
                    progress=search_progress,
                    description=f"Searching: {query.text[:50]}..."
                )
                
                # Execute search (like backend line 1407)
                logger.info(f"🔍 Searching: {query.text}")
                search_results = await self.serper.search(
                    query=query.text,
                    num_results=query.num_results,
                    use_queue=True
                )
                
                # Process results EXACTLY like backend (lines 1410-1412 in test file)
                contents = await self._process_search_results_like_backend(
                    search_results.get('organic', []), question
                )
                all_contents.extend(contents)
                
                # CRITICAL FIX 2: Check stopping condition exactly like backend (lines 1420-1423)
                relevant_contents = [c for c in all_contents if c.relevance_score >= RELEVANCE_THRESHOLD]
                if len(relevant_contents) >= 10:
                    logger.info("✅ Sufficient relevant content found (like backend lines 1420-1423)")
                    break  # ← STOPS THE LOOP EXACTLY LIKE BACKEND!
                
                logger.info(f"Query {i+1}/{len(queries)}: Found {len(contents)} contents, {len(relevant_contents)} relevant so far")
            
            # Phase 4: Content already evaluated in backend-like approach, just filter
            yield ProgressUpdate(
                session_id=session_id,
                step=ResearchStep.EVALUATING_RELEVANCE,
                progress=65,
                description=f"Filtering {len(all_contents)} already-evaluated sources"
            )
            
            # CRITICAL FIX: Content already has relevance scores from backend-style processing
            # Don't re-evaluate! Just filter by threshold to prevent infinite loop
            relevant_contents = [c for c in all_contents if c.relevance_score >= RELEVANCE_THRESHOLD]
            
            yield ProgressUpdate(
                session_id=session_id,
                step=ResearchStep.EVALUATING_RELEVANCE,
                progress=70,
                description=f"Found {len(relevant_contents)} relevant sources"
            )
            
            # Phase 5: Synthesize answer (75-90%)
            yield ProgressUpdate(
                session_id=session_id,
                step=ResearchStep.SYNTHESIZING_ANSWER,
                progress=80,
                description="Synthesizing comprehensive answer"
            )
            
            answer = await self._synthesize_answer(question, relevant_contents)
            
            # Phase 6: Complete and store results (90-100%)
            yield ProgressUpdate(
                session_id=session_id,
                step=ResearchStep.COMPLETING,
                progress=95,
                description="Finalizing research results"
            )
            
            # Create final result
            result = ResearchResult(
                question=question,
                answer=answer,
                session_id=session_id,
                queries_generated=len(queries),
                sources_analyzed=len(relevant_contents),
                cache_hits=cache_hits,
                total_duration=time.time() - start_time,
                metadata={
                    'relevance_threshold': RELEVANCE_THRESHOLD,
                    'timeout_reached': self._check_timeout(start_time),
                    'analysis': analysis
                }
            )
            
            # Store result in MongoDB
            await self._store_research_result(result)
            
            # Complete session using session manager
            session_result = SessionResult(
                answer_content=result.answer.content,
                confidence=result.answer.confidence,
                sources=result.answer.sources,
                statistics=result.answer.statistics,
                generation_time=result.answer.generation_time,
                total_duration=result.total_duration,
                queries_generated=result.queries_generated,
                sources_analyzed=result.sources_analyzed,
                cache_hits=result.cache_hits,
                metadata=result.metadata
            )
            
            await self._complete_session(session_id, session_result)
            
            # Calculate cache hit rate
            if len(all_contents) > 0:
                self.cache_hit_rate = cache_hits / len(all_contents)
            
            # Final progress update with session completion
            await self._update_session_progress(session_id, SessionStep.COMPLETING, 100,
                                              f"Research completed in {result.total_duration:.1f}s",
                                              {'result': {
                                                  'question': result.question,
                                                  'answer': result.answer.to_dict(),
                                                  'session_id': result.session_id,
                                                  'queries_generated': result.queries_generated,
                                                  'sources_analyzed': result.sources_analyzed,
                                                  'cache_hits': result.cache_hits,
                                                  'total_duration': result.total_duration,
                                                  'metadata': result.metadata
                                              }})
            
            yield ProgressUpdate(
                session_id=session_id,
                step=ResearchStep.COMPLETING,
                progress=100,
                description=f"Research completed in {result.total_duration:.1f}s",
                metadata={
                    'result': {
                        'answer': result.answer.to_dict(),
                        'queries_generated': result.queries_generated,
                        'sources_analyzed': result.sources_analyzed,
                        'cache_hits': result.cache_hits,
                        'total_duration': result.total_duration,
                        'metadata': result.metadata
                    }
                }
            )
            
            logger.info(f"Research session {session_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Research session {session_id} failed: {e}")
            
            # Fail session using session manager
            await self._fail_session(session_id, str(e))
            
            yield ProgressUpdate(
                session_id=session_id,
                step=ResearchStep.COMPLETING,
                progress=100,
                description=f"Research failed: {str(e)[:100]}"
            )
            raise
    
    async def _analyze_question(self, question: str) -> Dict[str, Any]:
        """Analyze research question using DeepSeek LLM"""
        prompt = f"""Analyze this research question and provide a JSON response:

Question: {question}

Provide:
1. main_topic: Primary subject (max 3 words)
2. intent: What user wants (definition/comparison/statistics/analysis)
3. scope: Time scope (current/historical/future)
4. complexity: Simple/Medium/Complex
5. domain: Field (technology/business/science/general)

JSON:"""
        
        try:
            response = await self.deepseek.async_chat_completion(
                query=prompt,
                system_message="You are a helpful research analyst",
                search_mode="deep"
            )
            
            # Parse JSON from response
            content = response.get('message', '')
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                logger.info(f"Question analysis: {analysis.get('main_topic', 'Unknown')}")
                return analysis
        except Exception as e:
            logger.warning(f"Question analysis failed: {e}")
        
        # Fallback analysis
        return {
            "main_topic": question.split()[:3],  # First 3 words
            "intent": "general",
            "scope": "current", 
            "complexity": "medium",
            "domain": "general"
        }
    
    async def _generate_queries(self, question: str, analysis: Dict[str, Any]) -> List[SearchQuery]:
        """Generate search queries based on question analysis"""
        main_topic = analysis.get('main_topic', question)
        intent = analysis.get('intent', 'general')
        
        queries = []
        
        # Core factual query
        queries.append(SearchQuery(
            text=f"{main_topic} overview definition",
            priority=9
        ))
        
        # Intent-specific queries
        if intent == "comparison":
            queries.append(SearchQuery(
                text=f"{main_topic} comparison analysis",
                priority=8
            ))
        elif intent == "statistics":
            queries.append(SearchQuery(
                text=f"{main_topic} statistics data numbers",
                priority=8
            ))
        elif intent == "analysis":
            queries.append(SearchQuery(
                text=f"{main_topic} impact effects analysis",
                priority=8
            ))
        
        # Recent developments
        queries.append(SearchQuery(
            text=f"latest {main_topic} 2024",
            priority=7
        ))
        
        # Expert sources
        queries.append(SearchQuery(
            text=f"{main_topic} research report",
            priority=6
        ))
        
        # Industry/academic perspective
        queries.append(SearchQuery(
            text=f"{main_topic} industry trends",
            priority=6
        ))
        
        # Ensure we have exact number requested
        return queries[:MAX_QUERIES_PER_RESEARCH]
    
    async def _search_and_extract(self, query: SearchQuery, question: str) -> Tuple[List[ScoredContent], int]:
        """Execute search and extract content with caching"""
        contents = []
        cache_hits = 0
        
        try:
            # Execute Serper search
            search_results = await self.serper.search(
                query=query.text,
                num_results=query.num_results,
                use_queue=True
            )
            
            organic_results = search_results.get('organic', [])
            logger.info(f"Found {len(organic_results)} search results for: {query.text}")
            
            # Process each result
            for result in organic_results:
                url = result.get('link', '')
                title = result.get('title', '')
                snippet = result.get('snippet', '')
                
                if not url:
                    continue
                
                try:
                    # Try to scrape with cache
                    content_data, from_cache = await self.serper.scrape_with_cache(
                        url=url,
                        extract_text=True,
                        extract_markdown=True,
                        use_queue=True,
                        cache_expiry_days=30
                    )
                    
                    if from_cache:
                        cache_hits += 1
                    
                    # Extract content text
                    content_text = content_data.get('text', '') or content_data.get('markdown', '') or snippet
                    
                    if content_text and len(content_text.strip()) > 50:  # Minimum content length
                        # Create scored content (evaluation done in next phase)
                        scored = ScoredContent(
                            url=url,
                            title=title,
                            content=self._optimize_content(content_text),
                            relevance_score=0.0,  # To be evaluated
                            confidence=0.8 if content_text != snippet else 0.5,
                            source_quality=self._assess_source_quality(url),
                            from_cache=from_cache
                        )
                        contents.append(scored)
                        
                except Exception as e:
                    logger.warning(f"Failed to extract content from {url}: {e}")
                    # Fallback to snippet
                    if snippet:
                        scored = ScoredContent(
                            url=url,
                            title=title,
                            content=snippet,
                            relevance_score=0.0,
                            confidence=0.3,
                            source_quality=self._assess_source_quality(url),
                            from_cache=False
                        )
                        contents.append(scored)
        
        except Exception as e:
            logger.error(f"Search and extract failed for query '{query.text}': {e}")
        
        return contents, cache_hits
    
    async def _filter_and_score_contents_old_approach(self, contents: List[ScoredContent], question: str,
                                       start_time: float) -> List[ScoredContent]:
        """OLD APPROACH - REPLACED BY BACKEND-STYLE PROCESSING
        
        This method caused the infinite loop by making individual DeepSeek API calls.
        Now we use backend-style processing that avoids individual evaluations.
        """
        # THIS METHOD IS NO LONGER USED - Kept for reference only
        logger.warning("🚨 OLD APPROACH - This method should not be called anymore")
        return contents
    
    async def _evaluate_relevance(self, question: str, content: str, url: str) -> float:
        """Evaluate content relevance using DeepSeek"""
        optimized_content = self._optimize_content(content)
        
        prompt = f"""Rate content relevance to the question on scale 0-10.

Question: {question}

Content from {url}:
{optimized_content}

Rate 0-10 where:
0=Irrelevant, 5=Somewhat relevant, 7=Relevant (threshold), 10=Perfect

Rating:"""
        
        try:
            response = await self.deepseek.async_chat_completion(
                query=prompt,
                system_message="You are a helpful content relevance evaluator",
                search_mode="deep"
            )
            
            # CRITICAL FIX: DeepSeek service returns 'message' not 'content'
            response_text = response.get('message', '') or response.get('content', '5')
            rating_match = re.search(r'\d+\.?\d*', response_text)
            if rating_match:
                rating = float(rating_match.group()) / 10.0
                logger.debug(f"Relevance rating for {url}: {rating:.1f}/10 -> {rating:.1%}")
                return min(max(rating, 0.0), 1.0)  # Ensure 0-1 range
            else:
                logger.warning(f"Could not extract rating from response: {response_text[:100]}")
                
        except Exception as e:
            logger.warning(f"Relevance evaluation error: {e}")
        
        return 0.5  # Default medium relevance
    
    async def _synthesize_answer(self, question: str, contents: List[ScoredContent]) -> ResearchAnswer:
        """Synthesize comprehensive answer from relevant sources"""
        if not contents:
            return ResearchAnswer(
                content="No relevant information found for this query.",
                confidence=0.0,
                sources=[],
                statistics={},
                generation_time=0.0
            )
        
        start_time = time.time()
        
        # Prepare source summaries
        source_texts = []
        for idx, content in enumerate(contents[:8], 1):  # Limit to top 8
            optimized = self._optimize_content(content.content, 400)  # Shorter per source
            source_texts.append(f"Source {idx} ({content.relevance_score:.0%}):\n{optimized}")
        
        sources_combined = "\n\n".join(source_texts)
        
        prompt = f"""Provide a comprehensive answer based on these sources.

Question: {question}

Sources:
{sources_combined}

Instructions:
1. Synthesize information from multiple sources
2. Structure with clear sections and bullet points
3. Include specific data and statistics when available
4. Reference sources with [1], [2], etc.
5. Note information gaps if any

Answer:"""
        
        try:
            response = await self.deepseek.async_chat_completion(
                query=prompt,
                system_message="You are a helpful research synthesizer",
                search_mode="deep"
            )
            
            answer_text = response.get('message', 'Unable to synthesize answer.')
            
            # Calculate confidence based on source quality
            confidence = min(0.95, sum(c.relevance_score for c in contents) / len(contents))
            
            # Extract statistics
            statistics = self._extract_statistics(contents)
            
            return ResearchAnswer(
                content=answer_text,
                confidence=confidence,
                sources=[c.url for c in contents[:8]],
                statistics=statistics,
                generation_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Answer synthesis failed: {e}")
            return ResearchAnswer(
                content=f"Error synthesizing answer: {str(e)}",
                confidence=0.0,
                sources=[c.url for c in contents[:5]],
                statistics={},
                generation_time=time.time() - start_time
            )
    
    def _extract_statistics(self, contents: List[ScoredContent]) -> Dict[str, Any]:
        """Extract statistical data from content using regex patterns"""
        stats = {
            "numbers_found": [],
            "percentages": [],
            "years": [],
            "currencies": []
        }
        
        for content in contents:
            text = content.content
            
            # Extract numbers with units
            numbers = re.findall(
                r'\b\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|thousand|K|M|B))?\b',
                text, re.IGNORECASE
            )
            stats["numbers_found"].extend(numbers[:3])  # Max 3 per source
            
            # Extract percentages
            percentages = re.findall(r'\b\d+(?:\.\d+)?%', text)
            stats["percentages"].extend(percentages[:2])
            
            # Extract years
            years = re.findall(r'\b20\d{2}\b', text)
            stats["years"].extend(years[:2])
            
            # Extract currency amounts
            currencies = re.findall(r'[\$€£¥]\s*\d+(?:,\d{3})*(?:\.\d+)?(?:\s*[KMBT])?', text)
            stats["currencies"].extend(currencies[:2])
        
        # Deduplicate and limit
        for key in stats:
            stats[key] = list(set(stats[key]))[:8]  # Max 8 of each type
        
        return stats
    
    def _optimize_content(self, content: str, max_length: int = MAX_CONTENT_LENGTH) -> str:
        """Optimize content length for token limits"""
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
    
    def _assess_source_quality(self, url: str) -> int:
        """Assess source quality based on domain patterns"""
        domain = url.lower()
        
        # High-quality domains
        if any(hq in domain for hq in [
            "wikipedia.org", "nature.com", "science.org", "ieee.org", 
            "harvard.edu", "mit.edu", "stanford.edu", "arxiv.org"
        ]):
            return 9
        
        # Educational/Government
        if domain.endswith(".edu") or domain.endswith(".gov"):
            return 8
        
        # Non-profit organizations
        if domain.endswith(".org"):
            return 7
        
        # Professional platforms
        if any(mp in domain for mp in ["github.com", "medium.com", "stackoverflow.com"]):
            return 6
        
        # Default
        return 5
    
    def _deduplicate_contents(self, contents: List[ScoredContent]) -> List[ScoredContent]:
        """Remove duplicate content based on URL and content similarity"""
        if len(contents) <= 1:
            return contents
        
        unique = []
        seen_urls = set()
        seen_content_hashes = set()
        
        for content in contents:
            # Check URL
            if content.url in seen_urls:
                continue
            
            # Simple content hash
            content_hash = hash(content.content[:100] if len(content.content) > 100 else content.content)
            if content_hash in seen_content_hashes:
                continue
            
            unique.append(content)
            seen_urls.add(content.url)
            seen_content_hashes.add(content_hash)
        
        logger.info(f"Deduplicated: {len(contents)} -> {len(unique)} unique sources")
        return unique
    
    async def _store_research_result(self, result: ResearchResult) -> None:
        """Store research result in MongoDB"""
        try:
            # Convert to dictionary for storage
            result_doc = {
                "session_id": result.session_id,
                "question": result.question,
                "answer_content": result.answer.content,
                "answer_confidence": result.answer.confidence,
                "sources": result.answer.sources,
                "statistics": result.answer.statistics,
                "queries_generated": result.queries_generated,
                "sources_analyzed": result.sources_analyzed,
                "cache_hits": result.cache_hits,
                "total_duration": result.total_duration,
                "metadata": result.metadata,
                "created_at": datetime.now(timezone.utc),
                "generation_time": result.answer.generation_time
            }
            
            # Store in research_sessions collection
            await self.mongodb.db.research_sessions.insert_one(result_doc)
            logger.info(f"Research result stored for session {result.session_id}")
            
        except Exception as e:
            logger.error(f"Failed to store research result: {e}")
    
    def _check_timeout(self, start_time: float) -> bool:
        """Check if research timeout reached"""
        return (time.time() - start_time) >= MAX_RESEARCH_TIME
    
    async def _search_and_extract_with_dedup(self, query: SearchQuery, question: str,
                                           processed_urls: set, max_content: int) -> Tuple[List[ScoredContent], int]:
        """Execute search and extract content with URL deduplication to prevent infinite loop"""
        contents = []
        cache_hits = 0
        
        try:
            # Execute Serper search
            search_results = await self.serper.search(
                query=query.text,
                num_results=min(query.num_results, max_content),  # Limit results
                use_queue=True
            )
            
            organic_results = search_results.get('organic', [])
            logger.info(f"Found {len(organic_results)} search results for: {query.text}")
            
            # Process each result with deduplication
            processed_count = 0
            for result in organic_results:
                url = result.get('link', '')
                title = result.get('title', '')
                snippet = result.get('snippet', '')
                
                if not url:
                    continue
                
                # CRITICAL FIX: Skip already processed URLs
                if url in processed_urls:
                    logger.debug(f"Skipping already processed URL: {url}")
                    continue
                
                # CRITICAL FIX: Limit processing to prevent infinite loop
                if processed_count >= max_content:
                    logger.info(f"Reached max content limit ({max_content}) for this query")
                    break
                
                try:
                    # Try to scrape with cache
                    content_data, from_cache = await self.serper.scrape_with_cache(
                        url=url,
                        extract_text=True,
                        extract_markdown=True,
                        use_queue=True,
                        cache_expiry_days=30
                    )
                    
                    if from_cache:
                        cache_hits += 1
                    
                    # Extract content text
                    content_text = content_data.get('text', '') or content_data.get('markdown', '') or snippet
                    
                    if content_text and len(content_text.strip()) > 50:  # Minimum content length
                        # Create scored content (evaluation done in next phase)
                        scored = ScoredContent(
                            url=url,
                            title=title,
                            content=self._optimize_content(content_text),
                            relevance_score=0.0,  # To be evaluated
                            confidence=0.8 if content_text != snippet else 0.5,
                            source_quality=self._assess_source_quality(url),
                            from_cache=from_cache
                        )
                        contents.append(scored)
                        processed_count += 1
                        logger.debug(f"Processed content from: {url}")
                        
                except Exception as e:
                    logger.warning(f"Failed to extract content from {url}: {e}")
                    # Fallback to snippet
                    if snippet and url not in processed_urls:
                        scored = ScoredContent(
                            url=url,
                            title=title,
                            content=snippet,
                            relevance_score=0.0,
                            confidence=0.3,
                            source_quality=self._assess_source_quality(url),
                            from_cache=False
                        )
                        contents.append(scored)
                        processed_count += 1
        
        except Exception as e:
            logger.error(f"Search and extract failed for query '{query.text}': {e}")
        
        return contents, cache_hits
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current session status"""
        return self.active_sessions.get(session_id)
    
    def cleanup_session(self, session_id: str) -> None:
        """Clean up completed session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Session {session_id} cleaned up")
    
    async def _update_session_progress(self, session_id: str, step: SessionStep, 
                                      progress: int, description: str, 
                                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update session progress in session manager"""
        if self.session_manager:
            try:
                await self.session_manager.update_session_progress(
                    session_id, step, progress, description, metadata
                )
            except Exception as e:
                logger.warning(f"Failed to update session progress {session_id}: {e}")
    
    async def _complete_session(self, session_id: str, result: SessionResult) -> None:
        """Complete session in session manager"""
        if self.session_manager:
            try:
                await self.session_manager.complete_session(session_id, result)
                self.successful_sessions += 1
            except Exception as e:
                logger.error(f"Failed to complete session {session_id}: {e}")
        else:
            # Update in-memory tracking
            if session_id in self.active_sessions:
                self.active_sessions[session_id]['status'] = 'completed'
                self.active_sessions[session_id]['result'] = result
            self.successful_sessions += 1
    
    async def _fail_session(self, session_id: str, error_message: str) -> None:
        """Fail session in session manager"""
        if self.session_manager:
            try:
                await self.session_manager.fail_session(session_id, error_message)
            except Exception as e:
                logger.error(f"Failed to mark session {session_id} as failed: {e}")
        else:
            # Update in-memory tracking
            if session_id in self.active_sessions:
                self.active_sessions[session_id]['status'] = 'failed'
                self.active_sessions[session_id]['error'] = error_message
    
    async def _process_search_results_like_backend(self, search_results: List[dict], question: str) -> List[ScoredContent]:
        """Process search results exactly like backend ResultProcessor.process_search_results()"""
        try:
            # Convert search results to ScoredContent (like backend lines 1114-1117)
            contents = []
            for result in search_results:
                try:
                    content = ScoredContent(
                        url=result.get('link', ''),
                        title=result.get('title', ''),
                        content=result.get('snippet', ''),  # Start with snippet
                        relevance_score=0.0,  # Will be evaluated individually
                        confidence=0.5,
                        source_quality=self._assess_source_quality(result.get('link', '')),
                        from_cache=False
                    )
                    contents.append(content)
                except Exception as e:
                    logger.warning(f"⚠️ Error creating content from result: {e}")
                    continue
            
            # Process each content individually (like backend lines 1125-1128)
            processed_contents = []
            for content in contents:
                try:
                    # Scrape full content if needed (like backend scraping)
                    if len(content.content) < 100:  # Only scrape if snippet is too short
                        scraped_content = await self._scrape_content_safely(content.url)
                        if scraped_content and len(scraped_content) > len(content.content):
                            content.content = self._optimize_content(scraped_content)
                    
                    # Evaluate relevance individually (like backend individual API calls)
                    relevance_score = await self._evaluate_relevance_like_backend(content, question)
                    content.relevance_score = relevance_score
                    
                    processed_contents.append(content)
                    logger.info(f"📄 Processed {content.url[:50]}: relevance {relevance_score:.1%}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error processing content {content.url}: {e}")
                    continue
            
            # Filter by relevance (like backend filter_by_relevance)
            relevant_contents = [c for c in processed_contents if c.relevance_score >= RELEVANCE_THRESHOLD]
            
            logger.info(f"🎯 Processed {len(processed_contents)} contents, {len(relevant_contents)} relevant")
            return relevant_contents
            
        except Exception as e:
            logger.error(f"❌ Error in _process_search_results_like_backend: {e}")
            return []
    
    async def _evaluate_relevance_like_backend(self, content: ScoredContent, question: str) -> float:
        """Evaluate content relevance EXACTLY like backend test file (lines 883-919)"""
        try:
            # REPLICATE EXACT BACKEND PROMPT (lines 888-901)
            optimized_content = self._optimize_content(content.content)
            
            prompt = f"""Rate the relevance of this content to the research question on a scale of 0-10.

Research Question: {question}

Content from {content.url}:
{optimized_content}

Provide only a number between 0 and 10, where:
0 = Completely irrelevant
5 = Somewhat relevant
7 = Relevant (meets threshold)
10 = Perfectly relevant

Rating:"""
            
            # REPLICATE EXACT BACKEND API CALL (lines 904-909)
            # Backend uses client.chat.completions.create() directly
            # But frontend must use service wrapper with correct parameters
            response = await self.deepseek.async_chat_completion(
                query=prompt,
                system_message="",  # Backend doesn't use system message
                search_mode=""      # Backend doesn't use search_mode
            )
            
            # REPLICATE EXACT BACKEND PARSING (lines 912-913)
            response_text = response.get('message', '') if isinstance(response, dict) else str(response)
            
            # Backend uses: float(re.search(r'\d+\.?\d*', rating_text).group()) / 10.0
            import re
            match = re.search(r'\d+\.?\d*', response_text)
            if match:
                rating = float(match.group()) / 10.0
                return min(max(rating, 0.0), 1.0)  # Ensure 0-1 range (line 915)
            else:
                logger.warning(f"⚠️ No rating found in response: {response_text}")
                return 0.5  # Backend default (line 919)
            
        except Exception as e:
            logger.warning(f"⚠️ Relevance evaluation failed for {content.url}: {e}")
            return 0.5  # Backend default fallback
    
    async def _scrape_content_safely(self, url: str) -> Optional[str]:
        """Safely scrape content with error handling"""
        try:
            content_data, _ = await self.serper.scrape_with_cache(
                url=url,
                extract_text=True,
                extract_markdown=True,
                use_queue=True,
                cache_expiry_days=30
            )
            return content_data.get('text', '') or content_data.get('markdown', '')
        except Exception as e:
            logger.warning(f"⚠️ Failed to scrape {url}: {e}")
            return None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get orchestrator performance metrics"""
        success_rate = (self.successful_sessions / max(self.total_sessions, 1)) * 100
        
        return {
            "total_sessions": self.total_sessions,
            "successful_sessions": self.successful_sessions,
            "active_sessions": len(self.active_sessions),
            "success_rate": round(success_rate, 1),
            "cache_hit_rate": round(self.cache_hit_rate * 100, 1) if self.cache_hit_rate else 0.0,
            "avg_research_time": MAX_RESEARCH_TIME / 2,  # Placeholder - could track actual averages
            "session_manager_available": self.session_manager is not None
        }
