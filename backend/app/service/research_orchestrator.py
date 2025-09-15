#!/usr/bin/env python3
"""
Research Orchestrator - Main research workflow implementation
Implements the exact algorithm from test_deepseek_advanced_web_research4_01.py
with JSON logging and streaming progress updates.

This orchestrator coordinates all components to deliver comprehensive web research
with exact compatibility to the test algorithm and standardized logging output.
"""

import os
import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, AsyncGenerator, Optional
from dataclasses import dataclass

# Import our extracted components
from app.service.token_manager import TokenManager, check_time_limit
from app.service.deep_thinking_engine import DeepThinkingEngine, SearchQuery
from app.service.serper_client import SerperClient, SearchResult, ScrapeResult
from app.service.result_processor import ResultProcessor, ScoredContent
from app.service.statistical_extractor import StatisticalExtractor
from app.service.json_log_writer import JSONLogWriter

# Import existing services
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

logger = logging.getLogger(__name__)

# Configuration constants matching the test algorithm
MAX_RESEARCH_TIME = int(os.getenv("MAX_RESEARCH_TIME", "600"))  # 10 minutes
MAX_QUERIES_PER_RESEARCH = 10
RELEVANCE_THRESHOLD = 0.7  # 70% relevance threshold
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_TEMPERATURE = 0.7
DEEPSEEK_MAX_TOKENS = 4000


@dataclass
class ResearchProgress:
    """Progress update for streaming"""
    step: int
    total_steps: int
    description: str
    progress: float  # 0.0 to 100.0
    metadata: Dict[str, Any]


class ResearchOrchestrator:
    """Main research orchestrator implementing the test algorithm"""

    def __init__(self,
                 deepseek_api_key: Optional[str] = None,
                 serper_api_key: Optional[str] = None,
                 log_directory: str = "logs",
                 max_research_time: int = MAX_RESEARCH_TIME):
        """
        Initialize research orchestrator

        Args:
            deepseek_api_key: DeepSeek API key
            serper_api_key: Serper API key
            log_directory: Directory for JSON logs
            max_research_time: Maximum research time in seconds
        """
        # Get API keys from environment if not provided
        self.deepseek_api_key = deepseek_api_key or os.getenv("DEEPSEEK_API_KEY")
        self.serper_api_key = serper_api_key or os.getenv("SERPER_API_KEY")
        self.max_research_time = max_research_time

        if not self.deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY is required")
        if not self.serper_api_key:
            raise ValueError("SERPER_API_KEY is required")

        # Initialize components
        self.token_manager = TokenManager()
        self.statistical_extractor = StatisticalExtractor()
        self.json_logger = JSONLogWriter(log_directory)

        # Initialize AI client
        self.deepseek_client = AsyncOpenAI(
            api_key=self.deepseek_api_key,
            base_url=os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com")
        )

        # Initialize other components (will be created in context managers)
        self.thinking_engine = None
        self.serper_client = None
        self.result_processor = None

        # Metrics
        self.start_time = None
        self.queries_generated = 0
        self.sources_analyzed = 0
        self.cache_hits = 0

    async def research_with_logging(self, question: str) -> Dict[str, Any]:
        """
        Conduct comprehensive research with JSON logging
        Implements the exact algorithm from test_deepseek_advanced_web_research4_01.py

        Args:
            question: Research question

        Returns:
            Research result dictionary matching reference format
        """
        self.start_time = time.time()
        logger.info(f"🚀 Starting research with logging: {question}")

        # Initialize components
        self.thinking_engine = DeepThinkingEngine(self.deepseek_client)

        async with SerperClient(self.serper_api_key) as serper_client:
            self.serper_client = serper_client
            self.result_processor = ResultProcessor(self, serper_client)

            try:
                # Phase 1: Generate queries using deep-thinking patterns
                queries = await self.thinking_engine.generate_queries(question)
                self.queries_generated = len(queries)
                logger.info(f"📋 Generated {len(queries)} research queries")

                # Phase 2: Execute searches and process results
                all_contents = []

                for i, query in enumerate(queries):
                    # Check timeout
                    if self.check_timeout():
                        logger.warning("⏰ Research timeout reached")
                        break

                    # Execute search
                    logger.info(f"🔍 Searching ({i+1}/{len(queries)}): {query.text}")
                    results = await serper_client.search(query)

                    # Process results
                    contents = await self.result_processor.process_search_results(results, question)
                    all_contents.extend(contents)

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
                    answer_text = await self.synthesize_answer(question, all_contents)

                    # Extract statistics
                    statistics = self.statistical_extractor.extract_statistics_from_multiple_sources(
                        [c.content for c in all_contents]
                    )
                else:
                    answer_text = "No relevant information found for this query."
                    statistics = {
                        "numbers_found": [],
                        "percentages": [],
                        "dates": [],
                        "metrics": {}
                    }

                # Calculate confidence based on content quality
                confidence = self.calculate_confidence(all_contents)

                # Create result in exact format matching reference
                result = {
                    "question": question,
                    "answer": answer_text,
                    "confidence": confidence,
                    "sources": [c.url for c in all_contents[:10]],  # Top 10 sources
                    "statistics": statistics,
                    "metadata": {
                        "relevance_threshold": RELEVANCE_THRESHOLD,
                        "timeout_reached": self.check_timeout(),
                        "serper_requests": serper_client.request_count
                    },
                    "duration": time.time() - self.start_time
                }

                # Write JSON log
                log_filename = self.json_logger.write_research_log(result)
                logger.info(f"💾 Research log written: {log_filename}")

                # Log summary
                self._log_summary(result)

                return result

            except Exception as e:
                logger.error(f"❌ Research failed: {e}")
                # Return error result
                error_result = {
                    "question": question,
                    "answer": f"Research failed due to error: {str(e)}",
                    "confidence": 0.0,
                    "sources": [],
                    "statistics": {
                        "numbers_found": [],
                        "percentages": [],
                        "dates": [],
                        "metrics": {}
                    },
                    "metadata": {
                        "relevance_threshold": RELEVANCE_THRESHOLD,
                        "timeout_reached": self.check_timeout(),
                        "serper_requests": getattr(self.serper_client, 'request_count', 0)
                    },
                    "duration": time.time() - self.start_time if self.start_time else 0.0
                }

                # Still log the error result
                self.json_logger.write_research_log(error_result)
                return error_result

    async def research_with_streaming(self, question: str) -> AsyncGenerator[ResearchProgress, None]:
        """
        Conduct research with streaming progress updates

        Args:
            question: Research question

        Yields:
            ResearchProgress updates
        """
        total_steps = 7  # Query generation, search, processing, synthesis, statistics, logging, completion
        current_step = 0

        try:
            # Step 1: Initialize
            current_step += 1
            yield ResearchProgress(
                step=current_step,
                total_steps=total_steps,
                description="Initializing research session",
                progress=(current_step / total_steps) * 100,
                metadata={"phase": "initialization"}
            )

            # Step 2: Generate queries
            current_step += 1
            yield ResearchProgress(
                step=current_step,
                total_steps=total_steps,
                description="Generating research queries",
                progress=(current_step / total_steps) * 100,
                metadata={"phase": "query_generation"}
            )

            # Step 3: Execute searches
            current_step += 1
            yield ResearchProgress(
                step=current_step,
                total_steps=total_steps,
                description="Executing web searches",
                progress=(current_step / total_steps) * 100,
                metadata={"phase": "web_search"}
            )

            # Step 4: Process content
            current_step += 1
            yield ResearchProgress(
                step=current_step,
                total_steps=total_steps,
                description="Processing and evaluating content",
                progress=(current_step / total_steps) * 100,
                metadata={"phase": "content_processing"}
            )

            # Step 5: Synthesize answer
            current_step += 1
            yield ResearchProgress(
                step=current_step,
                total_steps=total_steps,
                description="Synthesizing comprehensive answer",
                progress=(current_step / total_steps) * 100,
                metadata={"phase": "answer_synthesis"}
            )

            # Step 6: Generate statistics
            current_step += 1
            yield ResearchProgress(
                step=current_step,
                total_steps=total_steps,
                description="Extracting statistical data",
                progress=(current_step / total_steps) * 100,
                metadata={"phase": "statistics"}
            )

            # Execute the actual research
            result = await self.research_with_logging(question)

            # Step 7: Complete
            current_step += 1
            yield ResearchProgress(
                step=current_step,
                total_steps=total_steps,
                description="Research completed",
                progress=100.0,
                metadata={
                    "phase": "completion",
                    "result": result
                }
            )

        except Exception as e:
            yield ResearchProgress(
                step=current_step,
                total_steps=total_steps,
                description=f"Research failed: {str(e)}",
                progress=0.0,
                metadata={"phase": "error", "error": str(e)}
            )

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
            response = await self.deepseek_client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=DEEPSEEK_TEMPERATURE,
                max_tokens=DEEPSEEK_MAX_TOKENS
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"❌ Answer synthesis failed: {e}")
            return "Unable to synthesize answer due to an error."

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
            response = await self.deepseek_client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10
            )

            # Extract rating
            rating_text = response.choices[0].message.content.strip()
            import re
            rating = float(re.search(r'\d+\.?\d*', rating_text).group()) / 10.0

            return min(max(rating, 0.0), 1.0)  # Ensure 0-1 range

        except Exception as e:
            logger.warning(f"⚠️ Relevance evaluation failed: {e}")
            return 0.5  # Default middle score

    def calculate_confidence(self, contents: List[ScoredContent]) -> float:
        """Calculate confidence based on source quality and relevance"""
        if not contents:
            return 0.0

        # Weight by relevance score and source quality
        total_weight = 0.0
        weighted_confidence = 0.0

        for content in contents:
            weight = content.relevance_score * (content.source_quality / 10.0)
            total_weight += weight
            weighted_confidence += weight * content.confidence

        if total_weight > 0:
            base_confidence = weighted_confidence / total_weight

            # Boost confidence based on number of sources
            source_boost = min(len(contents) / 10.0, 0.3)  # Max 30% boost

            return min(base_confidence + source_boost, 0.95)  # Max 95% confidence

        return 0.0

    def check_timeout(self) -> bool:
        """Check if research timeout reached"""
        if self.start_time:
            return check_time_limit(self.start_time, self.max_research_time)
        return False

    def stream_progress(self, description: str, progress: float, metadata: Dict[str, Any] = None):
        """Stream progress update (for compatibility)"""
        logger.info(f"Progress: {progress:.1f}% - {description}")

    def _log_summary(self, result: Dict[str, Any]):
        """Log research summary"""
        logger.info("=" * 50)
        logger.info("📊 RESEARCH SUMMARY")
        logger.info(f"❓ Question: {result['question']}")
        logger.info(f"🔍 Queries generated: {self.queries_generated}")
        logger.info(f"📄 Sources analyzed: {self.sources_analyzed}")
        logger.info(f"💾 Cache hits: {self.cache_hits}")
        logger.info(f"🎯 Confidence: {result['confidence']:.1%}")
        logger.info(f"⏱️ Duration: {result['duration']:.1f}s")
        logger.info(f"📈 Statistics found: {len(result['statistics'].get('numbers_found', []))}")
        logger.info("=" * 50)