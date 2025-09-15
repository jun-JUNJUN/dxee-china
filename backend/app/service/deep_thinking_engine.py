#!/usr/bin/env python3
"""
Deep Think Query Generation Engine
Extracted from test_deepseek_advanced_web_research4_01.py

This module provides multi-perspective query generation using deep-thinking patterns
inspired by the 'jan' project for comprehensive web research.
"""

import json
import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

# Import LLM client type
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None
    logging.warning("OpenAI library not available, DeepThinkingEngine will require AsyncOpenAI client")

logger = logging.getLogger(__name__)

# Configuration constants
MAX_QUERIES_PER_RESEARCH = 10
MAX_RESULTS_PER_QUERY = 10
DEEPSEEK_MODEL = "deepseek-chat"
SERPER_DEFAULT_REGION = "us"
SERPER_DEFAULT_LANGUAGE = "en"


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
        params = {
            "q": self.text,
            "gl": self.region,
            "hl": self.language,
            "num": self.num_results,
            "page": self.page
        }

        # Add time filter if specified
        if self.time_filter:
            params["tbs"] = self.time_filter

        return params


@dataclass
class QuestionAnalysis:
    """Structured analysis of a research question"""
    main_topic: str
    subtopics: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    intent: str = "general"
    scope: str = "current"
    domain: str = "general"


class QueryPatterns:
    """Query patterns for multi-perspective search generation"""

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

    def __init__(self, llm_client):
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