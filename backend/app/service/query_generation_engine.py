#!/usr/bin/env python3
"""
Query Generation Engine using DeepSeek API
Generates diverse, optimized search queries from user input for comprehensive web research
"""

import os
import logging
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# Import the existing DeepSeek service
try:
    from .deepseek_service import DeepSeekService
except ImportError:
    # Fallback for testing
    DeepSeekService = None

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of search queries"""
    FACTUAL = "factual"           # Direct facts and information
    COMPARATIVE = "comparative"    # Comparisons and contrasts  
    TEMPORAL = "temporal"         # Time-based or historical
    STATISTICAL = "statistical"   # Numbers, data, statistics
    ANALYTICAL = "analytical"     # Analysis and explanations
    CURRENT = "current"          # Recent news and updates


class QuestionComplexity(Enum):
    """Complexity levels for user questions"""
    SIMPLE = "simple"             # Single concept, direct answer
    MODERATE = "moderate"         # Multiple concepts, some analysis
    COMPLEX = "complex"          # Multi-faceted, requires deep analysis


@dataclass
class QuestionAnalysis:
    """Analysis results for a user question"""
    entities: List[str]           # Key entities/concepts
    intent: str                   # What the user wants to know
    complexity: QuestionComplexity
    topics: List[str]             # Main topics/subjects
    question_type: str            # Type of question (what, how, why, etc.)
    requires_current_data: bool   # Whether recent information is needed
    suggested_operators: Dict[str, str]  # Search operators to use
    confidence: float             # Confidence in analysis (0-1)


@dataclass  
class GeneratedQuery:
    """A generated search query with metadata"""
    text: str                     # The actual query text
    query_type: QueryType         # Type of query
    priority: int                 # Priority (1=highest, 5=lowest)
    reasoning: str                # Why this query was generated
    operators: Dict[str, str]     # Advanced search operators
    expected_results: str         # What kind of results expected


class QueryGenerationEngine:
    """
    Engine for generating diverse, optimized search queries from user input
    Uses DeepSeek API for intelligent query analysis and generation
    """
    
    def __init__(self, deepseek_service: Optional[Any] = None):
        """
        Initialize the query generation engine
        
        Args:
            deepseek_service: DeepSeek service instance (optional, will create if None)
        """
        self.deepseek_service = deepseek_service
        if deepseek_service is None and DeepSeekService is not None:
            try:
                self.deepseek_service = DeepSeekService()
            except Exception as e:
                logger.warning(f"Could not initialize DeepSeek service: {e}")
        
        # Query generation configuration
        self.max_queries = 5
        self.min_queries = 3
        
        # Advanced search operators
        self.search_operators = {
            'site': 'site:',        # Search specific websites
            'filetype': 'filetype:',# Search specific file types  
            'intitle': 'intitle:',  # Search in page titles
            'intext': 'intext:',    # Search in page content
            'daterange': '',        # Date range (custom implementation)
            'related': 'related:',  # Related websites
            'cache': 'cache:',      # Cached versions
        }
        
        # Trusted sources for different types of queries
        self.trusted_sources = {
            'academic': ['scholar.google.com', 'arxiv.org', 'pubmed.ncbi.nlm.nih.gov'],
            'news': ['reuters.com', 'bbc.com', 'apnews.com', 'npr.org'],
            'government': ['gov', 'edu'],
            'statistics': ['census.gov', 'worldbank.org', 'statista.com'],
            'technical': ['github.com', 'stackoverflow.com', 'documentation'],
        }
        
        # Statistics tracking
        self.analysis_count = 0
        self.generation_count = 0
        self.success_count = 0
        self.error_count = 0
        
    async def analyze_question(self, user_question: str) -> QuestionAnalysis:
        """
        Analyze user question to extract entities, intent, and complexity
        
        Args:
            user_question: The user's input question
            
        Returns:
            QuestionAnalysis object with analysis results
            
        Raises:
            ValueError: If question is empty or invalid
            Exception: For API or processing errors
        """
        if not user_question or not user_question.strip():
            raise ValueError("Question cannot be empty")
        
        self.analysis_count += 1
        question = user_question.strip()
        
        try:
            logger.info(f"Analyzing question: '{question[:100]}...'")
            
            # Create analysis prompt
            analysis_prompt = self._create_analysis_prompt(question)
            
            # Get analysis from DeepSeek API
            if self.deepseek_service:
                analysis_response = await self._call_deepseek_api(analysis_prompt)
                parsed_analysis = self._parse_analysis_response(analysis_response)
            else:
                # Fallback to rule-based analysis
                logger.warning("DeepSeek service not available, using fallback analysis")
                parsed_analysis = self._fallback_analysis(question)
            
            # Create QuestionAnalysis object
            analysis = QuestionAnalysis(
                entities=parsed_analysis.get('entities', []),
                intent=parsed_analysis.get('intent', 'Find information'),
                complexity=QuestionComplexity(parsed_analysis.get('complexity', 'moderate')),
                topics=parsed_analysis.get('topics', []),
                question_type=parsed_analysis.get('question_type', 'general'),
                requires_current_data=parsed_analysis.get('requires_current_data', False),
                suggested_operators=parsed_analysis.get('suggested_operators', {}),
                confidence=parsed_analysis.get('confidence', 0.7)
            )
            
            logger.info(f"Question analysis completed: {len(analysis.entities)} entities, "
                       f"{analysis.complexity.value} complexity, confidence: {analysis.confidence}")
            
            self.success_count += 1
            return analysis
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Question analysis failed: {e}")
            raise
    
    async def generate_search_queries(self, user_question: str, 
                                    question_analysis: Optional[QuestionAnalysis] = None,
                                    num_queries: Optional[int] = None) -> List[GeneratedQuery]:
        """
        Generate diverse search queries covering different angles
        
        Args:
            user_question: The original user question
            question_analysis: Pre-computed analysis (optional, will analyze if None)
            num_queries: Number of queries to generate (optional, uses default range)
            
        Returns:
            List of GeneratedQuery objects ordered by priority
            
        Raises:
            ValueError: If inputs are invalid
            Exception: For API or processing errors
        """
        if not user_question or not user_question.strip():
            raise ValueError("Question cannot be empty")
        
        self.generation_count += 1
        
        try:
            # Get question analysis if not provided
            if question_analysis is None:
                question_analysis = await self.analyze_question(user_question)
            
            # Determine number of queries to generate
            if num_queries is None:
                if question_analysis.complexity == QuestionComplexity.SIMPLE:
                    num_queries = self.min_queries
                elif question_analysis.complexity == QuestionComplexity.COMPLEX:
                    num_queries = self.max_queries
                else:
                    num_queries = 4
            
            num_queries = max(self.min_queries, min(self.max_queries, num_queries))
            
            logger.info(f"Generating {num_queries} queries for {question_analysis.complexity.value} question")
            
            # Generate diverse query types
            queries = []
            
            # Always include a direct factual query
            queries.append(self._generate_factual_query(user_question, question_analysis))
            
            # Add other query types based on question analysis
            remaining_slots = num_queries - 1
            
            if question_analysis.requires_current_data and remaining_slots > 0:
                queries.append(self._generate_current_query(user_question, question_analysis))
                remaining_slots -= 1
            
            if remaining_slots > 0:
                queries.append(self._generate_analytical_query(user_question, question_analysis))
                remaining_slots -= 1
            
            if remaining_slots > 0 and len(question_analysis.entities) > 1:
                queries.append(self._generate_comparative_query(user_question, question_analysis))
                remaining_slots -= 1
            
            if remaining_slots > 0:
                queries.append(self._generate_statistical_query(user_question, question_analysis))
                remaining_slots -= 1
            
            # Fill remaining slots with additional queries
            while remaining_slots > 0 and len(queries) < num_queries:
                queries.append(self._generate_temporal_query(user_question, question_analysis))
                remaining_slots -= 1
            
            # Sort by priority and add priority numbers
            queries.sort(key=lambda q: q.priority)
            for i, query in enumerate(queries):
                query.priority = i + 1
            
            logger.info(f"Generated {len(queries)} queries: {[q.query_type.value for q in queries]}")
            self.success_count += 1
            
            return queries
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Query generation failed: {e}")
            raise
    
    def _create_analysis_prompt(self, question: str) -> str:
        """Create prompt for question analysis"""
        return f"""
Analyze this user question for web search optimization:

Question: "{question}"

Please provide a JSON response with the following analysis:

{{
    "entities": ["list", "of", "key", "entities", "and", "concepts"],
    "intent": "what the user wants to know",
    "complexity": "simple|moderate|complex",
    "topics": ["main", "subject", "areas"],
    "question_type": "what|how|why|when|where|who|general",
    "requires_current_data": true/false,
    "suggested_operators": {{
        "site": "specific websites if relevant",
        "filetype": "pdf|doc etc if relevant",
        "intitle": "key terms for titles"
    }},
    "confidence": 0.8
}}

Focus on extracting actionable information for creating effective search queries.
"""
    
    async def _call_deepseek_api(self, prompt: str) -> str:
        """Call DeepSeek API with the analysis prompt"""
        if not self.deepseek_service:
            raise Exception("DeepSeek service not available")
        
        # This would call the actual DeepSeek service
        # For now, we'll create a placeholder implementation
        logger.debug("Calling DeepSeek API for question analysis")
        
        try:
            # Use the correct DeepSeek service interface
            response = await self.deepseek_service.async_chat_completion(
                query=prompt,
                system_message="You are an expert at analyzing search queries. Respond with valid JSON only.",
                max_retries=3,
                search_mode="search"
            )
            return response.get('content', '')
        except Exception as e:
            logger.error(f"DeepSeek API call failed: {e}")
            raise
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from DeepSeek API"""
        try:
            # Extract JSON from response (handle potential markdown formatting)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                # Fallback parsing
                return self._extract_analysis_from_text(response)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return self._extract_analysis_from_text(response)
    
    def _extract_analysis_from_text(self, text: str) -> Dict[str, Any]:
        """Fallback text parsing for analysis"""
        # Simple text-based extraction
        return {
            'entities': re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)[:5],
            'intent': 'Find information',
            'complexity': 'moderate',
            'topics': [],
            'question_type': 'general',
            'requires_current_data': 'recent' in text.lower() or 'current' in text.lower(),
            'suggested_operators': {},
            'confidence': 0.7
        }
    
    def _fallback_analysis(self, question: str) -> Dict[str, Any]:
        """Rule-based fallback analysis when DeepSeek API is not available"""
        question_lower = question.lower()
        
        # Detect question type
        question_type = 'general'
        if question_lower.startswith(('what', 'which')):
            question_type = 'what'
        elif question_lower.startswith('how'):
            question_type = 'how'
        elif question_lower.startswith('why'):
            question_type = 'why'
        elif question_lower.startswith(('when', 'where')):
            question_type = question_lower.split()[0]
        elif question_lower.startswith('who'):
            question_type = 'who'
        
        # Detect complexity
        word_count = len(question.split())
        complexity = 'simple'
        
        # Check for complexity indicators
        complex_indicators = [' and ', ' or ', 'compare', 'contrast', 'impact', 'effect', 'relationship', 'implications']
        has_complex_indicators = any(indicator in question_lower for indicator in complex_indicators)
        
        if word_count > 15 or has_complex_indicators:
            complexity = 'complex'
        elif word_count >= 7:  # 7 or more words for moderate
            complexity = 'moderate'
        
        # Extract entities (simple approach)
        words = question.split()
        entities = [word.strip('.,!?') for word in words if word.istitle() and len(word) > 2][:5]
        
        # Detect if current data is needed
        requires_current = any(term in question_lower for term in 
                             ['recent', 'current', 'latest', 'today', 'now', '2024', '2025'])
        
        return {
            'entities': entities,
            'intent': f'Find information about {" ".join(entities[:3]) if entities else "the topic"}',
            'complexity': complexity,
            'topics': entities[:3],
            'question_type': question_type,
            'requires_current_data': requires_current,
            'suggested_operators': {},
            'confidence': 0.7
        }
    
    def _generate_factual_query(self, question: str, analysis: QuestionAnalysis) -> GeneratedQuery:
        """Generate a direct factual query"""
        # Use main entities and clean up the question
        entities = ' '.join(analysis.entities[:3])
        
        query_text = entities if entities else question
        
        operators = {}
        if analysis.suggested_operators.get('site'):
            operators['site'] = analysis.suggested_operators['site']
        
        return GeneratedQuery(
            text=query_text,
            query_type=QueryType.FACTUAL,
            priority=1,  # Highest priority
            reasoning="Direct factual query using main entities from the question",
            operators=operators,
            expected_results="Direct facts and basic information"
        )
    
    def _generate_analytical_query(self, question: str, analysis: QuestionAnalysis) -> GeneratedQuery:
        """Generate an analytical/explanatory query"""
        if analysis.question_type in ['how', 'why']:
            query_text = f"how does {' '.join(analysis.entities[:2])} work"
            if analysis.question_type == 'why':
                query_text = f"why is {' '.join(analysis.entities[:2])} important"
        else:
            query_text = f"explanation of {' '.join(analysis.entities[:3])}"
        
        return GeneratedQuery(
            text=query_text,
            query_type=QueryType.ANALYTICAL,
            priority=2,
            reasoning="Analytical query to understand mechanisms and explanations",
            operators={},
            expected_results="Explanations, analyses, and detailed information"
        )
    
    def _generate_current_query(self, question: str, analysis: QuestionAnalysis) -> GeneratedQuery:
        """Generate a current/recent information query"""
        entities = ' '.join(analysis.entities[:2])
        query_text = f"{entities} recent news 2024 2025"
        
        operators = {
            'site': 'news OR site:reuters.com OR site:bbc.com'
        }
        
        return GeneratedQuery(
            text=query_text,
            query_type=QueryType.CURRENT,
            priority=1,  # High priority for current info
            reasoning="Current information query for recent developments",
            operators=operators,
            expected_results="Recent news, updates, and current developments"
        )
    
    def _generate_comparative_query(self, question: str, analysis: QuestionAnalysis) -> GeneratedQuery:
        """Generate a comparative query"""
        entities = analysis.entities[:2]
        if len(entities) >= 2:
            query_text = f"{entities[0]} vs {entities[1]} comparison"
        else:
            query_text = f"{entities[0]} compared to alternatives" if entities else "comparison alternatives"
        
        return GeneratedQuery(
            text=query_text,
            query_type=QueryType.COMPARATIVE,
            priority=3,
            reasoning="Comparative query to understand differences and alternatives",
            operators={},
            expected_results="Comparisons, contrasts, and alternative options"
        )
    
    def _generate_statistical_query(self, question: str, analysis: QuestionAnalysis) -> GeneratedQuery:
        """Generate a statistical/data query"""
        entities = ' '.join(analysis.entities[:2])
        query_text = f"{entities} statistics data numbers trends"
        
        operators = {
            'site': 'site:statista.com OR site:census.gov OR site:worldbank.org'
        }
        
        return GeneratedQuery(
            text=query_text,
            query_type=QueryType.STATISTICAL,
            priority=4,
            reasoning="Statistical query for quantitative data and trends",
            operators=operators,
            expected_results="Statistics, data, numbers, and quantitative information"
        )
    
    def _generate_temporal_query(self, question: str, analysis: QuestionAnalysis) -> GeneratedQuery:
        """Generate a temporal/historical query"""
        entities = ' '.join(analysis.entities[:2])
        query_text = f"{entities} history timeline development"
        
        return GeneratedQuery(
            text=query_text,
            query_type=QueryType.TEMPORAL,
            priority=5,
            reasoning="Temporal query for historical context and development",
            operators={},
            expected_results="Historical information, timelines, and development history"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine usage statistics"""
        return {
            'analysis_count': self.analysis_count,
            'generation_count': self.generation_count,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'success_rate': round(self.success_count / max(self.analysis_count + self.generation_count, 1), 3),
            'deepseek_available': self.deepseek_service is not None
        }
    
    def reset_stats(self):
        """Reset usage statistics"""
        self.analysis_count = 0
        self.generation_count = 0
        self.success_count = 0
        self.error_count = 0
