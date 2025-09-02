#!/usr/bin/env python3
"""
Jan-style Deep Reasoning Engine for Content Analysis
Applies Jan framework's deep-thinking logic to evaluate content relevance and generate reasoning chains
"""

import os
import logging
import json
import re
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Import the existing data models
try:
    from .deepthink_models import RelevanceScore, ReasoningChain, ScrapedContent
    from .deepseek_service import DeepSeekService
except ImportError:
    # Fallback for testing
    RelevanceScore = None
    ReasoningChain = None
    ScrapedContent = None
    DeepSeekService = None

logger = logging.getLogger(__name__)


class ReasoningType(Enum):
    """Types of reasoning chains"""
    DEDUCTIVE = "deductive"       # General to specific
    INDUCTIVE = "inductive"       # Specific to general
    ABDUCTIVE = "abductive"       # Best explanation
    CAUSAL = "causal"             # Cause and effect
    COMPARATIVE = "comparative"   # Comparison-based
    ANALOGICAL = "analogical"     # Analogy-based


class ContradictionSeverity(Enum):
    """Severity levels for contradictions"""
    MINOR = "minor"               # Small inconsistencies
    MODERATE = "moderate"         # Significant disagreements
    MAJOR = "major"               # Fundamental contradictions


@dataclass
class ContentAnalysis:
    """Analysis results for scraped content"""
    content_id: str
    relevance_score: float        # 0-10 scale
    confidence: float             # 0-1 scale
    key_points: List[str]         # Main points extracted
    evidence_strength: str        # weak/moderate/strong
    source_credibility: str       # low/medium/high
    reasoning_type: ReasoningType
    supporting_facts: List[str]
    contradictory_facts: List[str]
    uncertainty_areas: List[str]
    evaluation_time: float


@dataclass
class Contradiction:
    """Detected contradiction between sources"""
    contradiction_id: str
    content_ids: List[str]        # IDs of conflicting content
    topic: str                    # What the contradiction is about
    conflicting_claims: List[str]  # The actual contradictory statements
    severity: ContradictionSeverity
    explanation: str              # Why this is considered a contradiction
    resolution_suggestions: List[str]  # How to resolve the contradiction
    confidence: float             # Confidence in contradiction detection


class JanReasoningEngine:
    """
    Jan-style deep reasoning engine for content analysis
    Evaluates relevance, generates reasoning chains, and detects contradictions
    """
    
    def __init__(self, deepseek_service: Optional[Any] = None):
        """
        Initialize the Jan reasoning engine
        
        Args:
            deepseek_service: DeepSeek service instance (optional)
        """
        self.deepseek_service = deepseek_service
        if deepseek_service is None and DeepSeekService is not None:
            try:
                self.deepseek_service = DeepSeekService()
            except Exception as e:
                logger.warning(f"Could not initialize DeepSeek service: {e}")
        
        # Relevance evaluation configuration
        self.relevance_threshold = 7.0
        self.min_confidence = 0.6
        self.max_content_length = 10000  # Characters for analysis
        
        # Reasoning chain configuration
        self.max_chain_length = 10
        self.min_evidence_items = 2
        
        # Contradiction detection settings
        self.similarity_threshold = 0.8
        self.contradiction_keywords = [
            'however', 'but', 'although', 'despite', 'contrary', 'opposite',
            'disagree', 'conflict', 'contradict', 'versus', 'different from'
        ]
        
        # Source credibility indicators
        self.credibility_indicators = {
            'high': ['peer-reviewed', 'academic', 'research', 'study', 'published', 'journal'],
            'medium': ['news', 'report', 'analysis', 'expert', 'professional'],
            'low': ['blog', 'opinion', 'rumor', 'unverified', 'anonymous']
        }
        
        # Statistics tracking
        self.evaluations_count = 0
        self.reasoning_chains_generated = 0
        self.contradictions_detected = 0
        self.high_relevance_count = 0
        self.processing_time_total = 0.0
    
    async def evaluate_relevance(self, content: ScrapedContent, user_query: str,
                               context_topics: Optional[List[str]] = None) -> ContentAnalysis:
        """
        Evaluate content relevance using Jan-style deep analysis
        
        Args:
            content: ScrapedContent object to analyze
            user_query: Original user query for relevance assessment
            context_topics: Additional context topics for evaluation
            
        Returns:
            ContentAnalysis with relevance score and detailed analysis
            
        Raises:
            ValueError: If inputs are invalid
            Exception: For processing errors
        """
        if not content or not content.text_content:
            raise ValueError("Content cannot be empty")
        
        if not user_query or not user_query.strip():
            raise ValueError("User query cannot be empty")
        
        start_time = datetime.utcnow()
        self.evaluations_count += 1
        
        try:
            logger.info(f"Evaluating relevance for content from {content.url}")
            
            # Prepare content for analysis (truncate if too long)
            analysis_content = content.text_content[:self.max_content_length]
            
            # Create relevance evaluation prompt
            evaluation_prompt = self._create_relevance_prompt(
                analysis_content, user_query, content.url, context_topics
            )
            
            # Get evaluation from DeepSeek API
            if self.deepseek_service:
                evaluation_response = await self._call_deepseek_api(evaluation_prompt)
                parsed_evaluation = self._parse_evaluation_response(evaluation_response)
            else:
                # Fallback to rule-based evaluation
                logger.warning("DeepSeek service not available, using fallback evaluation")
                parsed_evaluation = self._fallback_evaluation(analysis_content, user_query, content)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.processing_time_total += processing_time
            
            # Create ContentAnalysis object
            analysis = ContentAnalysis(
                content_id=self._generate_content_id(content),
                relevance_score=parsed_evaluation.get('relevance_score', 5.0),
                confidence=parsed_evaluation.get('confidence', 0.7),
                key_points=parsed_evaluation.get('key_points', []),
                evidence_strength=parsed_evaluation.get('evidence_strength', 'moderate'),
                source_credibility=parsed_evaluation.get('source_credibility', 'medium'),
                reasoning_type=ReasoningType(parsed_evaluation.get('reasoning_type', 'deductive')),
                supporting_facts=parsed_evaluation.get('supporting_facts', []),
                contradictory_facts=parsed_evaluation.get('contradictory_facts', []),
                uncertainty_areas=parsed_evaluation.get('uncertainty_areas', []),
                evaluation_time=processing_time
            )
            
            # Track high relevance content
            if analysis.relevance_score >= self.relevance_threshold:
                self.high_relevance_count += 1
            
            logger.info(f"Relevance evaluation completed: score={analysis.relevance_score:.1f}, "
                       f"confidence={analysis.confidence:.2f}, time={processing_time:.2f}s")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Relevance evaluation failed for {content.url}: {e}")
            raise
    
    async def generate_reasoning_chains(self, content_analyses: List[ContentAnalysis],
                                      user_query: str, max_chains: int = 5) -> List[ReasoningChain]:
        """
        Generate logical reasoning chains from analyzed content
        
        Args:
            content_analyses: List of ContentAnalysis objects
            user_query: Original user query
            max_chains: Maximum number of reasoning chains to generate
            
        Returns:
            List of ReasoningChain objects ordered by confidence
            
        Raises:
            ValueError: If inputs are invalid
            Exception: For processing errors
        """
        if not content_analyses:
            raise ValueError("Content analyses cannot be empty")
        
        if not user_query or not user_query.strip():
            raise ValueError("User query cannot be empty")
        
        self.reasoning_chains_generated += 1
        
        try:
            logger.info(f"Generating reasoning chains from {len(content_analyses)} content analyses")
            
            # Filter high-relevance content
            high_relevance_content = [
                analysis for analysis in content_analyses 
                if analysis.relevance_score >= self.relevance_threshold
            ]
            
            if not high_relevance_content:
                logger.warning("No high-relevance content found for reasoning chain generation")
                return []
            
            # Group content by reasoning type
            reasoning_groups = self._group_by_reasoning_type(high_relevance_content)
            
            # Generate reasoning chains for each group
            chains = []
            
            for reasoning_type, analyses in reasoning_groups.items():
                if len(chains) >= max_chains:
                    break
                
                chain = await self._generate_reasoning_chain(
                    analyses, user_query, reasoning_type
                )
                if chain:
                    chains.append(chain)
            
            # Sort by confidence
            chains.sort(key=lambda c: c.confidence, reverse=True)
            
            logger.info(f"Generated {len(chains)} reasoning chains")
            return chains[:max_chains]
            
        except Exception as e:
            logger.error(f"Reasoning chain generation failed: {e}")
            raise
    
    async def identify_contradictions(self, content_analyses: List[ContentAnalysis],
                                    user_query: str) -> List[Contradiction]:
        """
        Identify contradictions and conflicts between content sources
        
        Args:
            content_analyses: List of ContentAnalysis objects to compare
            user_query: Original user query for context
            
        Returns:
            List of Contradiction objects
            
        Raises:
            ValueError: If inputs are invalid
            Exception: For processing errors
        """
        if not content_analyses or len(content_analyses) < 2:
            return []  # Need at least 2 sources to find contradictions
        
        self.contradictions_detected += 1
        
        try:
            logger.info(f"Identifying contradictions among {len(content_analyses)} content analyses")
            
            contradictions = []
            
            # Compare each pair of content analyses
            for i in range(len(content_analyses)):
                for j in range(i + 1, len(content_analyses)):
                    content1 = content_analyses[i]
                    content2 = content_analyses[j]
                    
                    # Check for contradictions
                    contradiction = await self._detect_contradiction(content1, content2, user_query)
                    if contradiction:
                        contradictions.append(contradiction)
            
            # Remove duplicate contradictions and sort by severity
            unique_contradictions = self._deduplicate_contradictions(contradictions)
            unique_contradictions.sort(key=lambda c: c.severity.value, reverse=True)
            
            logger.info(f"Identified {len(unique_contradictions)} unique contradictions")
            return unique_contradictions
            
        except Exception as e:
            logger.error(f"Contradiction identification failed: {e}")
            raise
    
    def _generate_content_id(self, content: ScrapedContent) -> str:
        """Generate unique ID for content"""
        content_hash = hashlib.sha256(content.text_content.encode()).hexdigest()
        return f"content_{content_hash[:12]}"
    
    def _create_relevance_prompt(self, content: str, user_query: str, 
                               url: str, context_topics: Optional[List[str]] = None) -> str:
        """Create prompt for relevance evaluation"""
        context_str = ""
        if context_topics:
            context_str = f"Context topics: {', '.join(context_topics)}\n"
        
        return f"""
Evaluate the relevance of this content to the user's query using Jan-style deep reasoning:

User Query: "{user_query}"
{context_str}Source URL: {url}

Content to analyze:
{content}

Please provide a JSON response with detailed analysis:

{{
    "relevance_score": 8.5,  // 0-10 scale (7.0+ is high relevance)
    "confidence": 0.85,      // 0-1 scale for confidence in assessment
    "key_points": ["main point 1", "main point 2"],
    "evidence_strength": "strong|moderate|weak",
    "source_credibility": "high|medium|low",
    "reasoning_type": "deductive|inductive|abductive|causal|comparative|analogical",
    "supporting_facts": ["fact supporting relevance"],
    "contradictory_facts": ["fact that contradicts or complicates"],
    "uncertainty_areas": ["areas where info is uncertain"]
}}

Use Jan-style deep analysis: examine premises, evaluate evidence strength, consider alternative interpretations, and assess logical consistency.
"""
    
    async def _call_deepseek_api(self, prompt: str) -> str:
        """Call DeepSeek API for reasoning evaluation"""
        if not self.deepseek_service:
            raise Exception("DeepSeek service not available")
        
        logger.debug("Calling DeepSeek API for Jan-style reasoning evaluation")
        
        try:
            response = await self.deepseek_service.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,  # Low temperature for consistent reasoning
                max_tokens=1500
            )
            return response.get('content', '')
        except Exception as e:
            logger.error(f"DeepSeek API call failed: {e}")
            raise
    
    def _parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from DeepSeek API"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                return self._extract_evaluation_from_text(response)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return self._extract_evaluation_from_text(response)
    
    def _extract_evaluation_from_text(self, text: str) -> Dict[str, Any]:
        """Fallback text parsing for evaluation"""
        # Simple extraction from text
        relevance_score = 5.0
        if 'highly relevant' in text.lower():
            relevance_score = 8.5
        elif 'very relevant' in text.lower():
            relevance_score = 7.5
        elif 'relevant' in text.lower():
            relevance_score = 6.5
        elif 'not relevant' in text.lower():
            relevance_score = 3.0
        
        return {
            'relevance_score': relevance_score,
            'confidence': 0.6,
            'key_points': [],
            'evidence_strength': 'moderate',
            'source_credibility': 'medium',
            'reasoning_type': 'deductive',
            'supporting_facts': [],
            'contradictory_facts': [],
            'uncertainty_areas': []
        }
    
    def _fallback_evaluation(self, content: str, user_query: str, 
                           scraped_content: ScrapedContent) -> Dict[str, Any]:
        """Rule-based fallback evaluation when DeepSeek API is not available"""
        content_lower = content.lower()
        query_lower = user_query.lower()
        
        # Calculate basic relevance score
        query_words = query_lower.split()
        matches = sum(1 for word in query_words if word in content_lower)
        relevance_score = min(10.0, (matches / len(query_words)) * 10)
        
        # Assess source credibility
        url_lower = scraped_content.url.lower()
        source_credibility = 'medium'
        
        for level, indicators in self.credibility_indicators.items():
            if any(indicator in url_lower or indicator in content_lower[:500] 
                  for indicator in indicators):
                source_credibility = level
                break
        
        # Adjust relevance based on credibility
        if source_credibility == 'high':
            relevance_score += 1.0
        elif source_credibility == 'low':
            relevance_score -= 1.0
        
        relevance_score = max(0.0, min(10.0, relevance_score))
        
        return {
            'relevance_score': relevance_score,
            'confidence': 0.7,
            'key_points': [],
            'evidence_strength': 'moderate',
            'source_credibility': source_credibility,
            'reasoning_type': 'deductive',
            'supporting_facts': [],
            'contradictory_facts': [],
            'uncertainty_areas': []
        }
    
    def _group_by_reasoning_type(self, analyses: List[ContentAnalysis]) -> Dict[ReasoningType, List[ContentAnalysis]]:
        """Group content analyses by reasoning type"""
        groups = {}
        for analysis in analyses:
            reasoning_type = analysis.reasoning_type
            if reasoning_type not in groups:
                groups[reasoning_type] = []
            groups[reasoning_type].append(analysis)
        return groups
    
    async def _generate_reasoning_chain(self, analyses: List[ContentAnalysis], 
                                       user_query: str, reasoning_type: ReasoningType) -> Optional[ReasoningChain]:
        """Generate a single reasoning chain from grouped analyses"""
        if len(analyses) < self.min_evidence_items:
            return None
        
        try:
            # Extract key information from analyses
            all_facts = []
            source_urls = []
            
            for analysis in analyses:
                all_facts.extend(analysis.supporting_facts)
                source_urls.append(f"content_{analysis.content_id}")  # Placeholder URL
            
            if not all_facts:
                return None
            
            # Create reasoning chain
            chain_id = f"chain_{hashlib.sha256(user_query.encode()).hexdigest()[:8]}_{reasoning_type.value}"
            
            # Simple reasoning chain construction
            premise = f"Based on {len(analyses)} sources regarding: {user_query}"
            evidence = all_facts[:5]  # Take top 5 facts
            conclusion = f"The evidence suggests that {user_query.lower()} involves multiple factors"
            
            # Calculate confidence based on analysis confidence
            avg_confidence = sum(a.confidence for a in analyses) / len(analyses)
            
            return ReasoningChain(
                chain_id=chain_id,
                premise=premise,
                evidence=evidence,
                conclusion=conclusion,
                confidence=avg_confidence,
                source_urls=source_urls,
                contradictions=[]
            )
            
        except Exception as e:
            logger.error(f"Failed to generate reasoning chain: {e}")
            return None
    
    async def _detect_contradiction(self, content1: ContentAnalysis, content2: ContentAnalysis,
                                  user_query: str) -> Optional[Contradiction]:
        """Detect contradiction between two content analyses"""
        try:
            # Check for contradictory facts
            contradictions = []
            
            for fact1 in content1.supporting_facts:
                for fact2 in content2.supporting_facts:
                    if self._are_contradictory(fact1, fact2):
                        contradictions.append((fact1, fact2))
            
            # Check contradictory_facts fields
            for fact in content1.contradictory_facts:
                if any(keyword in fact.lower() for keyword in self.contradiction_keywords):
                    contradictions.append((fact, "implicit contradiction detected"))
            
            if not contradictions:
                return None
            
            # Determine severity
            severity = ContradictionSeverity.MINOR
            if len(contradictions) > 2:
                severity = ContradictionSeverity.MODERATE
            if abs(content1.relevance_score - content2.relevance_score) > 5.0:
                severity = ContradictionSeverity.MAJOR
            
            contradiction_id = f"contra_{content1.content_id[:8]}_{content2.content_id[:8]}"
            
            return Contradiction(
                contradiction_id=contradiction_id,
                content_ids=[content1.content_id, content2.content_id],
                topic=user_query,
                conflicting_claims=[c[0] for c in contradictions],
                severity=severity,
                explanation=f"Detected {len(contradictions)} conflicting claims",
                resolution_suggestions=["Compare source credibility", "Look for additional sources"],
                confidence=0.7
            )
            
        except Exception as e:
            logger.error(f"Failed to detect contradiction: {e}")
            return None
    
    def _are_contradictory(self, fact1: str, fact2: str) -> bool:
        """Check if two facts are contradictory"""
        fact1_lower = fact1.lower()
        fact2_lower = fact2.lower()
        
        # Simple contradiction detection
        contradiction_pairs = [
            ('increase', 'decrease'),
            ('improve', 'worsen'),
            ('positive', 'negative'),
            ('yes', 'no'),
            ('true', 'false'),
            ('effective', 'ineffective')
        ]
        
        for pos, neg in contradiction_pairs:
            if pos in fact1_lower and neg in fact2_lower:
                return True
            if neg in fact1_lower and pos in fact2_lower:
                return True
        
        return False
    
    def _deduplicate_contradictions(self, contradictions: List[Contradiction]) -> List[Contradiction]:
        """Remove duplicate contradictions"""
        seen = set()
        unique_contradictions = []
        
        for contradiction in contradictions:
            # Create a key based on content IDs
            key = tuple(sorted(contradiction.content_ids))
            if key not in seen:
                seen.add(key)
                unique_contradictions.append(contradiction)
        
        return unique_contradictions
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine usage statistics"""
        avg_processing_time = (
            self.processing_time_total / max(self.evaluations_count, 1)
        )
        
        high_relevance_rate = (
            self.high_relevance_count / max(self.evaluations_count, 1)
        )
        
        return {
            'evaluations_count': self.evaluations_count,
            'reasoning_chains_generated': self.reasoning_chains_generated,
            'contradictions_detected': self.contradictions_detected,
            'high_relevance_count': self.high_relevance_count,
            'high_relevance_rate': round(high_relevance_rate, 3),
            'avg_processing_time_seconds': round(avg_processing_time, 3),
            'total_processing_time_seconds': round(self.processing_time_total, 2),
            'relevance_threshold': self.relevance_threshold,
            'deepseek_available': self.deepseek_service is not None
        }
    
    def reset_stats(self):
        """Reset usage statistics"""
        self.evaluations_count = 0
        self.reasoning_chains_generated = 0
        self.contradictions_detected = 0
        self.high_relevance_count = 0
        self.processing_time_total = 0.0