#!/usr/bin/env python3
"""
Answer Synthesis Engine for Dual-Format Responses
Creates comprehensive answers and concise summaries from analyzed content with citations
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

# Import the existing data models and services
try:
    from .deepthink_models import DeepThinkResult, ReasoningChain
    from .jan_reasoning_engine import ContentAnalysis, Contradiction
    from .deepseek_service import DeepSeekService
    from .deepthink_models import count_tokens
except ImportError:
    # Fallback for testing
    DeepThinkResult = None
    ReasoningChain = None
    ContentAnalysis = None
    Contradiction = None
    DeepSeekService = None
    count_tokens = None

logger = logging.getLogger(__name__)


class ResponseFormat(Enum):
    """Response format types"""
    COMPREHENSIVE = "comprehensive"
    SUMMARY = "summary"
    MIXED = "mixed"


class ConfidenceLevel(Enum):
    """Confidence levels for answer synthesis"""
    HIGH = "high"        # 0.8-1.0
    MEDIUM = "medium"    # 0.6-0.8  
    LOW = "low"          # 0.0-0.6


@dataclass
class SourceCitation:
    """Citation for a source with confidence and relevance info"""
    url: str
    title: str
    relevance_score: float
    confidence: float
    credibility: str           # high/medium/low
    key_points: List[str]
    citation_text: str         # Formatted citation
    access_date: datetime


@dataclass
class SynthesizedAnswer:
    """Complete synthesized answer with both formats"""
    comprehensive_answer: str   # Detailed answer with full context
    summary: str               # Concise summary
    confidence: float          # Overall confidence (0-1)
    confidence_level: ConfidenceLevel
    sources_analyzed: int
    high_relevance_sources: int
    citations: List[SourceCitation]
    key_findings: List[str]
    uncertainties: List[str]   # Areas of uncertainty
    contradictions_noted: List[str]
    reasoning_chains_used: int
    word_count_comprehensive: int
    word_count_summary: int
    synthesis_time: float


class AnswerSynthesizer:
    """
    Engine for synthesizing comprehensive answers and summaries from analyzed content
    Creates dual-format responses with proper citations and confidence indicators
    """
    
    def __init__(self, deepseek_service: Optional[Any] = None):
        """
        Initialize the answer synthesizer
        
        Args:
            deepseek_service: DeepSeek service instance (optional)
        """
        self.deepseek_service = deepseek_service
        if deepseek_service is None and DeepSeekService is not None:
            try:
                self.deepseek_service = DeepSeekService()
            except Exception as e:
                logger.warning(f"Could not initialize DeepSeek service: {e}")
        
        # Synthesis configuration
        self.min_sources_for_high_confidence = 3
        self.min_relevance_threshold = 7.0
        self.max_comprehensive_length = 2000  # words
        self.max_summary_length = 300         # words
        self.max_tokens_per_request = 3000
        
        # Citation formatting
        self.citation_style = "numbered"  # numbered, apa, mla
        self.include_access_dates = True
        
        # Content organization
        self.organize_by_topics = True
        self.include_confidence_indicators = True
        self.highlight_contradictions = True
        
        # Statistics tracking
        self.syntheses_count = 0
        self.comprehensive_answers_generated = 0
        self.summaries_generated = 0
        self.high_confidence_answers = 0
        self.processing_time_total = 0.0
    
    async def generate_comprehensive_answer(self, content_analyses: List[ContentAnalysis],
                                          user_query: str, reasoning_chains: Optional[List[ReasoningChain]] = None,
                                          contradictions: Optional[List[Contradiction]] = None) -> str:
        """
        Generate comprehensive answer with full context and analysis
        
        Args:
            content_analyses: List of analyzed content
            user_query: Original user question
            reasoning_chains: Optional reasoning chains for logical flow
            contradictions: Optional detected contradictions
            
        Returns:
            Comprehensive answer with citations and full context
            
        Raises:
            ValueError: If inputs are invalid
            Exception: For synthesis errors
        """
        if not content_analyses:
            raise ValueError("Content analyses cannot be empty")
        
        if not user_query or not user_query.strip():
            raise ValueError("User query cannot be empty")
        
        self.comprehensive_answers_generated += 1
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Generating comprehensive answer from {len(content_analyses)} content analyses")
            
            # Filter high-relevance content
            high_relevance_content = [
                analysis for analysis in content_analyses
                if analysis.relevance_score >= self.min_relevance_threshold
            ]
            
            if not high_relevance_content:
                logger.warning("No high-relevance content available for comprehensive answer")
                return self._generate_fallback_answer(user_query, content_analyses)
            
            # Create synthesis prompt
            synthesis_prompt = self._create_comprehensive_prompt(
                user_query, high_relevance_content, reasoning_chains, contradictions
            )
            
            # Generate answer using DeepSeek API
            if self.deepseek_service:
                comprehensive_answer = await self._call_deepseek_for_synthesis(
                    synthesis_prompt, target_length=self.max_comprehensive_length
                )
            else:
                # Fallback to template-based synthesis
                logger.warning("DeepSeek service not available, using fallback synthesis")
                comprehensive_answer = self._fallback_comprehensive_synthesis(
                    user_query, high_relevance_content, reasoning_chains, contradictions
                )
            
            # Add citations and formatting
            formatted_answer = self._add_citations_to_answer(comprehensive_answer, high_relevance_content)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.processing_time_total += processing_time
            
            word_count = len(formatted_answer.split())
            logger.info(f"Comprehensive answer generated: {word_count} words, {processing_time:.2f}s")
            
            return formatted_answer
            
        except Exception as e:
            logger.error(f"Comprehensive answer generation failed: {e}")
            raise
    
    async def generate_summary(self, comprehensive_answer: str, user_query: str,
                             key_findings: Optional[List[str]] = None) -> str:
        """
        Generate concise summary from comprehensive answer
        
        Args:
            comprehensive_answer: Full comprehensive answer
            user_query: Original user question
            key_findings: Optional key findings to emphasize
            
        Returns:
            Concise summary highlighting key points
            
        Raises:
            ValueError: If inputs are invalid
            Exception: For synthesis errors
        """
        if not comprehensive_answer or not comprehensive_answer.strip():
            raise ValueError("Comprehensive answer cannot be empty")
        
        if not user_query or not user_query.strip():
            raise ValueError("User query cannot be empty")
        
        self.summaries_generated += 1
        
        try:
            logger.info(f"Generating summary from {len(comprehensive_answer.split())} word answer")
            
            # Create summarization prompt
            summary_prompt = self._create_summary_prompt(
                comprehensive_answer, user_query, key_findings
            )
            
            # Generate summary using DeepSeek API
            if self.deepseek_service:
                summary = await self._call_deepseek_for_synthesis(
                    summary_prompt, target_length=self.max_summary_length
                )
            else:
                # Fallback to extractive summarization
                logger.warning("DeepSeek service not available, using extractive summarization")
                summary = self._extractive_summarization(comprehensive_answer, key_findings)
            
            # Ensure summary is within length limits
            summary = self._trim_to_length(summary, self.max_summary_length)
            
            word_count = len(summary.split())
            logger.info(f"Summary generated: {word_count} words")
            
            return summary
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            raise
    
    async def synthesize_complete_response(self, content_analyses: List[ContentAnalysis],
                                         user_query: str, reasoning_chains: Optional[List[ReasoningChain]] = None,
                                         contradictions: Optional[List[Contradiction]] = None) -> SynthesizedAnswer:
        """
        Synthesize complete response with both comprehensive answer and summary
        
        Args:
            content_analyses: List of analyzed content
            user_query: Original user question
            reasoning_chains: Optional reasoning chains
            contradictions: Optional detected contradictions
            
        Returns:
            SynthesizedAnswer with both formats and metadata
            
        Raises:
            ValueError: If inputs are invalid
            Exception: For synthesis errors
        """
        if not content_analyses:
            raise ValueError("Content analyses cannot be empty")
        
        self.syntheses_count += 1
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Synthesizing complete response for query: '{user_query[:50]}...'")
            
            # Generate comprehensive answer
            comprehensive_answer = await self.generate_comprehensive_answer(
                content_analyses, user_query, reasoning_chains, contradictions
            )
            
            # Extract key findings from content analyses
            key_findings = self._extract_key_findings(content_analyses)
            
            # Generate summary
            summary = await self.generate_summary(comprehensive_answer, user_query, key_findings)
            
            # Create source citations
            citations = self._create_source_citations(content_analyses)
            
            # Calculate confidence metrics
            confidence, confidence_level = self._calculate_confidence(content_analyses, reasoning_chains)
            
            # Extract uncertainty and contradiction information
            uncertainties = self._extract_uncertainties(content_analyses, contradictions)
            contradictions_noted = self._extract_contradiction_notes(contradictions) if contradictions else []
            
            # Calculate metrics
            high_relevance_count = len([a for a in content_analyses if a.relevance_score >= self.min_relevance_threshold])
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.processing_time_total += processing_time
            
            # Track high confidence answers
            if confidence_level == ConfidenceLevel.HIGH:
                self.high_confidence_answers += 1
            
            synthesized_answer = SynthesizedAnswer(
                comprehensive_answer=comprehensive_answer,
                summary=summary,
                confidence=confidence,
                confidence_level=confidence_level,
                sources_analyzed=len(content_analyses),
                high_relevance_sources=high_relevance_count,
                citations=citations,
                key_findings=key_findings,
                uncertainties=uncertainties,
                contradictions_noted=contradictions_noted,
                reasoning_chains_used=len(reasoning_chains) if reasoning_chains else 0,
                word_count_comprehensive=len(comprehensive_answer.split()),
                word_count_summary=len(summary.split()),
                synthesis_time=processing_time
            )
            
            logger.info(f"Complete synthesis finished: {synthesized_answer.word_count_comprehensive} comp words, "
                       f"{synthesized_answer.word_count_summary} summary words, "
                       f"confidence: {confidence:.2f}, time: {processing_time:.2f}s")
            
            return synthesized_answer
            
        except Exception as e:
            logger.error(f"Complete response synthesis failed: {e}")
            raise
    
    def format_for_chat(self, synthesized_answer: SynthesizedAnswer, 
                       include_metadata: bool = True) -> str:
        """
        Format synthesized answer for chat display with markdown
        
        Args:
            synthesized_answer: Complete synthesized answer
            include_metadata: Whether to include confidence and source metadata
            
        Returns:
            Markdown-formatted response ready for chat display
        """
        try:
            # Build markdown response
            markdown_parts = []
            
            # Add confidence indicator if enabled
            if self.include_confidence_indicators and include_metadata:
                confidence_emoji = self._get_confidence_emoji(synthesized_answer.confidence_level)
                markdown_parts.append(f"**Deep Think Research Result** {confidence_emoji}")
                markdown_parts.append("")
            
            # Add summary section (default collapsed state)
            markdown_parts.append("## 📋 Summary")
            markdown_parts.append(synthesized_answer.summary)
            markdown_parts.append("")
            
            # Add expandable comprehensive answer
            markdown_parts.append("<details>")
            markdown_parts.append("<summary><strong>📚 Comprehensive Analysis (Click to expand)</strong></summary>")
            markdown_parts.append("")
            markdown_parts.append(synthesized_answer.comprehensive_answer)
            markdown_parts.append("</details>")
            markdown_parts.append("")
            
            # Add key findings if available
            if synthesized_answer.key_findings:
                markdown_parts.append("## 🔍 Key Findings")
                for finding in synthesized_answer.key_findings:
                    markdown_parts.append(f"• {finding}")
                markdown_parts.append("")
            
            # Add uncertainties if any
            if synthesized_answer.uncertainties:
                markdown_parts.append("## ⚠️ Areas of Uncertainty")
                for uncertainty in synthesized_answer.uncertainties:
                    markdown_parts.append(f"• {uncertainty}")
                markdown_parts.append("")
            
            # Add contradiction warnings if any
            if synthesized_answer.contradictions_noted:
                markdown_parts.append("## ⚡ Conflicting Information Detected")
                for contradiction in synthesized_answer.contradictions_noted:
                    markdown_parts.append(f"• {contradiction}")
                markdown_parts.append("")
            
            # Add source citations
            if synthesized_answer.citations:
                markdown_parts.append("## 📖 Sources")
                for i, citation in enumerate(synthesized_answer.citations, 1):
                    relevance_indicator = self._get_relevance_indicator(citation.relevance_score)
                    credibility_indicator = self._get_credibility_indicator(citation.credibility)
                    
                    citation_text = f"{i}. [{citation.title}]({citation.url}) {relevance_indicator}{credibility_indicator}"
                    if citation.key_points:
                        citation_text += f" - {', '.join(citation.key_points[:2])}"
                    
                    markdown_parts.append(citation_text)
                markdown_parts.append("")
            
            # Add metadata if requested
            if include_metadata:
                markdown_parts.append("<details>")
                markdown_parts.append("<summary><em>Research Metadata</em></summary>")
                markdown_parts.append("")
                markdown_parts.append(f"- **Confidence**: {synthesized_answer.confidence:.1%} ({synthesized_answer.confidence_level.value})")
                markdown_parts.append(f"- **Sources Analyzed**: {synthesized_answer.sources_analyzed}")
                markdown_parts.append(f"- **High Relevance Sources**: {synthesized_answer.high_relevance_sources}")
                markdown_parts.append(f"- **Reasoning Chains**: {synthesized_answer.reasoning_chains_used}")
                markdown_parts.append(f"- **Processing Time**: {synthesized_answer.synthesis_time:.1f}s")
                markdown_parts.append("</details>")
            
            return "\n".join(markdown_parts)
            
        except Exception as e:
            logger.error(f"Chat formatting failed: {e}")
            return f"Error formatting response: {str(e)}"
    
    def _create_comprehensive_prompt(self, user_query: str, content_analyses: List[ContentAnalysis],
                                   reasoning_chains: Optional[List[ReasoningChain]] = None,
                                   contradictions: Optional[List[Contradiction]] = None) -> str:
        """Create prompt for comprehensive answer generation"""
        # Extract relevant content
        content_summaries = []
        for i, analysis in enumerate(content_analyses[:10], 1):  # Limit to top 10
            summary = f"Source {i} (Relevance: {analysis.relevance_score:.1f}/10, Credibility: {analysis.source_credibility}):\n"
            summary += f"Key points: {', '.join(analysis.key_points[:3])}\n"
            if analysis.supporting_facts:
                summary += f"Facts: {'; '.join(analysis.supporting_facts[:2])}\n"
            content_summaries.append(summary)
        
        # Add reasoning chains if available
        reasoning_info = ""
        if reasoning_chains:
            reasoning_info = f"\nReasoning chains identified ({len(reasoning_chains)}):\n"
            for chain in reasoning_chains[:3]:
                reasoning_info += f"- {chain.premise} → {chain.conclusion} (confidence: {chain.confidence:.2f})\n"
        
        # Add contradiction info if available
        contradiction_info = ""
        if contradictions:
            contradiction_info = f"\nContradictions detected ({len(contradictions)}):\n"
            for contra in contradictions[:2]:
                contradiction_info += f"- {contra.topic}: {contra.explanation}\n"
        
        return f"""
Generate a comprehensive, well-structured answer to this question using the provided research data:

Question: "{user_query}"

Research Data:
{chr(10).join(content_summaries)}
{reasoning_info}
{contradiction_info}

Instructions:
1. Provide a thorough, analytical answer that directly addresses the question
2. Organize information logically with clear topic sections  
3. Include specific facts and evidence from the sources
4. Address any contradictions or uncertainties honestly
5. Use authoritative tone while acknowledging limitations
6. Aim for {self.max_comprehensive_length} words maximum
7. Include logical reasoning that connects evidence to conclusions

Format as clear, structured text suitable for markdown conversion.
"""
    
    def _create_summary_prompt(self, comprehensive_answer: str, user_query: str,
                              key_findings: Optional[List[str]] = None) -> str:
        """Create prompt for summary generation"""
        findings_text = ""
        if key_findings:
            findings_text = f"\nKey findings to emphasize: {', '.join(key_findings[:5])}"
        
        return f"""
Create a concise, informative summary of this comprehensive answer:

Original Question: "{user_query}"
{findings_text}

Comprehensive Answer:
{comprehensive_answer[:2000]}  # Truncate if too long

Instructions:
1. Capture the main answer to the question in 2-3 sentences
2. Include the most important findings and evidence
3. Maintain accuracy while being concise
4. Use clear, direct language
5. Maximum {self.max_summary_length} words
6. Focus on actionable insights and key takeaways

Provide only the summary text, no additional formatting.
"""
    
    async def _call_deepseek_for_synthesis(self, prompt: str, target_length: int) -> str:
        """Call DeepSeek API for answer synthesis"""
        if not self.deepseek_service:
            raise Exception("DeepSeek service not available")
        
        # Calculate appropriate max_tokens
        max_tokens = min(self.max_tokens_per_request, target_length * 2)  # Rough token estimate
        
        logger.debug(f"Calling DeepSeek API for synthesis (max_tokens: {max_tokens})")
        
        try:
            response = await self.deepseek_service.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Moderate creativity for synthesis
                max_tokens=max_tokens
            )
            return response.get('content', '').strip()
        except Exception as e:
            logger.error(f"DeepSeek API call failed: {e}")
            raise
    
    def _fallback_comprehensive_synthesis(self, user_query: str, content_analyses: List[ContentAnalysis],
                                        reasoning_chains: Optional[List[ReasoningChain]] = None,
                                        contradictions: Optional[List[Contradiction]] = None) -> str:
        """Fallback template-based comprehensive synthesis"""
        # Extract key information
        key_points = []
        supporting_facts = []
        
        for analysis in content_analyses:
            key_points.extend(analysis.key_points[:2])
            supporting_facts.extend(analysis.supporting_facts[:2])
        
        # Build answer using template
        answer_parts = [
            f"Based on analysis of {len(content_analyses)} sources, here's a comprehensive answer to: {user_query}",
            "",
            "## Key Findings:",
        ]
        
        # Add key points
        for point in key_points[:5]:
            answer_parts.append(f"• {point}")
        
        answer_parts.extend(["", "## Evidence:"])
        
        # Add supporting facts
        for fact in supporting_facts[:5]:
            answer_parts.append(f"• {fact}")
        
        # Add reasoning if available
        if reasoning_chains:
            answer_parts.extend(["", "## Analysis:"])
            for chain in reasoning_chains[:2]:
                answer_parts.append(f"• {chain.conclusion}")
        
        # Add contradiction warnings if applicable
        if contradictions:
            answer_parts.extend(["", "## Important Note:"])
            answer_parts.append("Some sources present conflicting information. Please review individual sources for complete context.")
        
        return "\n".join(answer_parts)
    
    def _extractive_summarization(self, comprehensive_answer: str, 
                                 key_findings: Optional[List[str]] = None) -> str:
        """Fallback extractive summarization"""
        # Split into sentences
        sentences = re.split(r'[.!?]+', comprehensive_answer)
        sentences = [s.strip() for s in sentences if s.strip() and len(s) > 20]
        
        if not sentences:
            return "Summary not available."
        
        # Simple scoring based on position and key findings
        scored_sentences = []
        
        for i, sentence in enumerate(sentences):
            score = 1.0
            
            # Higher score for sentences at the beginning
            if i < len(sentences) * 0.3:
                score += 0.5
            
            # Higher score if contains key findings
            if key_findings:
                for finding in key_findings:
                    if finding.lower() in sentence.lower():
                        score += 1.0
                        break
            
            # Higher score for sentences with numbers or specifics
            if re.search(r'\d+', sentence):
                score += 0.3
            
            scored_sentences.append((sentence, score))
        
        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Build summary from top sentences, respecting order
        selected_sentences = [s[0] for s in scored_sentences[:3]]
        
        # Sort selected sentences by original order
        ordered_sentences = []
        for sentence in sentences:
            if sentence in selected_sentences:
                ordered_sentences.append(sentence)
        
        summary = '. '.join(ordered_sentences) + '.'
        return self._trim_to_length(summary, self.max_summary_length)
    
    def _generate_fallback_answer(self, user_query: str, content_analyses: List[ContentAnalysis]) -> str:
        """Generate fallback answer when no high-relevance content is available"""
        return f"""
Based on the available sources, I found some information related to "{user_query}", though the relevance scores were below the optimal threshold.

The analysis included {len(content_analyses)} sources, but none reached the high-relevance threshold of {self.min_relevance_threshold}/10. This may indicate that:

• The question requires more specific or recent sources
• The available content doesn't directly address the question
• The topic may need a different search approach

For the most accurate answer, I recommend refining the search with more specific terms or looking for additional authoritative sources on this topic.
"""
    
    def _add_citations_to_answer(self, answer: str, content_analyses: List[ContentAnalysis]) -> str:
        """Add numbered citations to the answer"""
        # For now, just append citation numbers
        # In a more sophisticated implementation, this would insert citations inline
        if not content_analyses:
            return answer
        
        citation_note = f"\n\n*Based on analysis of {len(content_analyses)} sources with relevance scores ranging from {min(a.relevance_score for a in content_analyses):.1f} to {max(a.relevance_score for a in content_analyses):.1f}/10.*"
        
        return answer + citation_note
    
    def _extract_key_findings(self, content_analyses: List[ContentAnalysis]) -> List[str]:
        """Extract key findings from content analyses"""
        findings = []
        
        for analysis in content_analyses:
            if analysis.relevance_score >= self.min_relevance_threshold:
                findings.extend(analysis.key_points[:2])
                findings.extend(analysis.supporting_facts[:1])
        
        # Remove duplicates and return top findings
        unique_findings = list(dict.fromkeys(findings))  # Preserves order
        return unique_findings[:8]
    
    def _create_source_citations(self, content_analyses: List[ContentAnalysis]) -> List[SourceCitation]:
        """Create formatted source citations"""
        citations = []
        
        for analysis in content_analyses:
            if analysis.relevance_score >= self.min_relevance_threshold:
                citation = SourceCitation(
                    url=f"source_{analysis.content_id}",  # Placeholder - would need actual URL
                    title=f"Source Analysis {analysis.content_id[:8]}",
                    relevance_score=analysis.relevance_score,
                    confidence=analysis.confidence,
                    credibility=analysis.source_credibility,
                    key_points=analysis.key_points[:3],
                    citation_text=f"Content Analysis {analysis.content_id[:8]} (Relevance: {analysis.relevance_score:.1f}/10)",
                    access_date=datetime.utcnow()
                )
                citations.append(citation)
        
        # Sort by relevance score
        citations.sort(key=lambda c: c.relevance_score, reverse=True)
        return citations[:10]  # Limit to top 10
    
    def _calculate_confidence(self, content_analyses: List[ContentAnalysis],
                            reasoning_chains: Optional[List[ReasoningChain]] = None) -> Tuple[float, ConfidenceLevel]:
        """Calculate overall confidence in the synthesized answer"""
        if not content_analyses:
            return 0.0, ConfidenceLevel.LOW
        
        # Base confidence from content analysis
        high_relevance_count = len([a for a in content_analyses if a.relevance_score >= self.min_relevance_threshold])
        avg_confidence = sum(a.confidence for a in content_analyses) / len(content_analyses)
        avg_relevance = sum(a.relevance_score for a in content_analyses) / len(content_analyses)
        
        # Calculate base confidence
        confidence = avg_confidence * 0.4 + (avg_relevance / 10.0) * 0.4
        
        # Boost for sufficient high-quality sources
        if high_relevance_count >= self.min_sources_for_high_confidence:
            confidence += 0.15
        
        # Boost for reasoning chains
        if reasoning_chains and len(reasoning_chains) > 0:
            chain_confidence = sum(c.confidence for c in reasoning_chains) / len(reasoning_chains)
            confidence += chain_confidence * 0.1
        
        # Penalty for low source count
        if len(content_analyses) < 2:
            confidence *= 0.7
        
        # Ensure confidence is in valid range
        confidence = max(0.0, min(1.0, confidence))
        
        # Determine confidence level
        if confidence >= 0.8:
            confidence_level = ConfidenceLevel.HIGH
        elif confidence >= 0.6:
            confidence_level = ConfidenceLevel.MEDIUM
        else:
            confidence_level = ConfidenceLevel.LOW
        
        return confidence, confidence_level
    
    def _extract_uncertainties(self, content_analyses: List[ContentAnalysis], 
                             contradictions: Optional[List[Contradiction]] = None) -> List[str]:
        """Extract uncertainty areas from content analyses"""
        uncertainties = []
        
        for analysis in content_analyses:
            uncertainties.extend(analysis.uncertainty_areas)
        
        # Add contradiction-related uncertainties
        if contradictions:
            for contradiction in contradictions:
                uncertainties.append(f"Conflicting information about {contradiction.topic}")
        
        # Remove duplicates and return top uncertainties
        unique_uncertainties = list(dict.fromkeys(uncertainties))
        return unique_uncertainties[:5]
    
    def _extract_contradiction_notes(self, contradictions: List[Contradiction]) -> List[str]:
        """Extract contradiction notes for display"""
        notes = []
        
        for contradiction in contradictions[:3]:  # Limit to top 3
            note = f"{contradiction.topic}: {contradiction.explanation}"
            notes.append(note)
        
        return notes
    
    def _get_confidence_emoji(self, confidence_level: ConfidenceLevel) -> str:
        """Get emoji for confidence level"""
        if confidence_level == ConfidenceLevel.HIGH:
            return "🟢"
        elif confidence_level == ConfidenceLevel.MEDIUM:
            return "🟡"
        else:
            return "🔴"
    
    def _get_relevance_indicator(self, relevance_score: float) -> str:
        """Get relevance indicator for citations"""
        if relevance_score >= 9.0:
            return "⭐⭐⭐"
        elif relevance_score >= 8.0:
            return "⭐⭐"
        elif relevance_score >= 7.0:
            return "⭐"
        else:
            return ""
    
    def _get_credibility_indicator(self, credibility: str) -> str:
        """Get credibility indicator for citations"""
        if credibility == "high":
            return " 🏆"
        elif credibility == "medium":
            return " 📊"
        else:
            return " ℹ️"
    
    def _trim_to_length(self, text: str, max_words: int) -> str:
        """Trim text to maximum word count"""
        words = text.split()
        if len(words) <= max_words:
            return text
        
        # Trim to max words and add ellipsis
        trimmed_words = words[:max_words-1]
        return ' '.join(trimmed_words) + '...'
    
    def get_stats(self) -> Dict[str, Any]:
        """Get synthesis engine usage statistics"""
        avg_processing_time = (
            self.processing_time_total / max(self.syntheses_count, 1)
        )
        
        high_confidence_rate = (
            self.high_confidence_answers / max(self.syntheses_count, 1)
        )
        
        return {
            'syntheses_count': self.syntheses_count,
            'comprehensive_answers_generated': self.comprehensive_answers_generated,
            'summaries_generated': self.summaries_generated,
            'high_confidence_answers': self.high_confidence_answers,
            'high_confidence_rate': round(high_confidence_rate, 3),
            'avg_processing_time_seconds': round(avg_processing_time, 3),
            'total_processing_time_seconds': round(self.processing_time_total, 2),
            'min_relevance_threshold': self.min_relevance_threshold,
            'max_comprehensive_words': self.max_comprehensive_length,
            'max_summary_words': self.max_summary_length,
            'deepseek_available': self.deepseek_service is not None
        }
    
    def reset_stats(self):
        """Reset usage statistics"""
        self.syntheses_count = 0
        self.comprehensive_answers_generated = 0
        self.summaries_generated = 0
        self.high_confidence_answers = 0
        self.processing_time_total = 0.0