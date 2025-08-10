#!/usr/bin/env python3
"""
Result Validation Testing for DeepSeek Integration

This module validates the quality and formatting of DeepSeek research results,
ensuring they meet requirements for relevance, completeness, and structure.
"""

import asyncio
import json
import pytest
import logging
import re
from typing import List, Dict, Any, Optional
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

# Test imports
from app.service.enhanced_deepseek_research_service import (
    EnhancedDeepSeekResearchService,
    RelevanceEvaluator,
    AnswerAggregator,
    SummaryGenerator,
    ResultFormatter,
    AggregatedAnswer,
    SummaryResult,
    FormattedResult
)
from app.service.mongodb_service import MongoDBService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResultValidator:
    """Validates research results against quality criteria"""
    
    def __init__(self):
        self.validation_results = []
    
    def validate_research_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate complete research result structure and content"""
        
        validation = {
            'result_id': result.get('chat_id', 'unknown'),
            'timestamp': datetime.utcnow().isoformat(),
            'validation_passed': True,
            'issues': [],
            'scores': {}
        }
        
        # 1. Validate basic structure
        structure_score = self._validate_structure(result, validation)
        validation['scores']['structure'] = structure_score
        
        # 2. Validate content quality
        content_score = self._validate_content_quality(result, validation)
        validation['scores']['content_quality'] = content_score
        
        # 3. Validate metadata completeness
        metadata_score = self._validate_metadata(result, validation)
        validation['scores']['metadata'] = metadata_score
        
        # 4. Validate timing information
        timing_score = self._validate_timing(result, validation)
        validation['scores']['timing'] = timing_score
        
        # 5. Validate source information
        sources_score = self._validate_sources(result, validation)
        validation['scores']['sources'] = sources_score
        
        # Calculate overall score
        scores = validation['scores']
        overall_score = sum(scores.values()) / len(scores) if scores else 0
        validation['scores']['overall'] = overall_score
        
        # Determine if validation passed
        validation['validation_passed'] = overall_score >= 0.7 and len(validation['issues']) == 0
        
        self.validation_results.append(validation)
        return validation
    
    def _validate_structure(self, result: Dict[str, Any], validation: Dict[str, Any]) -> float:
        """Validate result structure"""
        required_fields = [
            'original_question',
            'chat_id', 
            'timestamp',
            'research_type',
            'success'
        ]
        
        score = 1.0
        
        for field in required_fields:
            if field not in result:
                validation['issues'].append(f"Missing required field: {field}")
                score -= 0.2
        
        # If successful, should have additional fields
        if result.get('success', False):
            success_fields = ['steps', 'timing_metrics', 'search_metrics']
            for field in success_fields:
                if field not in result:
                    validation['issues'].append(f"Missing success field: {field}")
                    score -= 0.1
        
        return max(0, score)
    
    def _validate_content_quality(self, result: Dict[str, Any], validation: Dict[str, Any]) -> float:
        """Validate content quality"""
        score = 1.0
        
        # Check if question is present and non-empty
        question = result.get('original_question', '')
        if not question or len(question.strip()) < 5:
            validation['issues'].append("Question is too short or empty")
            score -= 0.3
        
        # Check analysis content if present
        analysis = result.get('analysis', {})
        if analysis:
            analysis_content = analysis.get('analysis', '')
            if not analysis_content or len(analysis_content) < 50:
                validation['issues'].append("Analysis content is too short")
                score -= 0.3
            
            # Check confidence score
            confidence = analysis.get('confidence', 0)
            if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                validation['issues'].append("Invalid confidence score")
                score -= 0.2
        
        return max(0, score)
    
    def _validate_metadata(self, result: Dict[str, Any], validation: Dict[str, Any]) -> float:
        """Validate metadata completeness"""
        score = 1.0
        
        # Check timestamp format
        timestamp = result.get('timestamp', '')
        if not timestamp:
            validation['issues'].append("Missing timestamp")
            score -= 0.3
        else:
            try:
                datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except ValueError:
                validation['issues'].append("Invalid timestamp format")
                score -= 0.2
        
        # Check research type
        research_type = result.get('research_type', '')
        if research_type != 'enhanced_deepseek_research':
            validation['issues'].append("Invalid research type")
            score -= 0.2
        
        # Check search metrics if present
        search_metrics = result.get('search_metrics', {})
        if search_metrics:
            required_metrics = ['total_queries', 'total_results', 'successful_extractions']
            for metric in required_metrics:
                if metric not in search_metrics:
                    validation['issues'].append(f"Missing search metric: {metric}")
                    score -= 0.1
        
        return max(0, score)
    
    def _validate_timing(self, result: Dict[str, Any], validation: Dict[str, Any]) -> float:
        """Validate timing information"""
        score = 1.0
        
        timing_metrics = result.get('timing_metrics', {})
        if not timing_metrics:
            validation['issues'].append("Missing timing metrics")
            return 0.5
        
        # Check for reasonable timing values
        for metric_name, value in timing_metrics.items():
            if isinstance(value, (int, float)):
                if value < 0:
                    validation['issues'].append(f"Negative timing value: {metric_name}")
                    score -= 0.2
                elif value > 600:  # More than 10 minutes
                    validation['issues'].append(f"Excessive timing value: {metric_name}")
                    score -= 0.1
        
        return max(0, score)
    
    def _validate_sources(self, result: Dict[str, Any], validation: Dict[str, Any]) -> float:
        """Validate source information"""
        score = 1.0
        
        sources = result.get('sources', [])
        if not sources:
            validation['issues'].append("No sources provided")
            return 0.3
        
        for i, source in enumerate(sources):
            if not isinstance(source, dict):
                validation['issues'].append(f"Source {i} is not a dict")
                score -= 0.2
                continue
            
            # Check required source fields
            if 'url' not in source or not source['url']:
                validation['issues'].append(f"Source {i} missing URL")
                score -= 0.1
            
            if 'title' not in source or not source['title']:
                validation['issues'].append(f"Source {i} missing title")
                score -= 0.1
            
            # Check relevance score if present
            relevance_score = source.get('relevance_score', 0)
            if isinstance(relevance_score, (int, float)):
                if relevance_score < 0 or relevance_score > 10:
                    validation['issues'].append(f"Source {i} invalid relevance score")
                    score -= 0.1
        
        return max(0, score)
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validations"""
        if not self.validation_results:
            return {'total_validations': 0}
        
        total = len(self.validation_results)
        passed = sum(1 for v in self.validation_results if v['validation_passed'])
        
        avg_scores = {}
        for result in self.validation_results:
            for score_type, score in result['scores'].items():
                if score_type not in avg_scores:
                    avg_scores[score_type] = []
                avg_scores[score_type].append(score)
        
        for score_type in avg_scores:
            avg_scores[score_type] = sum(avg_scores[score_type]) / len(avg_scores[score_type])
        
        return {
            'total_validations': total,
            'passed_validations': passed,
            'pass_rate': passed / total if total > 0 else 0,
            'average_scores': avg_scores,
            'common_issues': self._get_common_issues()
        }
    
    def _get_common_issues(self) -> List[str]:
        """Get most common validation issues"""
        issue_counts = {}
        for result in self.validation_results:
            for issue in result['issues']:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        return sorted(issue_counts.keys(), key=lambda x: issue_counts[x], reverse=True)[:5]

@pytest.fixture
async def mock_research_service():
    """Create mock research service for result validation"""
    
    mock_mongodb = AsyncMock(spec=MongoDBService)
    mock_mongodb.get_cached_content = AsyncMock(return_value=None)
    mock_mongodb.cache_content = AsyncMock(return_value=True)
    mock_mongodb.create_research_indexes = AsyncMock(return_value=True)
    mock_mongodb.get_cache_stats = AsyncMock(return_value={'total_entries': 100, 'successful_entries': 85})
    
    service = EnhancedDeepSeekResearchService(mongodb_service=mock_mongodb)
    
    # Mock web search
    service.web_search.search = AsyncMock(return_value=[
        {
            'title': 'High Quality Source 1',
            'url': 'https://example.com/source1',
            'snippet': 'Comprehensive information about the topic with detailed analysis.',
            'display_link': 'example.com'
        },
        {
            'title': 'Relevant Source 2', 
            'url': 'https://example.com/source2',
            'snippet': 'Additional relevant details and supporting information.',
            'display_link': 'example.com'
        }
    ])
    
    # Mock content extraction
    service.content_extractor.extract_content = AsyncMock(return_value={
        'url': 'https://example.com/source1',
        'title': 'Comprehensive Analysis Article',
        'content': 'This is comprehensive content that provides detailed analysis of the topic. ' * 20,
        'success': True,
        'method': 'brightdata_api'
    })
    
    # Mock AI responses
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.usage = MagicMock()
    mock_response.usage.total_tokens = 250
    
    # Different responses for different phases
    responses = {
        'query_generation': 'Query: topic analysis\nQuery: detailed information\nQuery: comprehensive review',
        'relevance_evaluation': '8.5',
        'analysis_generation': '''## Analysis
Based on the comprehensive research, this topic shows significant importance with multiple key aspects.

## Key Findings
- Finding 1: Important aspect discovered
- Finding 2: Secondary but relevant information  
- Finding 3: Supporting evidence found

## Confidence Assessment
85% - High confidence based on multiple reliable sources'''
    }
    
    call_count = 0
    async def mock_ai_call(**kwargs):
        nonlocal call_count
        call_count += 1
        
        # Return different responses based on call order
        if call_count == 1:
            mock_response.choices[0].message.content = responses['query_generation']
        elif 'relevance' in str(kwargs).lower():
            mock_response.choices[0].message.content = responses['relevance_evaluation']
        else:
            mock_response.choices[0].message.content = responses['analysis_generation']
        
        return mock_response
    
    service.client.chat.completions.create = mock_ai_call
    
    return service

@pytest.mark.asyncio
async def test_complete_result_validation(mock_research_service):
    """Test validation of complete research results"""
    
    validator = ResultValidator()
    
    # Run research to generate results
    result = await mock_research_service.conduct_deepseek_research(
        "What are the benefits of renewable energy technology?",
        "validation-test-chat"
    )
    
    # Validate the result
    validation = validator.validate_research_result(result)
    
    # Check validation results
    assert validation['validation_passed'], f"Validation failed: {validation['issues']}"
    assert validation['scores']['overall'] >= 0.7, f"Overall score too low: {validation['scores']['overall']}"
    
    # Log validation results
    logger.info("Complete Result Validation:")
    logger.info(f"  Validation Passed: {validation['validation_passed']}")
    logger.info(f"  Overall Score: {validation['scores']['overall']:.2f}")
    logger.info(f"  Issues Found: {len(validation['issues'])}")
    for score_type, score in validation['scores'].items():
        if score_type != 'overall':
            logger.info(f"    {score_type}: {score:.2f}")
    
    return validation

@pytest.mark.asyncio
async def test_relevance_evaluator_validation():
    """Test relevance evaluation component"""
    
    # Mock OpenAI client
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    
    evaluator = RelevanceEvaluator(mock_client, threshold=7.0)
    
    # Test content
    test_content = {
        'title': 'Renewable Energy Benefits Analysis',
        'url': 'https://example.com/renewable',
        'content': 'Detailed analysis of renewable energy benefits including cost savings, environmental impact, and sustainability factors.'
    }
    
    # Test different relevance scores
    test_scores = ['9.2', '7.8', '5.5', '3.1', 'invalid_score']
    results = []
    
    for score in test_scores:
        mock_response.choices[0].message.content = score
        
        result = await evaluator.evaluate_relevance(
            "What are the benefits of renewable energy?",
            test_content
        )
        results.append(result)
    
    # Validate results
    assert len(results) == 5
    
    # Check valid scores
    assert results[0]['score'] == 9.2
    assert results[0]['meets_threshold'] == True
    
    assert results[1]['score'] == 7.8  
    assert results[1]['meets_threshold'] == True
    
    assert results[2]['score'] == 5.5
    assert results[2]['meets_threshold'] == False
    
    assert results[3]['score'] == 3.1
    assert results[3]['meets_threshold'] == False
    
    # Check invalid score handling
    assert results[4]['score'] == 5.0  # Default fallback
    assert results[4]['meets_threshold'] == False
    
    logger.info("Relevance Evaluator Validation:")
    logger.info(f"  Test Cases: {len(results)}")
    logger.info(f"  Valid Scores: {sum(1 for r in results[:4] if 'error' not in r)}")
    logger.info(f"  Fallback Handling: ✅")
    
    return results

@pytest.mark.asyncio
async def test_answer_aggregator_validation():
    """Test answer aggregation component"""
    
    aggregator = AnswerAggregator(deduplication_threshold=0.8)
    
    # Create test evaluations with different relevance scores
    test_evaluations = [
        {
            'content': 'High quality answer about renewable energy benefits',
            'url': 'https://source1.com',
            'relevance': {'score': 9.2, 'meets_threshold': True}
        },
        {
            'content': 'Another high quality answer with similar information',
            'url': 'https://source2.com', 
            'relevance': {'score': 8.7, 'meets_threshold': True}
        },
        {
            'content': 'Medium quality answer',
            'url': 'https://source3.com',
            'relevance': {'score': 6.5, 'meets_threshold': False}
        },
        {
            'content': 'Low quality answer',
            'url': 'https://source4.com',
            'relevance': {'score': 4.2, 'meets_threshold': False}
        }
    ]
    
    # Aggregate answers
    aggregated = aggregator.aggregate_answers(test_evaluations)
    
    # Validate aggregation
    assert len(aggregated) >= 1, "Should have at least one aggregated answer"
    
    # Check that only high-relevance answers are included
    for answer in aggregated:
        assert answer.relevance_score >= 7.0, "Only high-relevance answers should be aggregated"
        assert answer.rank >= 1, "Answers should have ranking"
    
    # Check ranking order
    if len(aggregated) > 1:
        for i in range(len(aggregated) - 1):
            assert aggregated[i].relevance_score >= aggregated[i+1].relevance_score, "Should be ranked by relevance"
    
    logger.info("Answer Aggregator Validation:")
    logger.info(f"  Input Evaluations: {len(test_evaluations)}")
    logger.info(f"  High-Relevance Evaluations: {sum(1 for e in test_evaluations if e['relevance']['meets_threshold'])}")
    logger.info(f"  Aggregated Answers: {len(aggregated)}")
    logger.info(f"  Top Answer Score: {aggregated[0].relevance_score if aggregated else 'N/A'}")
    
    return aggregated

@pytest.mark.asyncio
async def test_summary_generator_validation():
    """Test summary generation component"""
    
    # Mock OpenAI client
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.usage = MagicMock()
    mock_response.usage.total_tokens = 180
    
    # Mock comprehensive summary response
    mock_response.choices[0].message.content = """【Summary】
Renewable energy technology offers significant benefits including cost reduction, environmental protection, and energy independence. These technologies have matured considerably and now provide competitive alternatives to fossil fuels.

【Relevance】
This information directly addresses the question about renewable energy benefits by covering the three main advantage categories: economic, environmental, and strategic benefits.

【Sources】
The information comes from industry analysis reports and government studies, providing reliable and authoritative perspectives on renewable energy advantages.
"""
    
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    
    generator = SummaryGenerator(mock_client)
    
    # Create test aggregated answer
    test_answer = AggregatedAnswer(
        content="Renewable energy provides cost savings, environmental benefits, and energy security",
        relevance_score=9.1,
        source_urls=["https://energy.gov/analysis", "https://iea.org/renewable-report"],
        confidence_level="high",
        extraction_time=datetime.utcnow(),
        is_deduplicated=False
    )
    
    # Generate summary
    summary = await generator.generate_summary(
        "What are the benefits of renewable energy technology?",
        test_answer
    )
    
    # Validate summary
    assert isinstance(summary, SummaryResult)
    assert summary.original_question == "What are the benefits of renewable energy technology?"
    assert len(summary.summary_text) > 100, "Summary should be comprehensive"
    assert summary.relevance_score == 9.1
    assert len(summary.source_urls) == 2
    assert 'overall_confidence' in summary.confidence_metrics
    assert summary.token_usage > 0
    
    # Validate summary content structure
    summary_text = summary.summary_text
    assert '【Summary】' in summary_text, "Should have summary section"
    assert '【Relevance】' in summary_text, "Should have relevance section"  
    assert '【Sources】' in summary_text, "Should have sources section"
    
    logger.info("Summary Generator Validation:")
    logger.info(f"  Input Relevance Score: {test_answer.relevance_score}")
    logger.info(f"  Summary Length: {len(summary.summary_text)} chars")
    logger.info(f"  Confidence: {summary.confidence_metrics['overall_confidence']:.2f}")
    logger.info(f"  Token Usage: {summary.token_usage}")
    logger.info(f"  Structured Format: ✅")
    
    return summary

@pytest.mark.asyncio  
async def test_result_formatter_validation():
    """Test result formatting component"""
    
    formatter = ResultFormatter()
    
    # Create test summary result
    test_summary = SummaryResult(
        original_question="What are the benefits of renewable energy?",
        summary_text="""【Summary】
Renewable energy offers cost savings, environmental protection, and energy independence.

【Relevance】
Directly addresses the renewable energy benefits question.

【Sources】
Based on authoritative industry reports.""",
        relevance_score=8.7,
        source_urls=["https://example1.com", "https://example2.com"],
        confidence_metrics={
            'overall_confidence': 0.85,
            'relevance_confidence': 0.82,
            'source_diversity_bonus': 0.10,
            'source_count': 2.0
        },
        generation_time=datetime.utcnow(),
        token_usage=195
    )
    
    # Format result
    formatted = formatter.format_final_result(test_summary)
    
    # Validate formatting
    assert isinstance(formatted, FormattedResult)
    assert formatted.title.startswith("Research:")
    assert formatted.format_type == "research_summary"
    assert len(formatted.content) > 200, "Formatted content should be comprehensive"
    
    # Check content structure
    content = formatted.content
    assert "# Research Results:" in content
    assert "## Summary" in content
    assert "## Relevance Score" in content
    assert "## Key Sources" in content
    assert "## Confidence Metrics" in content
    assert "8.7/10" in content, "Should show relevance score"
    assert "85%" in content, "Should show confidence percentage"
    
    # Validate metadata
    assert 'relevance_score' in formatted.metadata
    assert 'source_count' in formatted.metadata
    assert 'confidence_metrics' in formatted.metadata
    assert 'token_usage' in formatted.metadata
    
    logger.info("Result Formatter Validation:")
    logger.info(f"  Input Summary Length: {len(test_summary.summary_text)} chars")
    logger.info(f"  Formatted Length: {len(formatted.content)} chars")
    logger.info(f"  Format Type: {formatted.format_type}")
    logger.info(f"  Metadata Fields: {len(formatted.metadata)}")
    logger.info(f"  Structured Markdown: ✅")
    
    return formatted

@pytest.mark.asyncio
async def test_batch_result_validation(mock_research_service):
    """Test validation of multiple research results"""
    
    validator = ResultValidator()
    
    # Generate multiple results
    test_questions = [
        "What are the advantages of cloud computing?",
        "How does machine learning improve business processes?",
        "What are the security challenges in IoT devices?"
    ]
    
    results = []
    for i, question in enumerate(test_questions):
        result = await mock_research_service.conduct_deepseek_research(
            question, 
            f"batch-validation-{i}"
        )
        results.append(result)
    
    # Validate all results
    validations = []
    for result in results:
        validation = validator.validate_research_result(result)
        validations.append(validation)
    
    # Get validation summary
    summary = validator.get_validation_summary()
    
    # Check validation results
    assert summary['total_validations'] == 3
    assert summary['pass_rate'] >= 0.7, f"Pass rate too low: {summary['pass_rate']}"
    assert 'overall' in summary['average_scores']
    
    logger.info("Batch Result Validation:")
    logger.info(f"  Total Results: {summary['total_validations']}")
    logger.info(f"  Passed: {summary['passed_validations']}")
    logger.info(f"  Pass Rate: {summary['pass_rate']:.1%}")
    logger.info(f"  Average Overall Score: {summary['average_scores']['overall']:.2f}")
    
    if summary['common_issues']:
        logger.info("  Common Issues:")
        for issue in summary['common_issues'][:3]:
            logger.info(f"    - {issue}")
    
    return summary

if __name__ == "__main__":
    """Run validation tests when executed directly"""
    
    async def run_validation_tests():
        logger.info("Starting DeepSeek Result Validation Tests...")
        
        # Create mock service
        mock_mongodb = AsyncMock(spec=MongoDBService)
        mock_mongodb.get_cached_content = AsyncMock(return_value=None)
        mock_mongodb.cache_content = AsyncMock(return_value=True)
        mock_mongodb.create_research_indexes = AsyncMock(return_value=True)
        mock_mongodb.get_cache_stats = AsyncMock(return_value={'total_entries': 50})
        
        service = EnhancedDeepSeekResearchService(mongodb_service=mock_mongodb)
        
        # Add mock implementations
        service.web_search.search = AsyncMock(return_value=[
            {'title': 'Test Source', 'url': 'https://test.com', 'snippet': 'Test snippet'}
        ])
        service.content_extractor.extract_content = AsyncMock(return_value={
            'success': True, 'content': 'Test content', 'title': 'Test Title', 'url': 'https://test.com'
        })
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test AI response"
        mock_response.usage.total_tokens = 100
        service.client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Run validation tests
        logger.info("1. Testing complete result validation...")
        await test_complete_result_validation(service)
        
        logger.info("2. Testing relevance evaluator...")
        await test_relevance_evaluator_validation()
        
        logger.info("3. Testing answer aggregator...")
        await test_answer_aggregator_validation()
        
        logger.info("4. Testing summary generator...")
        await test_summary_generator_validation()
        
        logger.info("5. Testing result formatter...")
        await test_result_formatter_validation()
        
        logger.info("6. Testing batch validation...")
        await test_batch_result_validation(service)
        
        logger.info("All result validation tests completed successfully!")
    
    # Run tests
    asyncio.run(run_validation_tests())