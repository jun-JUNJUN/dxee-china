#!/usr/bin/env python3
"""
Enhanced DeepSeek Web Research with MongoDB Caching and Multi-Query Strategy v3.07
This script implements an advanced research workflow with MongoDB caching:
1. Multi-angle search query generation
2. MongoDB caching for scraped web content
3. Content deduplication and smart caching
4. Enhanced content filtering and source diversification
5. Iterative query refinement based on gaps
6. Comprehensive logging and performance analysis

New Features in v3.07:
- Data-driven statistical summary generation using DeepSeek API reasoning
- Automatic extraction of numerical metrics from scraped content
- Statistical ranking based on actual data found in sources
- Fallback to qualitative analysis when no metrics available
- Source URL attribution for all summary data
- Enhanced metrics extraction (revenue, users, market share, etc.)

Previous Features (v3.04):
- 10-minute time limit for research sessions
- Content summarization to handle DeepSeek token limits (65536 tokens)
- Intelligent input size management and token counting
- Early termination when relevance targets are met
- Batch processing for large content sets
- Optimized error handling for timeout and token limit scenarios
- Replaced web scraping with Bright Data API for reliable content extraction
- MongoDB integration for caching scraped web content
- Smart URL matching to avoid duplicate scraping
- Keywords tracking for better cache management
- Content freshness checking
- Professional content extraction via Bright Data API

Usage:
    python test_deepseek_advanced_web_research3_05.py

Environment variables:
    DEEPSEEK_API_KEY: Your DeepSeek API key
    DEEPSEEK_API_URL: The base URL for the DeepSeek API (default: https://api.deepseek.com)
    GOOGLE_API_KEY: Your Google Custom Search API key
    GOOGLE_CSE_ID: Your Google Custom Search Engine ID
    BRIGHTDATA_API_KEY: Your Bright Data API key for content extraction
    BRIGHTDATA_API_URL: Bright Data API endpoint (default: https://api.brightdata.com/datasets/v3/scrape)
    MONGODB_URI: MongoDB connection string (default: mongodb://localhost:27017)
"""

import os
import sys
import json
import asyncio
import logging
import requests
import re
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import AsyncOpenAI
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass
from collections import defaultdict

# Bright Data API for content extraction
try:
    import aiohttp
    import tiktoken  # For token counting
    import urllib.request
    import ssl
except ImportError as e:
    print(f"❌ Missing required libraries: {e}")
    print("📦 Please install: pip install aiohttp tiktoken")
    sys.exit(1)

# MongoDB libraries
try:
    from pymongo import MongoClient
    from motor.motor_asyncio import AsyncIOMotorClient
except ImportError as e:
    print(f"❌ Missing MongoDB libraries: {e}")
    print("📦 Please install: pip install pymongo motor")
    sys.exit(1)

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='deepseek_enhanced_research_v307.log',
    filemode='a'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger('').addHandler(console)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants for v3.07 optimizations
MAX_RESEARCH_TIME = 600  # 10 minutes in seconds
MAX_CONTENT_LENGTH = 2000  # Max characters per content piece for DeepSeek
MAX_TOTAL_TOKENS = 50000  # Conservative limit for DeepSeek input
TARGET_SOURCES_PER_ITERATION = 8  # Optimal number of sources per analysis

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens in text using tiktoken"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        # Fallback: rough estimation (1 token ≈ 4 characters)
        return len(text) // 4

def summarize_content(content: str, max_length: int = MAX_CONTENT_LENGTH) -> str:
    """Summarize content to fit within max_length"""
    if len(content) <= max_length:
        return content
    
    # Extract key sentences (first, last, and middle parts)
    sentences = content.split('. ')
    if len(sentences) <= 3:
        return content[:max_length] + "..."
    
    # Take first 2 sentences, middle sentence, and last sentence
    first_part = '. '.join(sentences[:2])
    middle_idx = len(sentences) // 2
    middle_part = sentences[middle_idx]
    last_part = sentences[-1]
    
    summarized = f"{first_part}. ... {middle_part}. ... {last_part}"
    
    if len(summarized) > max_length:
        return content[:max_length] + "..."
    
    return summarized

def check_time_limit(start_time: float, max_duration: float = MAX_RESEARCH_TIME) -> bool:
    """Check if time limit has been exceeded"""
    return (time.time() - start_time) >= max_duration

@dataclass
class TimingMetrics:
    """Track timing metrics for performance analysis with time limits"""
    start_time: float
    end_time: Optional[float] = None
    phase_times: Dict[str, float] = None
    time_limit_exceeded: bool = False
    
    def __post_init__(self):
        if self.phase_times is None:
            self.phase_times = {}
    
    def start_phase(self, phase_name: str):
        """Start timing a phase"""
        if check_time_limit(self.start_time):
            self.time_limit_exceeded = True
            logger.warning(f"⏰ Time limit exceeded, skipping phase: {phase_name}")
            return False
            
        self.phase_times[f"{phase_name}_start"] = time.time()
        logger.info(f"⏱️ Starting phase: {phase_name}")
        return True
    
    def end_phase(self, phase_name: str):
        """End timing a phase"""
        start_key = f"{phase_name}_start"
        duration_key = f"{phase_name}_duration"
        
        if start_key in self.phase_times:
            duration = time.time() - self.phase_times[start_key]
            self.phase_times[duration_key] = duration
            logger.info(f"⏱️ Completed phase: {phase_name} in {duration:.2f}s")
            return duration
        return 0
    
    def get_total_duration(self) -> float:
        """Get total duration"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    def get_phase_summary(self) -> Dict[str, float]:
        """Get summary of all phase durations"""
        return {k: v for k, v in self.phase_times.items() if k.endswith('_duration')}

@dataclass
class SearchMetrics:
    """Track search and analysis metrics with v3.07 enhancements"""
    total_queries: int = 0
    total_results: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    source_types: Dict[str, int] = None
    relevance_scores: List[float] = None
    content_summarized: int = 0
    tokens_saved: int = 0
    statistical_data_found: int = 0  # v3.07: Track statistical data extraction
    
    def __post_init__(self):
        if self.source_types is None:
            self.source_types = defaultdict(int)
        if self.relevance_scores is None:
            self.relevance_scores = []

@dataclass
class AnswerVersion:
    """Track a version of the evolving answer"""
    version: int
    answer: str
    confidence_score: float
    sources_count: int
    timestamp: datetime
    improvement_reason: str

class ProgressiveAnswerTracker:
    """Track and update the answer progressively as more data is found"""
    
    def __init__(self, original_question: str):
        self.original_question = original_question
        self.current_answer = "Research in progress..."
        self.answer_versions: List[AnswerVersion] = []
        self.current_confidence = 0.0
        self.sources_analyzed = 0
        
    def update_answer(self, new_sources: List[Dict[str, Any]], api_client: AsyncOpenAI) -> bool:
        """Update the answer based on new sources"""
        if not new_sources:
            return False
            
        # Only process successful extractions
        successful_sources = [s for s in new_sources if s.get('success', False)]
        if not successful_sources:
            return False
            
        self.sources_analyzed += len(successful_sources)
        
        # Create a quick update prompt
        return True  # Will be implemented in the async method
    
    async def async_update_answer(self, new_sources: List[Dict[str, Any]], api_client: AsyncOpenAI) -> bool:
        """Async version of answer update"""
        if not new_sources:
            return False
            
        successful_sources = [s for s in new_sources if s.get('success', False)]
        if not successful_sources:
            return False
            
        try:
            # Prepare source summaries for quick analysis
            source_summaries = []
            for i, source in enumerate(successful_sources[:5]):  # Limit for speed
                summary = f"Source {i+1}: {source['title']}\nContent: {source['content'][:800]}..."
                source_summaries.append(summary)
            
            # Quick answer update prompt
            update_prompt = f"""Based on the following new sources, provide an updated answer to the research question.

Research Question: {self.original_question}

Current Answer: {self.current_answer}

New Sources:
{chr(10).join(source_summaries)}

Instructions:
1. Integrate new information with the current answer
2. Improve accuracy and completeness
3. Keep the answer concise but comprehensive
4. If new sources significantly improve the answer, provide the updated version
5. If no significant improvement, return "NO_UPDATE"

Updated Answer:"""

            response = await api_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "user", "content": update_prompt}
                ],
                stream=False,
                timeout=20.0
            )
            
            updated_answer = response.choices[0].message.content.strip()
            
            if updated_answer and updated_answer != "NO_UPDATE" and len(updated_answer) > 20:
                # Calculate confidence based on source quality and count
                avg_quality = sum(s.get('domain_info', {}).get('quality_score', 5) for s in successful_sources) / len(successful_sources)
                new_confidence = min(0.9, (self.sources_analyzed * 0.1) + (avg_quality / 10))
                
                # Only update if confidence improved or answer significantly changed
                if new_confidence > self.current_confidence or len(updated_answer) > len(self.current_answer) * 1.2:
                    # Save version
                    version = AnswerVersion(
                        version=len(self.answer_versions) + 1,
                        answer=updated_answer,
                        confidence_score=new_confidence,
                        sources_count=self.sources_analyzed,
                        timestamp=datetime.utcnow(),
                        improvement_reason=f"Added {len(successful_sources)} new sources"
                    )
                    
                    self.answer_versions.append(version)
                    self.current_answer = updated_answer
                    self.current_confidence = new_confidence
                    
                    logger.info(f"📝 Answer updated (v{version.version}): confidence {new_confidence:.2f}, {self.sources_analyzed} sources")
                    print(f"\n📝 ANSWER UPDATE v{version.version} (confidence: {new_confidence:.2f}):")
                    print(f"📋 {updated_answer[:200]}..." if len(updated_answer) > 200 else updated_answer)
                    print(f"📊 Based on {self.sources_analyzed} sources\n")
                    
                    return True
                    
        except Exception as e:
            logger.warning(f"⚠️ Answer update failed: {e}")
            
        return False
    
    def get_final_answer(self) -> Dict[str, Any]:
        """Get the final comprehensive answer"""
        return {
            'question': self.original_question,
            'final_answer': self.current_answer,
            'confidence_score': self.current_confidence,
            'sources_analyzed': self.sources_analyzed,
            'versions_count': len(self.answer_versions),
            'answer_evolution': [
                {
                    'version': v.version,
                    'confidence': v.confidence_score,
                    'sources_count': v.sources_count,
                    'timestamp': v.timestamp.isoformat(),
                    'improvement': v.improvement_reason
                } for v in self.answer_versions
            ]
        }

@dataclass
class OptimizedResearchSession:
    """Track optimized research session with time and token constraints"""
    session_id: str
    question: str
    start_time: float
    max_duration: int = 600  # 10 minutes
    target_sources: int = 10
    max_tokens_per_request: int = 50000
    
    # Progress tracking
    current_phase: str = "initialization"
    sources_processed: int = 0
    tokens_used: int = 0
    cache_hits: int = 0
    
    # Results
    progressive_answer: str = ""
    confidence_score: float = 0.0
    completion_status: str = "in_progress"  # in_progress, completed, time_limited

@dataclass
class OptimizedSource:
    """Enhanced source with optimization metadata"""
    url: str
    title: str
    content: str
    priority_score: float
    token_count: int
    extraction_time: float
    quality_score: int
    relevance_score: float
    from_cache: bool = False
    
    # Optimization metadata
    original_length: int = 0
    summarized: bool = False
    processing_priority: int = 0

class TimeManager:
    """Enforce strict time limits and coordinate early termination"""
    
    def __init__(self, max_duration: int = 600):
        self.max_duration = max_duration
        self.start_time = None
        self.phase_budgets = {
            "query_generation": 30,      # 5%
            "web_search": 60,           # 10%
            "content_extraction": 360,   # 60%
            "analysis": 120,            # 20%
            "summary_generation": 30     # 5%
        }
        
    def start_timer(self) -> float:
        """Initialize research timer"""
        self.start_time = time.time()
        logger.info(f"⏰ Timer started - {self.max_duration}s limit")
        return self.start_time
        
    def check_time_remaining(self) -> int:
        """Get remaining time in seconds"""
        if self.start_time is None:
            return self.max_duration
        elapsed = time.time() - self.start_time
        return max(0, int(self.max_duration - elapsed))
        
    def should_terminate(self) -> bool:
        """Check if early termination needed (8-minute mark)"""
        if self.start_time is None:
            return False
        elapsed = time.time() - self.start_time
        return elapsed >= (self.max_duration - 120)  # 8 minutes
        
    def get_phase_time_budget(self, phase: str) -> int:
        """Allocate time per phase"""
        return self.phase_budgets.get(phase, 60)
        
    def is_phase_time_exceeded(self, phase: str, phase_start: float) -> bool:
        """Check if phase time budget exceeded"""
        phase_elapsed = time.time() - phase_start
        phase_budget = self.get_phase_time_budget(phase)
        return phase_elapsed >= phase_budget

class TokenOptimizer:
    """Manage content size and prevent API token limit errors"""
    
    def __init__(self, max_tokens: int = 50000):
        self.max_tokens = max_tokens
        self.encoding = None
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except:
            logger.warning("⚠️ Could not load tiktoken encoding, using estimation")
    
    def count_tokens(self, text: str) -> int:
        """Accurate token counting"""
        if self.encoding:
            try:
                return len(self.encoding.encode(text))
            except:
                pass
        # Fallback estimation
        return len(text) // 4
    
    def count_total_tokens(self, sources: List[Dict[str, Any]]) -> int:
        """Count total tokens across all sources"""
        total = 0
        for source in sources:
            content = source.get('content', '')
            total += self.count_tokens(content)
        return total
    
    def optimize_content(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Intelligent content reduction"""
        if not sources:
            return sources
            
        total_tokens = self.count_total_tokens(sources)
        if total_tokens <= self.max_tokens:
            return sources
            
        logger.info(f"🔧 Optimizing content: {total_tokens} -> target {self.max_tokens} tokens")
        
        # Priority-based selection and summarization
        optimized = []
        remaining_tokens = self.max_tokens
        
        # Sort by priority (quality * relevance)
        sorted_sources = sorted(sources, 
                              key=lambda s: (s.get('domain_info', {}).get('quality_score', 5) * 
                                           s.get('relevance_score', 0.5)), 
                              reverse=True)
        
        for source in sorted_sources:
            content = source.get('content', '')
            content_tokens = self.count_tokens(content)
            
            if content_tokens <= remaining_tokens:
                optimized.append(source)
                remaining_tokens -= content_tokens
            elif remaining_tokens > 500:  # Minimum viable content
                # Summarize to fit
                summarized_source = self.summarize_source(source, remaining_tokens - 100)
                if summarized_source:
                    optimized.append(summarized_source)
                    remaining_tokens -= self.count_tokens(summarized_source.get('content', ''))
            
            if remaining_tokens <= 100:
                break
                
        logger.info(f"📊 Optimized to {len(optimized)}/{len(sources)} sources, ~{self.count_total_tokens(optimized)} tokens")
        return optimized
    
    def summarize_source(self, source: Dict[str, Any], target_tokens: int) -> Optional[Dict[str, Any]]:
        """Smart content summarization"""
        content = source.get('content', '')
        if not content:
            return None
            
        # Calculate target length (rough estimation: 1 token = 4 chars)
        target_length = target_tokens * 4
        
        if len(content) <= target_length:
            return source
            
        # Intelligent summarization - keep key sections
        lines = content.split('\n')
        important_lines = []
        
        # Keep lines with numbers, statistics, or key terms
        key_indicators = ['percent', '%', 'million', 'billion', 'users', 'revenue', 'market', 'growth']
        
        for line in lines:
            line_lower = line.lower()
            if (any(indicator in line_lower for indicator in key_indicators) or
                any(char.isdigit() for char in line) or
                len(line.strip()) > 50):  # Substantial content
                important_lines.append(line)
                
        # If we have important lines, use them
        if important_lines:
            summarized_content = '\n'.join(important_lines)
        else:
            # Fallback: take beginning and end
            mid_point = len(content) // 2
            first_half = content[:target_length//2]
            second_half = content[mid_point:mid_point + target_length//2]
            summarized_content = first_half + '\n...\n' + second_half
            
        # Ensure we don't exceed target
        if len(summarized_content) > target_length:
            summarized_content = summarized_content[:target_length] + "..."
            
        summarized_source = source.copy()
        summarized_source['content'] = summarized_content
        summarized_source['summarized'] = True
        summarized_source['original_length'] = len(content)
        
        return summarized_source
    
    def batch_content(self, sources: List[Dict[str, Any]], batch_size: int = None) -> List[List[Dict[str, Any]]]:
        """Split into processable batches"""
        if batch_size is None:
            # Determine batch size based on token limits
            avg_tokens = self.count_total_tokens(sources) / len(sources) if sources else 1000
            batch_size = max(1, int(self.max_tokens // avg_tokens))
            
        batches = []
        for i in range(0, len(sources), batch_size):
            batch = sources[i:i + batch_size]
            batches.append(batch)
            
        logger.info(f"📦 Split into {len(batches)} batches (max {batch_size} sources each)")
        return batches

class ContentPrioritizer:
    """Rank and prioritize sources for optimal resource utilization"""
    
    def __init__(self):
        self.domain_quality_cache = {}
        
    def calculate_priority_score(self, source: Dict[str, Any]) -> float:
        """Multi-factor scoring"""
        domain_quality = source.get('domain_info', {}).get('quality_score', 5) / 10.0  # 40%
        relevance = source.get('relevance_score', 0.5)  # 30%
        cache_bonus = 0.2 if source.get('from_cache', False) else 0.0  # 20%
        
        # Content quality ratio - length vs quality (10%)
        content_length = len(source.get('content', ''))
        quality_ratio = min(1.0, content_length / 1000) * 0.1  # Normalize to 1000 chars
        
        priority = (domain_quality * 0.4 + 
                   relevance * 0.3 + 
                   cache_bonus + 
                   quality_ratio)
                   
        return min(1.0, priority)
    
    def rank_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort by priority"""
        for source in sources:
            source['priority_score'] = self.calculate_priority_score(source)
            
        return sorted(sources, key=lambda s: s['priority_score'], reverse=True)
    
    def filter_by_time_budget(self, sources: List[Dict[str, Any]], time_remaining: int) -> List[Dict[str, Any]]:
        """Time-aware filtering"""
        if time_remaining <= 0:
            return []
            
        # Estimate processing time per source (rough: 30-60 seconds each)
        avg_processing_time = 45
        max_sources = max(1, time_remaining // avg_processing_time)
        
        ranked_sources = self.rank_sources(sources)
        filtered = ranked_sources[:max_sources]
        
        logger.info(f"⏱️ Time budget: {time_remaining}s allows ~{max_sources} sources (filtered {len(sources)} -> {len(filtered)})")
        return filtered

class ProgressiveResponseGenerator:
    """Build and update responses incrementally as data becomes available"""
    
    def __init__(self):
        self.response_state = {}
        
    def initialize_response(self, question: str) -> Dict[str, Any]:
        """Set up progressive tracking"""
        session_id = f"session_{int(time.time())}"
        self.response_state[session_id] = {
            'question': question,
            'current_answer': "🔍 Research initiated...",
            'confidence': 0.0,
            'sources_count': 0,
            'last_update': time.time(),
            'findings': []
        }
        return {'session_id': session_id}
    
    def update_with_sources(self, session_id: str, sources: List[Dict[str, Any]]) -> bool:
        """Incorporate new findings"""
        if session_id not in self.response_state:
            return False
            
        state = self.response_state[session_id]
        successful_sources = [s for s in sources if s.get('success', False)]
        
        if not successful_sources:
            return False
            
        # Update state
        state['sources_count'] += len(successful_sources)
        state['last_update'] = time.time()
        
        # Add key findings
        for source in successful_sources[:3]:  # Top 3 sources
            finding = {
                'title': source.get('title', 'Unknown'),
                'url': source.get('url', ''),
                'key_content': source.get('content', '')[:200] + "..." if len(source.get('content', '')) > 200 else source.get('content', ''),
                'quality': source.get('domain_info', {}).get('quality_score', 5)
            }
            state['findings'].append(finding)
            
        # Update confidence based on source count and quality
        avg_quality = sum(s.get('domain_info', {}).get('quality_score', 5) for s in successful_sources) / len(successful_sources)
        state['confidence'] = min(0.9, (state['sources_count'] * 0.08) + (avg_quality / 15))
        
        return True
    
    def generate_intermediate_summary(self, session_id: str) -> str:
        """Create progress summary"""
        if session_id not in self.response_state:
            return "No active research session"
            
        state = self.response_state[session_id]
        
        summary = f"🔍 Research Progress Update\n"
        summary += f"📊 Sources analyzed: {state['sources_count']}\n"
        summary += f"🎯 Confidence: {state['confidence']:.2f}\n"
        
        if state['findings']:
            summary += f"📋 Key findings from top sources:\n"
            for i, finding in enumerate(state['findings'][-3:], 1):  # Last 3 findings
                summary += f"  {i}. {finding['title']} (Quality: {finding['quality']}/10)\n"
                summary += f"     {finding['key_content']}\n"
                
        return summary
    
    def finalize_response(self, session_id: str, time_constrained: bool = False) -> Dict[str, Any]:
        """Complete final response"""
        if session_id not in self.response_state:
            return {'error': 'Session not found'}
            
        state = self.response_state[session_id]
        
        response = {
            'question': state['question'],
            'sources_analyzed': state['sources_count'],
            'confidence_score': state['confidence'],
            'time_constrained': time_constrained,
            'findings_summary': self.generate_intermediate_summary(session_id),
            'total_findings': len(state['findings'])
        }
        
        if time_constrained:
            response['limitation_note'] = "Analysis completed under time constraints. Results based on available data."
            response['confidence_score'] = min(response['confidence_score'], 0.7)  # Cap confidence for time-limited
            
        # Cleanup
        del self.response_state[session_id]
        
        return response

class TokenLimitHandler:
    """Handle token limit errors with recovery strategies"""
    
    def __init__(self, token_optimizer: TokenOptimizer):
        self.token_optimizer = token_optimizer
        self.retry_count = 0
        self.max_retries = 3
        self.last_error_time = 0
        
    def handle_token_error(self, sources: List[Dict[str, Any]], error: Exception) -> List[Dict[str, Any]]:
        """Recover from token limit errors by reducing content size"""
        error_str = str(error).lower()
        
        if ("exceeds the model's max input limit" in error_str or 
            "token limit" in error_str or 
            "maximum context length" in error_str):
            
            logger.warning(f"🔧 Token limit error detected: {error}")
            return self.progressive_reduction(sources)
            
        return sources
    
    def progressive_reduction(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply multiple reduction strategies in sequence"""
        if not sources:
            return sources
            
        original_count = len(sources)
        reduction_strategies = [
            self.remove_low_priority_sources,
            self.summarize_long_content,
            self.truncate_to_essentials,
            self.keep_only_top_sources
        ]
        
        for i, strategy in enumerate(reduction_strategies):
            logger.info(f"🔧 Applying reduction strategy {i+1}/{len(reduction_strategies)}")
            sources = strategy(sources)
            
            total_tokens = self.token_optimizer.count_total_tokens(sources)
            if total_tokens < self.token_optimizer.max_tokens:
                logger.info(f"✅ Token limit satisfied: {total_tokens} < {self.token_optimizer.max_tokens}")
                break
                
        final_count = len(sources)
        logger.info(f"📊 Reduced sources: {original_count} -> {final_count}")
        return sources
    
    def remove_low_priority_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove lowest priority sources (bottom 30%)"""
        if len(sources) <= 3:
            return sources
            
        # Sort by priority score
        sorted_sources = sorted(sources, 
                              key=lambda s: s.get('priority_score', 0.5), 
                              reverse=True)
        
        # Keep top 70%
        keep_count = max(3, int(len(sorted_sources) * 0.7))
        return sorted_sources[:keep_count]
    
    def summarize_long_content(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Summarize content that's longer than average"""
        if not sources:
            return sources
            
        # Calculate average content length
        avg_length = sum(len(s.get('content', '')) for s in sources) / len(sources)
        
        summarized_sources = []
        for source in sources:
            content_length = len(source.get('content', ''))
            if content_length > avg_length * 1.5:  # 50% above average
                # Summarize to average length
                target_tokens = int(avg_length // 4)  # Rough token estimation
                summarized = self.token_optimizer.summarize_source(source, target_tokens)
                summarized_sources.append(summarized or source)
            else:
                summarized_sources.append(source)
                
        return summarized_sources
    
    def truncate_to_essentials(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep only essential content from each source"""
        for source in sources:
            content = source.get('content', '')
            if len(content) > 1000:  # Aggressive truncation
                # Keep first 500 and last 300 characters with key indicators
                first_part = content[:500]
                last_part = content[-300:]
                
                # Find statistical information in middle
                lines = content.split('\n')
                stat_lines = []
                for line in lines:
                    if any(indicator in line.lower() for indicator in ['%', 'million', 'billion', 'users', 'revenue']):
                        stat_lines.append(line)
                        if len(stat_lines) >= 3:  # Limit statistical excerpts
                            break
                
                middle_stats = '\n'.join(stat_lines) if stat_lines else ""
                source['content'] = f"{first_part}\n\n[Key Statistics]\n{middle_stats}\n\n[...]\n{last_part}"
                source['truncated'] = True
                
        return sources
    
    def keep_only_top_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Emergency reduction: keep only top 5 highest quality sources"""
        if len(sources) <= 5:
            return sources
            
        # Sort by combined quality and priority
        sorted_sources = sorted(sources, 
                              key=lambda s: (s.get('domain_info', {}).get('quality_score', 5) * 
                                           s.get('priority_score', 0.5)), 
                              reverse=True)
        
        return sorted_sources[:5]

class TimeConstraintHandler:
    """Handle time constraints with graceful degradation"""
    
    def __init__(self, time_manager: TimeManager):
        self.time_manager = time_manager
        
    def check_and_adjust_phase(self, current_phase: str, elapsed_time: float) -> str:
        """Adjust research phases based on time constraints"""
        remaining_time = self.time_manager.max_duration - elapsed_time
        
        if remaining_time < 120:  # Less than 2 minutes
            return "emergency_summary"
        elif remaining_time < 240:  # Less than 4 minutes
            return "accelerated_analysis"
        elif current_phase == "content_extraction" and remaining_time < 300:
            return "priority_extraction_only"
        
        return current_phase
    
    def emergency_termination(self, partial_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate best possible response with available data"""
        sources_count = len(partial_results.get("sources", []))
        avg_quality = 5.0  # Default
        
        if sources_count > 0:
            sources = partial_results.get("sources", [])
            total_quality = sum(s.get('domain_info', {}).get('quality_score', 5) for s in sources)
            avg_quality = total_quality / sources_count
            
        confidence = min(0.7, (sources_count * 0.08) + (avg_quality / 15))
        
        return {
            "status": "time_limited",
            "message": "Research completed with time constraints",
            "confidence": confidence,
            "sources_analyzed": sources_count,
            "time_constraint_note": "Analysis limited by 10-minute time constraint",
            "partial_analysis": partial_results.get("analysis", "Limited analysis due to time constraints"),
            "available_sources": partial_results.get("sources", [])[:5]  # Include top 5 sources
        }
    
    def should_skip_phase(self, phase: str, remaining_time: int) -> bool:
        """Determine if a phase should be skipped due to time constraints"""
        phase_budgets = {
            "query_generation": 30,
            "web_search": 60,
            "content_extraction": 180,  # Minimum for extraction
            "analysis": 90,            # Minimum for analysis
            "summary_generation": 30
        }
        
        required_time = phase_budgets.get(phase, 60)
        return remaining_time < required_time

class CircuitBreaker:
    """Circuit breaker pattern for external API calls"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "open":
            if (time.time() - self.last_failure_time) > self.recovery_timeout:
                self.state = "half_open"
                logger.info("🔄 Circuit breaker moving to half-open state")
            else:
                raise Exception("Circuit breaker is open - too many recent failures")
        
        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise e
    
    def on_success(self):
        """Reset circuit breaker on successful call"""
        self.failure_count = 0
        if self.state == "half_open":
            self.state = "closed"
            logger.info("✅ Circuit breaker reset to closed state")
    
    def on_failure(self):
        """Handle failure in circuit breaker"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"⚠️ Circuit breaker opened after {self.failure_count} failures")

class MongoDBCacheService:
    """MongoDB service for caching scraped web content"""
    
    def __init__(self, mongodb_uri: str = None):
        self.mongodb_uri = mongodb_uri or os.environ.get('MONGODB_URI', 'mongodb://localhost:27017')
        self.client = None
        self.db = None
        self.collection = None
        self.is_connected = False
        
    async def connect(self):
        """Connect to MongoDB"""
        try:
            self.client = AsyncIOMotorClient(self.mongodb_uri)
            self.db = self.client.web_research_cache
            self.collection = self.db.scraped_content
            
            # Test connection
            await self.client.admin.command('ping')
            self.is_connected = True
            
            # Create indexes for better performance
            await self.collection.create_index("url", unique=True)
            await self.collection.create_index("keywords")
            await self.collection.create_index("accessed_date")
            
            logger.info("✅ Connected to MongoDB cache")
            
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            self.is_connected = False
    
    async def search_cached_content(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Search for cached content by keywords"""
        if not self.is_connected:
            return []
        
        try:
            # Build search query for keywords
            query = {
                "$or": [
                    {"keywords": {"$in": keywords}},
                    {"title": {"$regex": "|".join(keywords), "$options": "i"}},
                    {"content": {"$regex": "|".join(keywords), "$options": "i"}}
                ]
            }
            
            cursor = self.collection.find(query)
            cached_results = []
            
            async for doc in cursor:
                # Check if content is still fresh (within 14 days for better cache utilization)
                accessed_date = doc.get('accessed_date')
                if isinstance(accessed_date, datetime):
                    days_old = (datetime.utcnow() - accessed_date).total_seconds() / 86400  # More precise calculation
                    if days_old <= 14:  # Extended cache validity
                        cached_results.append(doc)
                        logger.info(f"🔍 Found cached content: {doc['url']} ({days_old:.1f} days old)")
                else:
                    # Include entries without accessed_date (legacy cache)
                    cached_results.append(doc)
                    logger.info(f"🔍 Found cached content (no date): {doc['url']}")
            
            logger.info(f"📊 Found {len(cached_results)} cached results for keywords: {keywords}")
            return cached_results
            
        except Exception as e:
            logger.error(f"❌ Cache search failed: {e}")
            return []
    
    async def get_cached_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Get cached content by URL"""
        if not self.is_connected:
            return None
        
        try:
            result = await self.collection.find_one({"url": url})
            if result:
                # Check freshness (within 7 days)
                accessed_date = result.get('accessed_date')
                if isinstance(accessed_date, datetime):
                    days_old = (datetime.utcnow() - accessed_date).days
                    if days_old <= 7:
                        logger.info(f"💾 Cache hit: {url} ({days_old} days old)")
                        return result
                    else:
                        logger.info(f"⏰ Cache expired: {url} ({days_old} days old)")
                        # Remove expired cache
                        await self.collection.delete_one({"url": url})
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Cache retrieval failed for {url}: {e}")
            return None
    
    async def save_content(self, url: str, title: str, content: str, keywords: List[str], 
                          method: str, word_count: int, domain_info: Dict[str, Any]) -> bool:
        """Save scraped content to cache"""
        if not self.is_connected:
            return False
        
        try:
            document = {
                "url": url,
                "title": title,
                "content": content,
                "keywords": keywords,
                "accessed_date": datetime.utcnow(),
                "method": method,
                "word_count": word_count,
                "domain_info": domain_info,
                "created_at": datetime.utcnow()
            }
            
            # Use upsert to handle duplicates
            await self.collection.replace_one(
                {"url": url}, 
                document, 
                upsert=True
            )
            
            logger.info(f"💾 Cached content: {url} ({word_count} words)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Cache save failed for {url}: {e}")
            return False
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.is_connected:
            return {"total_entries": 0, "connected": False}
        
        try:
            total_entries = await self.collection.count_documents({})
            fresh_entries = await self.collection.count_documents({
                "accessed_date": {"$gte": datetime.utcnow() - timedelta(days=7)}
            })
            
            return {
                "total_entries": total_entries,
                "fresh_entries": fresh_entries,
                "connected": True
            }
            
        except Exception as e:
            logger.error(f"❌ Cache stats failed: {e}")
            return {"total_entries": 0, "connected": False, "error": str(e)}
    
    async def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("🔌 MongoDB connection closed")

class BrightDataContentExtractor:
    """Hybrid Bright Data service with proxy fallback to Web Unlocker API"""
    
    def __init__(self, cache_service: MongoDBCacheService):
        self.api_key = os.environ.get('BRIGHTDATA_API_KEY', '1893dc9730e9a6ee6b79b263bffc61033781d58a93ab975250fa849a1c0094cf')
        self.api_url = os.environ.get('BRIGHTDATA_API_URL', 'https://api.brightdata.com/request')
        self.domain_quality_cache = {}
        self.cache_service = cache_service
        
        # Proxy configuration for SERP API
        self.proxy = 'http://brd-customer-hl_68b47d39-zone-serp_api1:w158p0glp07q@brd.superproxy.io:33335'
        
        # Create proxy opener for search URLs
        self.proxy_opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({'https': self.proxy, 'http': self.proxy}),
            urllib.request.HTTPSHandler(context=ssl._create_unverified_context())
        )
        
        if not self.api_key:
            logger.warning("⚠️ BRIGHTDATA_API_KEY not set. Content extraction will be limited.")
        else:
            logger.info("✅ Bright Data hybrid API configured successfully")
        
        # Session for backup requests if needed
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def assess_domain_quality(self, url: str) -> Dict[str, Any]:
        """Assess the quality and type of a domain for content extraction"""
        try:
            domain = urlparse(url).netloc.lower()
            
            # Check cache first
            if domain in self.domain_quality_cache:
                return self.domain_quality_cache[domain]
            
            # High-quality business/news domains
            high_quality_indicators = [
                'reuters.com', 'bloomberg.com', 'wsj.com', 'ft.com', 'economist.com',
                'forbes.com', 'fortune.com', 'businessweek.com', 'harvard.edu',
                'mckinsey.com', 'bcg.com', 'deloitte.com', 'pwc.com', 'kpmg.com',
                'gartner.com', 'forrester.com', 'idc.com', 'statista.com'
            ]
            
            # Medium-quality sources
            medium_quality_indicators = [
                'techcrunch.com', 'venturebeat.com', 'wired.com', 'arstechnica.com',
                'zdnet.com', 'computerworld.com', 'infoworld.com', 'cio.com',
                'wikipedia.org', 'investopedia.com'
            ]
            
            # Low-quality or problematic sources
            low_quality_indicators = [
                'reddit.com', 'quora.com', 'yahoo.com/answers', 'stackoverflow.com',
                'medium.com', 'linkedin.com/pulse', 'facebook.com', 'twitter.com'
            ]
            
            quality_score = 5  # Default medium quality
            source_type = "general"
            
            # Assess quality
            if any(indicator in domain for indicator in high_quality_indicators):
                quality_score = 9
                source_type = "premium"
            elif any(indicator in domain for indicator in medium_quality_indicators):
                quality_score = 7
                source_type = "reliable"
            elif any(indicator in domain for indicator in low_quality_indicators):
                quality_score = 3
                source_type = "social"
            elif domain.endswith('.edu') or domain.endswith('.gov'):
                quality_score = 8
                source_type = "academic"
            elif domain.endswith('.org'):
                quality_score = 6
                source_type = "organization"
            
            # Company websites (ending in .com with short domains)
            if domain.endswith('.com') and len(domain.split('.')) == 2 and len(domain.split('.')[0]) < 15:
                quality_score = max(quality_score, 6)
                source_type = "corporate"
            
            result = {
                'domain': domain,
                'quality_score': quality_score,
                'source_type': source_type,
                'recommended': quality_score >= 6,
                'brightdata_compatible': True  # Most domains work with Bright Data API
            }
            
            self.domain_quality_cache[domain] = result
            logger.info(f"🔍 Domain assessment: {domain} - Quality: {quality_score}/10, Type: {source_type}, API Compatible: Yes")
            return result
            
        except Exception as e:
            logger.warning(f"⚠️ Domain assessment failed for {url}: {e}")
            return {'domain': 'unknown', 'quality_score': 5, 'source_type': 'unknown', 'recommended': True, 'brightdata_compatible': True}
    
    async def extract_article_content(self, url: str, keywords: List[str] = None) -> Dict[str, Any]:
        """Extract article content with MongoDB caching support"""
        start_time = time.time()
        
        try:
            logger.info(f"🔄 Extracting content from: {url}")
            
            # Check cache first
            cached_content = await self.cache_service.get_cached_content(url)
            if cached_content:
                extraction_time = time.time() - start_time
                logger.info(f"💾 Using cached content: {len(cached_content.get('content', ''))} chars in {extraction_time:.2f}s")
                
                return {
                    'url': url,
                    'title': cached_content.get('title', 'No title'),
                    'content': cached_content.get('content', ''),
                    'method': f"cache_{cached_content.get('method', 'unknown')}",
                    'word_count': cached_content.get('word_count', 0),
                    'extraction_time': extraction_time,
                    'domain_info': cached_content.get('domain_info', {}),
                    'success': True,
                    'from_cache': True,
                    'cache_date': cached_content.get('accessed_date')
                }
            
            # Assess domain quality
            domain_info = self.assess_domain_quality(url)
            
            # Extract content using Bright Data API
            extraction_result = await self._extract_content_brightdata(url, domain_info)
            
            # Save to cache if extraction was successful
            if extraction_result['success'] and keywords:
                await self.cache_service.save_content(
                    url=url,
                    title=extraction_result['title'],
                    content=extraction_result['content'],
                    keywords=keywords,
                    method=extraction_result['method'],
                    word_count=extraction_result['word_count'],
                    domain_info=domain_info
                )
            
            extraction_result['from_cache'] = False
            return extraction_result
            
        except Exception as e:
            extraction_time = time.time() - start_time
            logger.error(f"❌ Content extraction failed for {url}: {e} in {extraction_time:.2f}s")
            return {
                'url': url,
                'title': 'Error',
                'content': f'Error extracting content: {str(e)}',
                'method': 'error',
                'word_count': 0,
                'extraction_time': extraction_time,
                'domain_info': {'quality_score': 0, 'source_type': 'error'},
                'success': False,
                'error': str(e),
                'from_cache': False
            }
    
    async def _extract_content_brightdata(self, url: str, domain_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content using hybrid Bright Data approach: proxy first, then Web Unlocker API"""
        start_time = time.time()
        
        if not self.api_key:
            return await self._fallback_extraction(url, domain_info, start_time)
        
        # First try: Proxy method (good for search results)
        try:
            logger.info(f"🔄 Trying Bright Data proxy for: {url}")
            
            # Use proxy in a separate thread to avoid blocking
            import asyncio
            import functools
            
            def proxy_request():
                try:
                    response = self.proxy_opener.open(url, timeout=15)
                    return response.read().decode()
                except Exception as e:
                    raise e
            
            # Run proxy request in thread pool
            loop = asyncio.get_event_loop()
            html_content = await loop.run_in_executor(None, proxy_request)
            
            extraction_time = time.time() - start_time
            
            # Extract title and content from HTML
            title = self._extract_title_from_html(html_content)
            text_content = self._extract_text_from_html(html_content)
            
            if text_content and len(text_content.strip()) > 50:
                logger.info(f"✅ Successful extraction via Bright Data proxy: {len(text_content)} chars in {extraction_time:.2f}s")
                
                return {
                    'url': url,
                    'title': title,
                    'content': text_content,
                    'method': 'brightdata_proxy',
                    'word_count': len(text_content.split()),
                    'extraction_time': extraction_time,
                    'domain_info': domain_info,
                    'success': True
                }
            else:
                logger.warning(f"⚠️ Bright Data proxy returned insufficient content for {url}")
                raise Exception("Insufficient content from proxy")
                
        except Exception as proxy_error:
            logger.warning(f"⚠️ Bright Data proxy failed for {url}: {proxy_error}")
            
            # Check if it's a 400 error (unsupported URL for SERP API)
            if "400" in str(proxy_error) or "HTTP Error 400" in str(proxy_error):
                logger.info(f"🔄 Trying Web Unlocker API for {url} (proxy returned 400)")
                return await self._extract_content_web_unlocker(url, domain_info, start_time)
            else:
                # Other errors, try Web Unlocker as fallback
                logger.info(f"🔄 Trying Web Unlocker API as fallback for {url}")
                return await self._extract_content_web_unlocker(url, domain_info, start_time)
    
    async def _extract_content_web_unlocker(self, url: str, domain_info: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """Extract content using Web Unlocker API (fallback method)"""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'zone': 'web_unlocker2',
                'url': url,
                'format': 'raw'
            }
            
            logger.info(f"🔄 Extracting content via Web Unlocker API: {url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status == 200:
                        html_content = await response.text()
                        extraction_time = time.time() - start_time
                        
                        title = self._extract_title_from_html(html_content)
                        text_content = self._extract_text_from_html(html_content)
                        
                        if text_content and len(text_content.strip()) > 50:
                            logger.info(f"✅ Successful extraction via Web Unlocker API: {len(text_content)} chars in {extraction_time:.2f}s")
                            
                            return {
                                'url': url,
                                'title': title,
                                'content': text_content,
                                'method': 'brightdata_web_unlocker',
                                'word_count': len(text_content.split()),
                                'extraction_time': extraction_time,
                                'domain_info': domain_info,
                                'success': True
                            }
                        else:
                            logger.warning(f"⚠️ Web Unlocker returned insufficient content for {url}")
                            return await self._fallback_extraction(url, domain_info, start_time)
                    
                    else:
                        error_text = await response.text()
                        logger.warning(f"⚠️ Web Unlocker API error {response.status} for {url}: {error_text}")
                        return await self._fallback_extraction(url, domain_info, start_time)
                        
        except Exception as e:
            logger.warning(f"⚠️ Web Unlocker API error for {url}: {e}")
            return await self._fallback_extraction(url, domain_info, start_time)
    
    def _extract_title_from_html(self, html_content: str) -> str:
        """Extract title from HTML content"""
        try:
            # Try to extract title
            title_match = re.search(r'<title[^>]*>([^<]+)</title>', html_content, re.IGNORECASE)
            if title_match:
                return title_match.group(1).strip()
            
            # Fallback to h1 tag
            h1_match = re.search(r'<h1[^>]*>([^<]+)</h1>', html_content, re.IGNORECASE)
            if h1_match:
                return h1_match.group(1).strip()
            
            return 'No title'
        except:
            return 'No title'
    
    def _extract_text_from_html(self, html_content: str) -> str:
        """Extract text content from HTML"""
        try:
            # Remove script and style elements
            html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
            html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
            
            # Remove HTML tags
            text_content = re.sub(r'<[^>]+>', ' ', html_content)
            
            # Clean up whitespace
            text_content = re.sub(r'\s+', ' ', text_content).strip()
            
            return text_content
        except:
            return ''
    
    async def _fallback_extraction(self, url: str, domain_info: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """Fallback content extraction when Bright Data API is unavailable"""
        try:
            # Simple fallback using requests
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Very basic content extraction
            content = response.text
            
            # Try to extract title
            title_match = re.search(r'<title[^>]*>([^<]+)</title>', content, re.IGNORECASE)
            title = title_match.group(1).strip() if title_match else 'No title'
            
            # Basic text extraction (remove HTML tags)
            text_content = re.sub(r'<[^>]+>', ' ', content)
            text_content = re.sub(r'\s+', ' ', text_content).strip()
            
            if len(text_content) > 100:
                extraction_time = time.time() - start_time
                logger.info(f"✅ Fallback extraction successful: {len(text_content)} chars in {extraction_time:.2f}s")
                
                return {
                    'url': url,
                    'title': title,
                    'content': text_content[:5000],  # Limit content length
                    'method': 'fallback_requests',
                    'word_count': len(text_content.split()),
                    'extraction_time': extraction_time,
                    'domain_info': domain_info,
                    'success': True
                }
                
        except Exception as e:
            logger.warning(f"⚠️ Fallback extraction failed for {url}: {e}")
        
        # Complete failure
        extraction_time = time.time() - start_time
        logger.error(f"❌ All extraction methods failed for {url} in {extraction_time:.2f}s")
        
        return {
            'url': url,
            'title': 'Extraction failed',
            'content': 'Could not extract content from this URL',
            'method': 'none',
            'word_count': 0,
            'extraction_time': extraction_time,
            'domain_info': domain_info,
            'success': False,
            'error': 'All extraction methods failed'
        }

class EnhancedGoogleWebSearchService:
    """Enhanced Google web search service with filtering and source diversification"""
    
    def __init__(self):
        self.api_key = os.environ.get('GOOGLE_API_KEY', '')
        self.cse_id = os.environ.get('GOOGLE_CSE_ID', '')
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        self.search_cache = {}  # Cache to avoid duplicate searches
        
        if not self.api_key:
            logger.warning("⚠️ GOOGLE_API_KEY not set. Web search functionality will be limited.")
        if not self.cse_id:
            logger.warning("⚠️ GOOGLE_CSE_ID not set. Web search functionality will be limited.")
    
    async def search_with_filters(self, query: str, num_results: int = 10, 
                                 exclude_domains: List[str] = None,
                                 prefer_domains: List[str] = None) -> List[Dict[str, Any]]:
        """Enhanced search with domain filtering"""
        if not self.api_key or not self.cse_id:
            logger.error("❌ Google API credentials not configured")
            return []
        
        # Check cache
        cache_key = f"{query}_{num_results}"
        if cache_key in self.search_cache:
            logger.info(f"🔄 Using cached results for: {query}")
            return self.search_cache[cache_key]
        
        try:
            # Build query with domain filters
            modified_query = query
            if exclude_domains:
                for domain in exclude_domains:
                    modified_query += f" -site:{domain}"
            
            params = {
                'key': self.api_key,
                'cx': self.cse_id,
                'q': modified_query,
                'num': min(num_results, 10),
                'safe': 'active'
            }
            
            logger.info(f"🔍 Google search: {modified_query}")
            start_time = time.time()
            
            response = requests.get(self.base_url, params=params, timeout=15)
            response.raise_for_status()
            
            search_time = time.time() - start_time
            data = response.json()
            results = []
            
            if 'items' in data:
                for item in data['items']:
                    result = {
                        'title': item.get('title', ''),
                        'link': item.get('link', ''),
                        'snippet': item.get('snippet', ''),
                        'displayLink': item.get('displayLink', ''),
                        'search_query': query
                    }
                    results.append(result)
            
            # Cache results
            self.search_cache[cache_key] = results
            
            logger.info(f"✅ Found {len(results)} search results in {search_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"❌ Google search failed: {e}")
            return []

class EnhancedScoreCalculator:
    """Enhanced score calculation with maximum iteration score logic"""
    
    def __init__(self):
        self.iteration_scores = []
        self.threshold = 7  # Default target threshold
    
    def add_iteration_score(self, score: int) -> None:
        """Add a score from an iteration"""
        if isinstance(score, (int, float)) and score > 0:
            self.iteration_scores.append(int(score))
    
    def get_final_score(self) -> int:
        """Get the maximum score from all iterations"""
        if not self.iteration_scores:
            return 0
        return max(self.iteration_scores)
    
    def is_target_achieved(self, threshold: int = None) -> bool:
        """Check if target threshold is achieved"""
        target = threshold or self.threshold
        return self.get_final_score() >= target
    
    def get_score_progression(self) -> list:
        """Get the progression of scores across iterations"""
        return self.iteration_scores.copy()
    
    def get_calculation_summary(self) -> dict:
        """Get a summary of the score calculation"""
        return {
            'final_relevance_score': self.get_final_score(),
            'target_achieved': self.is_target_achieved(),
            'score_progression': self.get_score_progression(),
            'iterations_with_scores': len(self.iteration_scores),
            'calculation_method': 'maximum_iteration_score'
        }

class RobustAPIResponseParser:
    """Robust parsing of DeepSeek API responses with fallback handling"""
    
    def __init__(self):
        self.relevance_patterns = [
            r'OVERALL_RELEVANCE_SCORE:\s*(\d+)',
            r'relevance.*?score.*?[\s:]*(\d+)',
            r'score.*?[\s:]*(\d+)(?:/10)?',
            r'(\d+)(?:/10)?\s*(?:out of 10|relevance)',
        ]
    
    def parse_analysis_response(self, response_text: str) -> dict:
        """Parse DeepSeek API response with fallback handling"""
        if not response_text:
            return self._create_fallback_response("Empty response", response_text)
        
        # Try direct JSON parsing first
        try:
            if response_text.strip().startswith('{') and response_text.strip().endswith('}'):
                parsed = json.loads(response_text)
                # Validate it has expected structure
                if isinstance(parsed, dict):
                    return self._enhance_parsed_response(parsed, response_text)
        except json.JSONDecodeError:
            logger.warning("JSON parsing failed, attempting text extraction")
        
        # Extract structured data from text response
        return self._extract_from_text_response(response_text)
    
    def _enhance_parsed_response(self, parsed: dict, original_text: str) -> dict:
        """Enhance parsed JSON response with extracted data"""
        # Ensure overall_relevance_score is present
        if 'overall_relevance_score' not in parsed:
            extracted_score = self._extract_relevance_score(original_text)
            if extracted_score > 0:
                parsed['overall_relevance_score'] = extracted_score
        
        parsed['parsing_method'] = 'json_parsed'
        return parsed
    
    def _extract_from_text_response(self, text: str) -> dict:
        """Extract key components from text-based responses"""
        relevance_score = self._extract_relevance_score(text)
        confidence_indicators = self._extract_confidence_markers(text)
        
        return {
            'analysis_content': text,
            'overall_relevance_score': relevance_score,
            'confidence_indicators': confidence_indicators,
            'parsing_method': 'text_extraction',
            'original_response_length': len(text),
            'extraction_success': relevance_score > 0
        }
    
    def _extract_relevance_score(self, text: str) -> int:
        """Extract relevance score using multiple patterns"""
        for pattern in self.relevance_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    score = int(match.group(1))
                    if 0 <= score <= 10:
                        return score
                except (ValueError, IndexError):
                    continue
        
        # Fallback: estimate score based on content quality indicators
        return self._estimate_score_from_content(text)
    
    def _estimate_score_from_content(self, text: str) -> int:
        """Estimate relevance score based on content quality indicators"""
        if not text or len(text.strip()) < 50:
            return 0
        
        quality_indicators = [
            'comprehensive analysis',
            'key findings',
            'statistics',
            'market share',
            'revenue',
            'data shows',
            'according to',
            'research indicates'
        ]
        
        found_indicators = sum(1 for indicator in quality_indicators 
                              if indicator.lower() in text.lower())
        
        # Score based on content length and quality indicators
        length_score = min(3, len(text) // 500)  # Up to 3 points for length
        indicator_score = min(5, found_indicators)  # Up to 5 points for indicators
        
        return min(8, max(2, length_score + indicator_score))
    
    def _extract_confidence_markers(self, text: str) -> list:
        """Extract confidence-indicating phrases from text"""
        confidence_patterns = [
            r'high confidence',
            r'strongly indicates',
            r'clear evidence',
            r'definitive data',
            r'comprehensive coverage',
            r'partial information',
            r'limited data',
            r'unclear',
            r'insufficient'
        ]
        
        markers = []
        for pattern in confidence_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                markers.append(pattern.replace('r\'', '').replace('\'', ''))
        
        return markers
    
    def _create_fallback_response(self, error_reason: str, original_text: str) -> dict:
        """Create fallback response when parsing completely fails"""
        return {
            'analysis_content': original_text[:1000] if original_text else 'No content available',
            'overall_relevance_score': 1,  # Minimal score to indicate failure
            'parsing_method': 'fallback',
            'error_reason': error_reason,
            'confidence_indicators': [],
            'extraction_success': False
        }

class EnhancedDisplayFormatter:
    """Enhanced display formatting for comprehensive analysis"""
    
    def __init__(self):
        self.section_patterns = {
            'executive_summary': [
                r'executive summary[:\s]*(.+?)(?=\n\n|\n[A-Z]|$)',
                r'summary[:\s]*(.+?)(?=\n\n|\n[A-Z]|$)',
                r'overview[:\s]*(.+?)(?=\n\n|\n[A-Z]|$)'
            ],
            'key_findings': [
                r'key findings?[:\s]*(.+?)(?=\n\n|\n[A-Z]|$)',
                r'main findings?[:\s]*(.+?)(?=\n\n|\n[A-Z]|$)',
                r'findings?[:\s]*(.+?)(?=\n\n|\n[A-Z]|$)'
            ],
            'statistical_data': [
                r'statistics?[:\s]*(.+?)(?=\n\n|\n[A-Z]|$)',
                r'data[:\s]*(.+?)(?=\n\n|\n[A-Z]|$)',
                r'numbers?[:\s]*(.+?)(?=\n\n|\n[A-Z]|$)',
                r'metrics?[:\s]*(.+?)(?=\n\n|\n[A-Z]|$)'
            ]
        }
    
    def format_comprehensive_analysis(self, analysis_data: dict) -> str:
        """Format comprehensive analysis for prominent display"""
        if not analysis_data:
            return "📋 COMPREHENSIVE ANALYSIS:\n❌ No analysis data available"
        
        content = analysis_data.get('analysis_content', '') or analysis_data.get('comprehensive_answer', '')
        
        if not content:
            return "📋 COMPREHENSIVE ANALYSIS:\n❌ No analysis content available"
        
        # Extract structured sections
        sections = self._extract_analysis_sections(content)
        
        formatted_output = []
        formatted_output.append("📋 COMPREHENSIVE ANALYSIS SUMMARY:")
        formatted_output.append("=" * 60)
        
        # Add relevance score prominently
        relevance_score = analysis_data.get('overall_relevance_score', 0)
        parsing_method = analysis_data.get('parsing_method', 'unknown')
        formatted_output.append(f"🎯 Relevance Score: {relevance_score}/10 (Method: {parsing_method})")
        formatted_output.append("")
        
        # Add structured sections if found
        if sections.get('executive_summary'):
            formatted_output.append("🎯 Executive Summary:")
            formatted_output.append(self._format_section_content(sections['executive_summary']))
            formatted_output.append("")
        
        if sections.get('key_findings'):
            formatted_output.append("📊 Key Findings:")
            formatted_output.append(self._format_section_content(sections['key_findings']))
            formatted_output.append("")
        
        if sections.get('statistical_data'):
            formatted_output.append("📈 Statistical Data:")
            formatted_output.append(self._format_section_content(sections['statistical_data']))
            formatted_output.append("")
        
        # If no structured sections found, show first part of content
        if not any(sections.values()):
            formatted_output.append("📄 Analysis Content:")
            preview = self._create_content_preview(content)
            formatted_output.append(preview)
        
        # Add confidence indicators if available
        confidence_indicators = analysis_data.get('confidence_indicators', [])
        if confidence_indicators:
            formatted_output.append("🔍 Confidence Indicators:")
            for indicator in confidence_indicators[:3]:  # Show top 3
                formatted_output.append(f"  • {indicator}")
            formatted_output.append("")
        
        return "\n".join(formatted_output)
    
    def _extract_analysis_sections(self, content: str) -> dict:
        """Extract structured sections from analysis content"""
        sections = {}
        
        for section_name, patterns in self.section_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                if match:
                    section_content = match.group(1).strip()
                    if len(section_content) > 20:  # Minimum content length
                        sections[section_name] = section_content
                        break
        
        return sections
    
    def _format_section_content(self, content: str) -> str:
        """Format section content for better readability"""
        # Clean up the content
        content = content.strip()
        
        # Add bullet points if content looks like a list
        if '\n-' in content or content.count('.') > 2:
            lines = content.split('\n')
            formatted_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('•') and not line.startswith('-'):
                    if line.endswith('.') or line.endswith(':'):
                        formatted_lines.append(f"  • {line}")
                    else:
                        formatted_lines.append(f"    {line}")
                elif line:
                    formatted_lines.append(f"  {line}")
            content = '\n'.join(formatted_lines)
        else:
            # Indent single paragraph
            content = f"  {content}"
        
        return content
    
    def _create_content_preview(self, content: str) -> str:
        """Create a preview of the content when no sections are found"""
        # Take first 800 characters and try to break at sentence boundary
        preview = content[:800]
        if len(content) > 800:
            # Try to break at last sentence
            last_period = preview.rfind('.')
            if last_period > 400:  # Only if we have substantial content before the period
                preview = preview[:last_period + 1]
            preview += "\n  [... content continues ...]"
        
        # Indent the preview
        lines = preview.split('\n')
        indented_lines = [f"  {line}" if line.strip() else line for line in lines]
        return '\n'.join(indented_lines)
    
    def format_simple_analysis(self, analysis_data: dict) -> str:
        """Simple fallback formatting when structured parsing fails"""
        if not analysis_data:
            return "📋 ANALYSIS: No data available"
        
        content = analysis_data.get('analysis_content', '') or analysis_data.get('comprehensive_answer', '')
        score = analysis_data.get('overall_relevance_score', 0)
        
        formatted = [
            "📋 ANALYSIS SUMMARY:",
            f"Score: {score}/10",
            ""
        ]
        
        if content:
            preview = content[:1000]
            if len(content) > 1000:
                preview += "..."
            formatted.append(f"Content: {preview}")
        else:
            formatted.append("Content: No analysis content available")
        
        return "\n".join(formatted)

class SuccessValidator:
    """Multi-criteria success determination validator"""
    
    def __init__(self):
        self.relevance_threshold = 7  # Default threshold
        self.min_sources_threshold = 3  # Minimum sources for success
        self.min_content_length = 200  # Minimum analysis content length
    
    def determine_research_success(self, metrics: dict, analysis: dict, sources: list = None) -> dict:
        """Comprehensive success determination logic"""
        relevance_score = metrics.get('final_relevance_score', 0)
        sources = sources or []
        
        # Multiple success criteria
        criteria = {
            'relevance_threshold_met': relevance_score >= self.relevance_threshold,
            'analysis_content_available': self._has_substantial_analysis(analysis),
            'sources_processed': len(sources) >= self.min_sources_threshold,
            'statistical_data_found': self._has_statistical_data(analysis),
            'parsing_successful': self._was_parsing_successful(analysis),
            'content_quality_adequate': self._has_quality_content(analysis)
        }
        
        # Calculate success score (0.0 to 1.0)
        success_score = sum(criteria.values()) / len(criteria)
        
        # Determine overall success level
        success_level = self._classify_success_level(success_score)
        overall_success = success_score >= 0.75
        
        return {
            'overall_success': overall_success,
            'success_level': success_level,
            'success_score': success_score,
            'criteria_met': criteria,
            'success_reasoning': self._explain_success_determination(criteria, relevance_score, len(sources)),
            'recommendations': self._generate_recommendations(criteria, relevance_score, len(sources))
        }
    
    def _has_substantial_analysis(self, analysis: dict) -> bool:
        """Check if analysis has substantial content"""
        if not analysis:
            return False
        
        content = analysis.get('analysis_content', '') or analysis.get('comprehensive_answer', '')
        return len(content.strip()) >= self.min_content_length
    
    def _has_statistical_data(self, analysis: dict) -> bool:
        """Check if analysis contains statistical data"""
        if not analysis:
            return False
        
        content = (analysis.get('analysis_content', '') or analysis.get('comprehensive_answer', '')).lower()
        
        statistical_indicators = [
            'percent', '%', 'statistics', 'data shows', 'according to',
            'revenue', 'market share', 'users', 'million', 'billion',
            'growth', 'increase', 'decrease', 'ratio', 'rate'
        ]
        
        return any(indicator in content for indicator in statistical_indicators)
    
    def _was_parsing_successful(self, analysis: dict) -> bool:
        """Check if response parsing was successful"""
        if not analysis:
            return False
            
        parsing_method = analysis.get('parsing_method', 'unknown')
        return parsing_method in ['json_parsed', 'text_extraction'] and parsing_method != 'fallback'
    
    def _has_quality_content(self, analysis: dict) -> bool:
        """Check if content meets quality standards"""
        if not analysis:
            return False
        
        content = analysis.get('analysis_content', '') or analysis.get('comprehensive_answer', '')
        
        # Check for quality indicators
        quality_indicators = [
            'comprehensive', 'analysis', 'findings', 'research',
            'evidence', 'data', 'results', 'conclusion'
        ]
        
        indicator_count = sum(1 for indicator in quality_indicators 
                             if indicator.lower() in content.lower())
        
        # Quality based on length and indicator presence
        return len(content) >= 300 and indicator_count >= 3
    
    def _classify_success_level(self, score: float) -> str:
        """Classify success level based on score"""
        if score >= 0.9:
            return 'excellent'
        elif score >= 0.75:
            return 'full'
        elif score >= 0.5:
            return 'partial'
        else:
            return 'failed'
    
    def _explain_success_determination(self, criteria: dict, relevance_score: int, source_count: int) -> str:
        """Provide explanation for success determination"""
        met_criteria = [key for key, value in criteria.items() if value]
        failed_criteria = [key for key, value in criteria.items() if not value]
        
        explanation = []
        explanation.append(f"Research evaluation based on {len(criteria)} criteria:")
        explanation.append(f"✅ Met: {len(met_criteria)}/{len(criteria)} criteria")
        
        if met_criteria:
            explanation.append(f"Successful areas: {', '.join(met_criteria[:3])}")
        
        if failed_criteria:
            explanation.append(f"Areas needing improvement: {', '.join(failed_criteria[:3])}")
        
        explanation.append(f"Relevance score: {relevance_score}/10, Sources: {source_count}")
        
        return " | ".join(explanation)
    
    def _generate_recommendations(self, criteria: dict, relevance_score: int, source_count: int) -> list:
        """Generate recommendations for improvement"""
        recommendations = []
        
        if not criteria.get('relevance_threshold_met'):
            recommendations.append(f"Improve relevance score (current: {relevance_score}, target: {self.relevance_threshold}+)")
        
        if not criteria.get('sources_processed'):
            recommendations.append(f"Increase source count (current: {source_count}, target: {self.min_sources_threshold}+)")
        
        if not criteria.get('analysis_content_available'):
            recommendations.append("Ensure comprehensive analysis generation")
        
        if not criteria.get('statistical_data_found'):
            recommendations.append("Include more statistical data and quantitative findings")
        
        if not criteria.get('parsing_successful'):
            recommendations.append("Improve API response parsing reliability")
        
        if not criteria.get('content_quality_adequate'):
            recommendations.append("Enhance analysis content quality and depth")
        
        return recommendations

@dataclass
class RelevanceEvaluation:
    """関連性評価結果を格納するデータクラス"""
    content_id: str
    url: str
    relevance_score: float  # 0-10スケール
    evaluation_reason: str
    meets_threshold: bool
    evaluation_time: datetime
    token_usage: int

class RelevanceEvaluator:
    """関連性評価システム - 70%閾値による高精度な関連性判定"""
    
    def __init__(self, api_client: AsyncOpenAI, threshold: float = 0.7):
        """
        初期化処理
        
        Args:
            api_client: DeepSeek API クライアント
            threshold: 関連性閾値 (0.7 = 70%)
        """
        self.api_client = api_client
        self.threshold = threshold
        self.evaluation_cache = {}  # URL -> RelevanceEvaluation のキャッシュ
        self.logger = logging.getLogger(__name__)
        
        # 関連性評価用プロンプトテンプレート
        self.evaluation_prompt_template = """
質問: {question}

コンテンツ:
{content}

情報源URL: {url}

以下の指示に従って、このコンテンツの関連性を評価してください：

1. **関連性スコア**: 質問に対するこのコンテンツの関連性を0-10のスケールで評価
   - 0-3: 関連性が低い (質問とは無関係または間接的)
   - 4-6: 中程度の関連性 (一部関連するが不完全)
   - 7-10: 高い関連性 (質問に直接的に回答している)

2. **評価理由**: スコアの根拠を具体的に説明

以下の形式で回答してください：
RELEVANCE_SCORE: [0-10の数値]
REASON: [評価理由の詳細説明]
"""
    
    async def evaluate_relevance(self, question: str, content: str, url: str) -> RelevanceEvaluation:
        """
        単一コンテンツの関連性を0-10スケールで評価
        
        Args:
            question: 元の質問
            content: 評価対象のコンテンツ
            url: コンテンツの情報源URL
            
        Returns:
            RelevanceEvaluation: 評価結果
        """
        # キャッシュチェック
        cache_key = f"{hash(question)}_{hash(content)}_{url}"
        if cache_key in self.evaluation_cache:
            self.logger.info(f"✅ 関連性評価キャッシュヒット: {url}")
            return self.evaluation_cache[cache_key]
        
        start_time = time.time()
        content_id = f"content_{hash(content)}"
        
        try:
            # プロンプト生成
            prompt = self.evaluation_prompt_template.format(
                question=question,
                content=content[:2000],  # コンテンツを制限してトークン数を抑制
                url=url
            )
            
            # DeepSeek APIで関連性評価
            response = await self.api_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "あなたは情報の関連性を正確に評価する専門家です。指定された形式で評価を行ってください。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1  # 一貫性のある評価のため低温度設定
            )
            
            evaluation_text = response.choices[0].message.content
            token_usage = response.usage.total_tokens if response.usage else 0
            
            # 評価結果をパース
            score, reason = self._parse_evaluation_response(evaluation_text)
            meets_threshold = score >= (self.threshold * 10)
            
            # 評価結果オブジェクト作成
            evaluation = RelevanceEvaluation(
                content_id=content_id,
                url=url,
                relevance_score=score,
                evaluation_reason=reason,
                meets_threshold=meets_threshold,
                evaluation_time=datetime.now(),
                token_usage=token_usage
            )
            
            # キャッシュに保存
            self.evaluation_cache[cache_key] = evaluation
            
            # ログ記録
            processing_time = time.time() - start_time
            self.logger.info(f"🎯 関連性評価完了: {url}")
            self.logger.info(f"   スコア: {score}/10 {'✅' if meets_threshold else '❌'}")
            self.logger.info(f"   処理時間: {processing_time:.2f}秒")
            self.logger.info(f"   理由: {reason[:100]}...")
            
            return evaluation
            
        except Exception as e:
            self.logger.error(f"❌ 関連性評価エラー: {url} - {e}")
            # エラー時のフォールバック評価
            return RelevanceEvaluation(
                content_id=content_id,
                url=url,
                relevance_score=5.0,  # 中間スコア
                evaluation_reason=f"評価エラーによりデフォルトスコア設定: {str(e)}",
                meets_threshold=False,
                evaluation_time=datetime.now(),
                token_usage=0
            )
    
    def _parse_evaluation_response(self, response_text: str) -> Tuple[float, str]:
        """
        DeepSeek APIの応答から関連性スコアと理由を抽出
        
        Args:
            response_text: API応答テキスト
            
        Returns:
            Tuple[float, str]: (関連性スコア, 評価理由)
        """
        try:
            # スコア抽出
            score_match = re.search(r'RELEVANCE_SCORE:\s*([0-9.]+)', response_text)
            if score_match:
                score = float(score_match.group(1))
                score = max(0.0, min(10.0, score))  # 0-10の範囲に制限
            else:
                # パターンマッチング失敗時のフォールバック
                score = self._extract_score_fallback(response_text)
            
            # 理由抽出
            reason_match = re.search(r'REASON:\s*(.+)', response_text, re.DOTALL)
            if reason_match:
                reason = reason_match.group(1).strip()
            else:
                reason = "評価理由の抽出に失敗しました"
            
            return score, reason
            
        except Exception as e:
            self.logger.warning(f"⚠️ 評価応答パースエラー: {e}")
            return 5.0, f"パースエラー: {str(e)}"
    
    def _extract_score_fallback(self, text: str) -> float:
        """
        フォールバック用のスコア抽出ロジック
        
        Args:
            text: 応答テキスト
            
        Returns:
            float: 推定関連性スコア
        """
        # 数値パターンを探索
        numbers = re.findall(r'\b([0-9]|10)\b', text)
        if numbers:
            # 最初に見つかった0-10の数値を使用
            potential_score = float(numbers[0])
            if 0 <= potential_score <= 10:
                return potential_score
        
        # 品質指標に基づく推定
        quality_indicators = [
            'excellent', 'perfect', 'highly relevant', '非常に関連',
            'good', 'relevant', '関連', 'appropriate',
            'poor', 'irrelevant', '無関係', 'unrelated'
        ]
        
        text_lower = text.lower()
        if any(indicator in text_lower for indicator in quality_indicators[:4]):
            return 8.0  # 高評価
        elif any(indicator in text_lower for indicator in quality_indicators[4:8]):
            return 6.0  # 中評価
        else:
            return 3.0  # 低評価
    
    async def batch_evaluate(self, question: str, contents: List[Dict]) -> List[RelevanceEvaluation]:
        """
        複数コンテンツの一括関連性評価
        
        Args:
            question: 元の質問
            contents: 評価対象のコンテンツリスト (各要素は url, content, title を含む辞書)
            
        Returns:
            List[RelevanceEvaluation]: 評価結果リスト
        """
        self.logger.info(f"🔄 バッチ関連性評価開始: {len(contents)}件")
        
        # 並列処理で評価を実行（最大5件同時処理）
        semaphore = asyncio.Semaphore(5)
        
        async def evaluate_single(content_dict):
            async with semaphore:
                return await self.evaluate_relevance(
                    question=question,
                    content=content_dict.get('content', ''),
                    url=content_dict.get('url', 'unknown')
                )
        
        # 全ての評価を並列実行
        evaluations = await asyncio.gather(
            *[evaluate_single(content) for content in contents],
            return_exceptions=True
        )
        
        # エラーハンドリング
        valid_evaluations = []
        for i, evaluation in enumerate(evaluations):
            if isinstance(evaluation, Exception):
                self.logger.error(f"❌ バッチ評価エラー ({i}): {evaluation}")
                # エラー時のデフォルト評価
                valid_evaluations.append(RelevanceEvaluation(
                    content_id=f"error_content_{i}",
                    url=contents[i].get('url', 'unknown'),
                    relevance_score=0.0,
                    evaluation_reason=f"評価エラー: {str(evaluation)}",
                    meets_threshold=False,
                    evaluation_time=datetime.now(),
                    token_usage=0
                ))
            else:
                valid_evaluations.append(evaluation)
        
        # 結果サマリー
        high_relevance_count = sum(1 for eval in valid_evaluations if eval.meets_threshold)
        avg_score = sum(eval.relevance_score for eval in valid_evaluations) / len(valid_evaluations)
        
        self.logger.info(f"✅ バッチ評価完了:")
        self.logger.info(f"   総数: {len(valid_evaluations)}件")
        self.logger.info(f"   高関連性（70%以上）: {high_relevance_count}件")
        self.logger.info(f"   平均スコア: {avg_score:.2f}/10")
        
        return valid_evaluations
    
    def meets_threshold(self, score: float) -> bool:
        """
        70%閾値チェック
        
        Args:
            score: 関連性スコア (0-10)
            
        Returns:
            bool: 閾値を満たすかどうか
        """
        return score >= (self.threshold * 10)
    
    def filter_high_relevance(self, evaluations: List[RelevanceEvaluation]) -> List[RelevanceEvaluation]:
        """
        70%以上の高関連性コンテンツのみをフィルタリング
        
        Args:
            evaluations: 評価結果リスト
            
        Returns:
            List[RelevanceEvaluation]: 高関連性評価結果リスト
        """
        high_relevance = [eval for eval in evaluations if eval.meets_threshold]
        
        self.logger.info(f"🎯 高関連性フィルタリング結果:")
        self.logger.info(f"   入力: {len(evaluations)}件")
        self.logger.info(f"   出力: {len(high_relevance)}件 (70%以上)")
        
        return high_relevance
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """
        評価統計情報を取得
        
        Returns:
            Dict[str, Any]: 統計情報
        """
        if not self.evaluation_cache:
            return {"cache_size": 0, "stats": "評価履歴なし"}
        
        evaluations = list(self.evaluation_cache.values())
        scores = [eval.relevance_score for eval in evaluations]
        
        return {
            "cache_size": len(self.evaluation_cache),
            "total_evaluations": len(evaluations),
            "average_score": sum(scores) / len(scores),
            "high_relevance_count": sum(1 for eval in evaluations if eval.meets_threshold),
            "total_tokens_used": sum(eval.token_usage for eval in evaluations)
        }

@dataclass
class AggregatedAnswer:
    """集計済み回答を格納するデータクラス"""
    content: str
    relevance_score: float
    source_urls: List[str]
    confidence_level: str
    extraction_time: datetime
    is_deduplicated: bool
    rank: int = 0

class AnswerAggregator:
    """高関連性回答の自動集計・ランキングシステム"""
    
    def __init__(self, deduplication_threshold: float = 0.8):
        """
        初期化処理
        
        Args:
            deduplication_threshold: 重複判定閾値 (0.8 = 80%の類似度で重複とみなす)
        """
        self.deduplication_threshold = deduplication_threshold
        self.aggregated_answers = []
        self.logger = logging.getLogger(__name__)
    
    def aggregate_answers(self, evaluations: List[RelevanceEvaluation]) -> List[AggregatedAnswer]:
        """
        70%以上の関連性を持つ回答を集計しランキング
        
        Args:
            evaluations: 関連性評価結果リスト
            
        Returns:
            List[AggregatedAnswer]: 集計・ランキング済み回答リスト
        """
        self.logger.info(f"🔄 回答集計開始: {len(evaluations)}件")
        
        # 70%以上の高関連性評価のみを抽出
        high_relevance_evaluations = [eval for eval in evaluations if eval.meets_threshold]
        
        if not high_relevance_evaluations:
            self.logger.warning("❌ 70%以上の関連性を持つ回答がありません")
            return []
        
        # 評価結果をAggregatedAnswerに変換
        candidates = []
        for eval in high_relevance_evaluations:
            # コンテンツを取得（evaluationからは直接取得できないため、URLから推定）
            content = f"高関連性コンテンツ (スコア: {eval.relevance_score}/10)"  # 実際のコンテンツは別途取得が必要
            
            aggregated = AggregatedAnswer(
                content=content,
                relevance_score=eval.relevance_score,
                source_urls=[eval.url],
                confidence_level=self._calculate_confidence_level(eval.relevance_score),
                extraction_time=eval.evaluation_time,
                is_deduplicated=False
            )
            candidates.append(aggregated)
        
        # 重複除去
        deduplicated_answers = self.deduplicate_content(candidates)
        
        # 関連性スコア順にランキング
        ranked_answers = self.rank_by_relevance(deduplicated_answers)
        
        self.aggregated_answers = ranked_answers
        
        self.logger.info(f"✅ 回答集計完了:")
        self.logger.info(f"   入力: {len(evaluations)}件")
        self.logger.info(f"   高関連性: {len(high_relevance_evaluations)}件")
        self.logger.info(f"   重複除去後: {len(deduplicated_answers)}件")
        self.logger.info(f"   最終ランキング: {len(ranked_answers)}件")
        
        return ranked_answers
    
    def deduplicate_content(self, answers: List[AggregatedAnswer]) -> List[AggregatedAnswer]:
        """
        重複コンテンツの除去
        
        Args:
            answers: 集計対象の回答リスト
            
        Returns:
            List[AggregatedAnswer]: 重複除去済み回答リスト
        """
        if len(answers) <= 1:
            return answers
        
        deduplicated = []
        processed_indices = set()
        
        for i, answer in enumerate(answers):
            if i in processed_indices:
                continue
            
            # 同じ回答と類似する回答を検索
            similar_answers = [answer]
            similar_indices = {i}
            
            for j, other_answer in enumerate(answers[i+1:], i+1):
                if j in processed_indices:
                    continue
                
                # 類似度計算（簡易的な実装）
                similarity = self._calculate_similarity(answer, other_answer)
                
                if similarity >= self.deduplication_threshold:
                    similar_answers.append(other_answer)
                    similar_indices.add(j)
            
            # 類似回答をマージして最高スコアの回答を採用
            merged_answer = self._merge_similar_answers(similar_answers)
            merged_answer.is_deduplicated = len(similar_answers) > 1
            
            deduplicated.append(merged_answer)
            processed_indices.update(similar_indices)
        
        self.logger.info(f"🎯 重複除去: {len(answers)}件 → {len(deduplicated)}件")
        
        return deduplicated
    
    def _calculate_similarity(self, answer1: AggregatedAnswer, answer2: AggregatedAnswer) -> float:
        """
        2つの回答の類似度を計算
        
        Args:
            answer1, answer2: 比較対象の回答
            
        Returns:
            float: 類似度 (0.0-1.0)
        """
        # URL重複チェック
        common_urls = set(answer1.source_urls) & set(answer2.source_urls)
        if common_urls:
            return 1.0  # 同じソースURLがあれば100%類似とみなす
        
        # コンテンツ類似度（簡易実装）
        words1 = set(answer1.content.lower().split())
        words2 = set(answer2.content.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _merge_similar_answers(self, similar_answers: List[AggregatedAnswer]) -> AggregatedAnswer:
        """
        類似回答をマージして最適な回答を生成
        
        Args:
            similar_answers: マージ対象の類似回答リスト
            
        Returns:
            AggregatedAnswer: マージ済み回答
        """
        # 最高スコアの回答をベースにする
        best_answer = max(similar_answers, key=lambda x: x.relevance_score)
        
        # 情報源URLを統合
        all_urls = []
        for answer in similar_answers:
            all_urls.extend(answer.source_urls)
        unique_urls = list(set(all_urls))
        
        # マージされた回答を作成
        merged = AggregatedAnswer(
            content=best_answer.content,
            relevance_score=best_answer.relevance_score,
            source_urls=unique_urls,
            confidence_level=self._calculate_confidence_level(best_answer.relevance_score),
            extraction_time=best_answer.extraction_time,
            is_deduplicated=True
        )
        
        return merged
    
    def rank_by_relevance(self, answers: List[AggregatedAnswer]) -> List[AggregatedAnswer]:
        """
        関連性スコア順にランキング
        
        Args:
            answers: ランキング対象の回答リスト
            
        Returns:
            List[AggregatedAnswer]: ランキング済み回答リスト
        """
        # 関連性スコア順でソート（降順）
        sorted_answers = sorted(answers, key=lambda x: x.relevance_score, reverse=True)
        
        # ランク番号を設定
        for i, answer in enumerate(sorted_answers, 1):
            answer.rank = i
        
        self.logger.info(f"📊 ランキング完了: 1位スコア {sorted_answers[0].relevance_score}/10" if sorted_answers else "📊 ランキング対象なし")
        
        return sorted_answers
    
    def _calculate_confidence_level(self, relevance_score: float) -> str:
        """
        関連性スコアに基づく信頼性レベルの計算
        
        Args:
            relevance_score: 関連性スコア (0-10)
            
        Returns:
            str: 信頼性レベル
        """
        if relevance_score >= 9.0:
            return "非常に高い"
        elif relevance_score >= 8.0:
            return "高い"
        elif relevance_score >= 7.0:
            return "中程度"
        else:
            return "低い"
    
    def get_top_answer(self) -> Optional[AggregatedAnswer]:
        """
        最も関連性の高い回答を取得
        
        Returns:
            Optional[AggregatedAnswer]: 最高ランクの回答（存在しない場合はNone）
        """
        if not self.aggregated_answers:
            return None
        
        return self.aggregated_answers[0]  # 既にランキング済みなので最初の要素が最高ランク
    
    def get_aggregation_stats(self) -> Dict[str, Any]:
        """
        集計統計情報を取得
        
        Returns:
            Dict[str, Any]: 統計情報
        """
        if not self.aggregated_answers:
            return {"total_answers": 0, "stats": "集計結果なし"}
        
        scores = [answer.relevance_score for answer in self.aggregated_answers]
        deduplicated_count = sum(1 for answer in self.aggregated_answers if answer.is_deduplicated)
        
        return {
            "total_answers": len(self.aggregated_answers),
            "average_score": sum(scores) / len(scores),
            "top_score": max(scores),
            "deduplicated_count": deduplicated_count,
            "confidence_distribution": {
                level: sum(1 for answer in self.aggregated_answers if answer.confidence_level == level)
                for level in ["非常に高い", "高い", "中程度", "低い"]
            }
        }

@dataclass
class SummaryResult:
    """要約結果を格納するデータクラス"""
    original_question: str
    summary_text: str
    relevance_score: float
    source_urls: List[str]
    confidence_metrics: Dict[str, float]
    generation_time: datetime
    token_usage: int

class SummaryGenerator:
    """最高関連性回答の要約生成システム"""
    
    def __init__(self, api_client: AsyncOpenAI, token_handler):
        """
        初期化処理
        
        Args:
            api_client: DeepSeek API クライアント
            token_handler: トークン制限ハンドラー
        """
        self.api_client = api_client
        self.token_handler = token_handler
        self.logger = logging.getLogger(__name__)
        
        # 要約生成用プロンプトテンプレート
        self.summary_prompt_template = """
質問: {question}

最高関連性コンテンツ（スコア: {relevance_score}/10）:
{content}

情報源URL: {source_urls}

以下の指示に従って、この情報の要約を作成してください：

1. **要約内容**: 元の質問に対する直接的で簡潔な回答
2. **関連性**: 元の質問との関連性を明確に示す
3. **信頼性**: 情報源の信頼性について言及
4. **構造**: 読みやすく整理された形で提示

要約は以下の形式で作成してください：
【要約】
[元の質問に対する直接的な回答を3-5文で簡潔に記述]

【関連性】
[この情報が元の質問にどのように関連するかを説明]

【情報源】
[情報源URLと信頼性についてのコメント]
"""
    
    async def generate_summary(self, question: str, best_answer: AggregatedAnswer) -> SummaryResult:
        """
        最高関連性回答の要約生成
        
        Args:
            question: 元の質問
            best_answer: 最高関連性回答
            
        Returns:
            SummaryResult: 要約結果
        """
        start_time = time.time()
        
        self.logger.info(f"📝 要約生成開始: スコア {best_answer.relevance_score}/10")
        
        try:
            # プロンプト生成
            prompt = self.create_summary_prompt(question, best_answer)
            
            # トークン制限チェック
            if self.token_handler:
                prompt = self.token_handler.truncate_content_if_needed(prompt)
            
            # DeepSeek APIで要約生成
            response = await self.api_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "あなたは情報要約の専門家です。与えられた情報を元の質問に対する簡潔で正確な要約を作成してください。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3  # 一貫性重視
            )
            
            summary_text = response.choices[0].message.content
            token_usage = response.usage.total_tokens if response.usage else 0
            
            # 要約品質検証
            validation_result = await self.validate_summary(summary_text, question)
            
            # 信頼性メトリクス計算
            confidence_metrics = self._calculate_confidence_metrics(
                best_answer.relevance_score,
                len(best_answer.source_urls),
                validation_result
            )
            
            # 要約結果オブジェクト作成
            summary_result = SummaryResult(
                original_question=question,
                summary_text=summary_text,
                relevance_score=best_answer.relevance_score,
                source_urls=best_answer.source_urls,
                confidence_metrics=confidence_metrics,
                generation_time=datetime.now(),
                token_usage=token_usage
            )
            
            processing_time = time.time() - start_time
            
            self.logger.info(f"✅ 要約生成完了:")
            self.logger.info(f"   処理時間: {processing_time:.2f}秒")
            self.logger.info(f"   トークン使用: {token_usage}")
            self.logger.info(f"   信頼性: {confidence_metrics.get('overall_confidence', 0):.2f}")
            
            return summary_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ 要約生成エラー: {e}")
            
            # エラー時のフォールバック要約
            return SummaryResult(
                original_question=question,
                summary_text=f"要約生成に失敗しました。エラー: {str(e)}\n\n元の回答（スコア: {best_answer.relevance_score}/10）を参照してください。",
                relevance_score=best_answer.relevance_score,
                source_urls=best_answer.source_urls,
                confidence_metrics={"overall_confidence": 0.0, "error": True},
                generation_time=datetime.now(),
                token_usage=0
            )
    
    def create_summary_prompt(self, question: str, best_answer: AggregatedAnswer) -> str:
        """
        要約生成用プロンプト作成
        
        Args:
            question: 元の質問
            best_answer: 最高関連性回答
            
        Returns:
            str: 生成されたプロンプト
        """
        # 情報源URLを整形
        source_urls_text = "\n".join([f"- {url}" for url in best_answer.source_urls[:3]])  # 最大3つまで
        
        return self.summary_prompt_template.format(
            question=question,
            relevance_score=best_answer.relevance_score,
            content=best_answer.content[:1500],  # コンテンツを制限
            source_urls=source_urls_text
        )
    
    async def validate_summary(self, summary: str, original_question: str) -> bool:
        """
        要約品質検証
        
        Args:
            summary: 生成された要約
            original_question: 元の質問
            
        Returns:
            bool: 要約が適切かどうか
        """
        try:
            # 基本的な品質チェック
            if len(summary.strip()) < 50:
                return False
            
            # 必要なセクションが含まれているかチェック
            required_sections = ["【要約】", "【関連性】", "【情報源】"]
            missing_sections = [section for section in required_sections if section not in summary]
            
            if len(missing_sections) > 1:  # 1つまでの欠落は許容
                self.logger.warning(f"⚠️ 要約に不足セクション: {missing_sections}")
                return False
            
            # 元の質問に関連するキーワードが含まれているかチェック
            question_keywords = set(original_question.lower().split())
            summary_keywords = set(summary.lower().split())
            
            overlap = len(question_keywords & summary_keywords)
            relevance_ratio = overlap / len(question_keywords) if question_keywords else 0
            
            return relevance_ratio >= 0.3  # 30%以上のキーワード重複で関連性ありとみなす
            
        except Exception as e:
            self.logger.error(f"❌ 要約検証エラー: {e}")
            return False
    
    def _calculate_confidence_metrics(self, relevance_score: float, source_count: int, validation_result: bool) -> Dict[str, float]:
        """
        信頼性メトリクス計算
        
        Args:
            relevance_score: 関連性スコア
            source_count: 情報源数
            validation_result: 要約検証結果
            
        Returns:
            Dict[str, float]: 信頼性メトリクス
        """
        # 関連性信頼度 (0-1)
        relevance_confidence = min(1.0, relevance_score / 10.0)
        
        # 情報源信頼度 (0-1)
        source_confidence = min(1.0, source_count / 3.0)  # 3つ以上の情報源で最高評価
        
        # 検証信頼度 (0-1)
        validation_confidence = 1.0 if validation_result else 0.5
        
        # 総合信頼度
        overall_confidence = (
            relevance_confidence * 0.5 +
            source_confidence * 0.3 +
            validation_confidence * 0.2
        )
        
        return {
            "relevance_confidence": relevance_confidence,
            "source_confidence": source_confidence,
            "validation_confidence": validation_confidence,
            "overall_confidence": overall_confidence,
            "quality_grade": self._get_quality_grade(overall_confidence)
        }
    
    def _get_quality_grade(self, confidence: float) -> str:
        """
        信頼度に基づく品質グレード
        
        Args:
            confidence: 総合信頼度
            
        Returns:
            str: 品質グレード
        """
        if confidence >= 0.8:
            return "A (優秀)"
        elif confidence >= 0.6:
            return "B (良好)"
        elif confidence >= 0.4:
            return "C (普通)"
        else:
            return "D (要改善)"

@dataclass
class FormattedResult:
    """表示用にフォーマットされた結果"""
    summary_text: str
    metadata_table: str
    confidence_display: str
    source_list: str
    error_message: Optional[str] = None
    fallback_data: Optional[str] = None

class ResultFormatter:
    """要約結果の構造化表示を担当するクラス"""
    
    def __init__(self):
        """ResultFormatterの初期化"""
        self.display_templates = {
            'summary': self._format_summary_template,
            'table': self._format_table_template,
            'fallback': self._format_fallback_template,
            'error': self._format_error_template
        }
        logger.info("✅ ResultFormatterが初期化されました")
    
    def format_final_result(self, 
                          summary_result: Optional[SummaryResult], 
                          aggregated_answers: List[AggregatedAnswer],
                          original_question: str) -> FormattedResult:
        """
        最終結果の構造化表示を生成
        
        Args:
            summary_result: 生成された要約結果
            aggregated_answers: 集計された回答リスト
            original_question: 元の質問
            
        Returns:
            FormattedResult: フォーマットされた表示結果
        """
        try:
            if summary_result is None:
                # 要約生成に失敗した場合のフォールバック表示
                return self._create_fallback_result(aggregated_answers, original_question)
            
            # 正常な要約結果の表示
            summary_text = self._format_summary_section(summary_result)
            metadata_table = self._format_metadata_table(summary_result)
            confidence_display = self._format_confidence_section(summary_result)
            source_list = self._format_source_section(summary_result)
            
            return FormattedResult(
                summary_text=summary_text,
                metadata_table=metadata_table,
                confidence_display=confidence_display,
                source_list=source_list
            )
            
        except Exception as e:
            logger.error(f"❌ 結果フォーマット処理でエラー: {e}")
            return self._create_error_result(str(e), original_question)
    
    def _format_summary_template(self, summary_result: SummaryResult) -> str:
        """要約用テンプレートの適用"""
        return f"""
【質問】
{summary_result.original_question}

【要約回答】
{summary_result.summary_text}

【関連性スコア】
{summary_result.relevance_score:.1f}/10.0 ({summary_result.relevance_score * 10:.0f}%)
"""
    
    def _format_table_template(self, summary_result: SummaryResult) -> str:
        """表形式テンプレートの生成"""
        confidence_metrics = summary_result.confidence_metrics
        
        table = """
| 項目 | 値 |
|------|-----|
"""
        
        table += f"| 関連性スコア | {summary_result.relevance_score:.1f}/10.0 |\n"
        table += f"| 信頼性 | {confidence_metrics.get('reliability', 0.0):.2f} |\n"
        table += f"| 完全性 | {confidence_metrics.get('completeness', 0.0):.2f} |\n"
        table += f"| 明確性 | {confidence_metrics.get('clarity', 0.0):.2f} |\n"
        table += f"| 情報源数 | {len(summary_result.source_urls)} |\n"
        table += f"| 生成時刻 | {summary_result.generation_time.strftime('%Y-%m-%d %H:%M:%S')} |\n"
        table += f"| トークン使用量 | {summary_result.token_usage} |\n"
        
        return table
    
    def _format_fallback_template(self, aggregated_answers: List[AggregatedAnswer]) -> str:
        """フォールバック表示テンプレート"""
        if not aggregated_answers:
            return """
【エラー】
要約生成に失敗し、集計された回答も存在しません。
検索条件を見直してください。
"""
        
        fallback_text = """
【要約生成失敗】
以下は集計された生データです：

"""
        
        for i, answer in enumerate(aggregated_answers[:3], 1):  # 上位3件まで表示
            fallback_text += f"""
--- 回答 {i} ---
関連性スコア: {answer.relevance_score:.1f}/10.0
信頼性レベル: {answer.confidence_level}
内容: {answer.content[:500]}{'...' if len(answer.content) > 500 else ''}
情報源: {', '.join(answer.source_urls[:2])}
"""
        
        return fallback_text
    
    def _format_error_template(self, error_message: str, original_question: str) -> str:
        """エラー表示テンプレート"""
        return f"""
【システムエラー】
質問: {original_question}
エラー内容: {error_message}

トラブルシューティング:
1. ネットワーク接続を確認してください
2. API制限に達していないか確認してください
3. 質問を簡単にして再試行してください
"""
    
    def _format_summary_section(self, summary_result: SummaryResult) -> str:
        """要約セクションのフォーマット"""
        return self._format_summary_template(summary_result)
    
    def _format_metadata_table(self, summary_result: SummaryResult) -> str:
        """メタデータテーブルのフォーマット"""
        return self._format_table_template(summary_result)
    
    def _format_confidence_section(self, summary_result: SummaryResult) -> str:
        """信頼性情報セクションのフォーマット"""
        metrics = summary_result.confidence_metrics
        confidence_text = f"""
【信頼性評価】
• 全体評価: {self._calculate_overall_confidence(metrics):.2f}
• 信頼性: {metrics.get('reliability', 0.0):.2f}
• 完全性: {metrics.get('completeness', 0.0):.2f}
• 明確性: {metrics.get('clarity', 0.0):.2f}
"""
        return confidence_text
    
    def _format_source_section(self, summary_result: SummaryResult) -> str:
        """情報源セクションのフォーマット"""
        sources_text = "【情報源】\n"
        
        for i, url in enumerate(summary_result.source_urls, 1):
            sources_text += f"{i}. {url}\n"
        
        return sources_text
    
    def _create_fallback_result(self, 
                              aggregated_answers: List[AggregatedAnswer], 
                              original_question: str) -> FormattedResult:
        """フォールバック結果の生成"""
        fallback_data = self._format_fallback_template(aggregated_answers)
        
        return FormattedResult(
            summary_text="要約生成に失敗しました",
            metadata_table="| 項目 | 値 |\n|------|-----|\n| ステータス | フォールバック表示 |",
            confidence_display="【信頼性評価】\n評価不可（要約生成失敗）",
            source_list="【情報源】\n利用可能な情報源なし",
            fallback_data=fallback_data
        )
    
    def _create_error_result(self, error_message: str, original_question: str) -> FormattedResult:
        """エラー結果の生成"""
        error_text = self._format_error_template(error_message, original_question)
        
        return FormattedResult(
            summary_text="システムエラーが発生しました",
            metadata_table="| 項目 | 値 |\n|------|-----|\n| ステータス | エラー |",
            confidence_display="【信頼性評価】\n評価不可（システムエラー）",
            source_list="【情報源】\n利用不可",
            error_message=error_text
        )
    
    def _calculate_overall_confidence(self, metrics: Dict[str, float]) -> float:
        """全体的な信頼性スコアの計算"""
        if not metrics:
            return 0.0
        
        weights = {
            'reliability': 0.4,
            'completeness': 0.3,
            'clarity': 0.3
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for key, weight in weights.items():
            if key in metrics:
                total_score += metrics[key] * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def display_formatted_result(self, formatted_result: FormattedResult) -> None:
        """フォーマットされた結果の表示"""
        print("\n" + "="*80)
        print("📋 研究結果サマリー")
        print("="*80)
        
        if formatted_result.error_message:
            print(formatted_result.error_message)
            return
        
        print(formatted_result.summary_text)
        
        print("\n" + "-"*60)
        print("📊 詳細メタデータ")
        print("-"*60)
        print(formatted_result.metadata_table)
        
        print("\n" + "-"*60)
        print(formatted_result.confidence_display)
        
        print("\n" + "-"*60)
        print(formatted_result.source_list)
        
        if formatted_result.fallback_data:
            print("\n" + "-"*60)
            print("📄 代替情報")
            print("-"*60)
            print(formatted_result.fallback_data)
        
        print("="*80)

class EnhancedDeepSeekResearchService:
    """Enhanced research service with optimization components, time management, and token optimization"""
    
    def __init__(self):
        self.api_key = os.environ.get('DEEPSEEK_API_KEY', '')
        self.api_url = os.environ.get('DEEPSEEK_API_URL', 'https://api.deepseek.com')
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.api_url)
        self.web_search = EnhancedGoogleWebSearchService()
        self.cache_service = MongoDBCacheService()
        self.content_extractor = BrightDataContentExtractor(self.cache_service)
        self.metrics = SearchMetrics()
        self.answer_tracker = None  # Will be initialized in research
        
        # New optimization components
        self.time_manager = TimeManager(max_duration=600)  # 10 minutes
        self.token_optimizer = TokenOptimizer(max_tokens=50000)
        self.content_prioritizer = ContentPrioritizer()
        self.progressive_response = ProgressiveResponseGenerator()
        self.token_handler = TokenLimitHandler(self.token_optimizer)
        self.time_handler = TimeConstraintHandler(self.time_manager)
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=300)
        self.score_calculator = EnhancedScoreCalculator()  # Enhanced score calculation
        self.response_parser = RobustAPIResponseParser()  # Enhanced response parsing
        self.display_formatter = EnhancedDisplayFormatter()  # Enhanced display formatting
        self.success_validator = SuccessValidator()  # Enhanced success determination
        
        # New relevance evaluation components
        self.relevance_evaluator = RelevanceEvaluator(self.client)  # 関連性評価システム
        self.answer_aggregator = AnswerAggregator()  # 回答集計システム
        self.summary_generator = SummaryGenerator(self.client, self.token_handler)  # 要約生成システム
        self.result_formatter = ResultFormatter()  # 結果表示システム
        
        # Research session tracking
        self.current_session = None
        
        if not self.api_key:
            logger.error("❌ DEEPSEEK_API_KEY not set")
            raise ValueError("DEEPSEEK_API_KEY environment variable is required")
    
    async def initialize(self):
        """Initialize MongoDB connection"""
        await self.cache_service.connect()
        
        # Display cache stats
        cache_stats = await self.cache_service.get_cache_stats()
        logger.info(f"📊 Cache stats: {cache_stats['total_entries']} total entries, {cache_stats.get('fresh_entries', 0)} fresh")
    
    async def search_existing_cache(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Search for existing cached content based on queries"""
        logger.info(f"🔍 Searching cache for {len(queries)} queries")
        
        # Extract keywords from all queries
        all_keywords = []
        for query in queries:
            # Simple keyword extraction - split and clean
            keywords = [word.strip().lower() for word in re.split(r'[^\w]+', query) if len(word.strip()) > 2]
            all_keywords.extend(keywords)
        
        # Remove duplicates while preserving order
        unique_keywords = list(dict.fromkeys(all_keywords))
        
        cached_results = await self.cache_service.search_cached_content(unique_keywords)
        
        if cached_results:
            logger.info(f"💾 Found {len(cached_results)} cached results")
            self.metrics.cache_hits += len(cached_results)
            
            # Convert cached results to extraction format
            extracted_contents = []
            for cached in cached_results:
                content = {
                    'url': cached['url'],
                    'title': cached.get('title', 'No title'),
                    'content': cached.get('content', ''),
                    'method': f"cache_{cached.get('method', 'unknown')}",
                    'word_count': cached.get('word_count', 0),
                    'extraction_time': 0.0,  # Instant from cache
                    'domain_info': cached.get('domain_info', {}),
                    'success': True,
                    'from_cache': True,
                    'cache_date': cached.get('accessed_date'),
                    'search_result': {
                        'title': cached.get('title', 'No title'),
                        'link': cached['url'],
                        'snippet': cached.get('content', '')[:200] + '...',
                        'displayLink': urlparse(cached['url']).netloc,
                        'search_query': 'cached_result'
                    }
                }
                extracted_contents.append(content)
            
            return extracted_contents
        else:
            logger.info("📭 No cached results found")
            return []
    
    async def generate_multi_angle_queries(self, original_question: str) -> List[str]:
        """Generate multiple search queries from different angles"""
        start_time = time.time()
        
        try:
            logger.info("🎯 Generating multi-angle search queries")
            
            system_message = """You are an expert research strategist. Generate 4-5 different search queries to comprehensively research a topic from multiple angles.

For business research questions, consider these angles:
1. Company/product names and direct information
2. Market analysis and industry reports  
3. Financial data and revenue information
4. Competitive analysis and rankings
5. Regional/geographic specific information

Instructions:
1. Analyze the question to identify key aspects
2. Generate 4-5 distinct search queries covering different angles
3. Make queries specific and targeted
4. Include relevant industry terms and modifiers
5. Format response as: Query1="...", Query2="...", Query3="...", etc.
6. Ensure queries complement each other without too much overlap"""

            user_prompt = f"""Original Question: {original_question}

Generate 4-5 comprehensive search queries that approach this question from different angles. Focus on finding authoritative, data-rich sources."""

            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                stream=False,
                timeout=30.0
            )
            
            response_text = response.choices[0].message.content
            generation_time = time.time() - start_time
            
            # Extract queries from response
            queries = []
            query_pattern = r'Query\d*="([^"]+)"'
            matches = re.findall(query_pattern, response_text)
            
            if matches:
                queries = matches
                logger.info(f"✅ Generated {len(queries)} search queries in {generation_time:.2f}s")
                for i, query in enumerate(queries, 1):
                    logger.info(f"  {i}. {query}")
            else:
                # Fallback: use original question
                queries = [original_question]
                logger.warning("⚠️ Could not extract queries, using original question")
            
            self.metrics.total_queries += len(queries)
            return queries
                
        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(f"❌ Query generation failed in {generation_time:.2f}s: {e}")
            return [original_question]
    
    async def perform_comprehensive_search(self, queries: List[str], max_results_per_query: int = 5) -> List[Dict[str, Any]]:
        """Perform comprehensive search across multiple queries"""
        start_time = time.time()
        all_results = []
        seen_urls = set()
        
        logger.info(f"🔍 Performing comprehensive search with {len(queries)} queries")
        
        # Domains to exclude for business research (low quality sources)
        exclude_domains = [
            'reddit.com', 'quora.com', 'yahoo.com', 'facebook.com', 
            'twitter.com', 'instagram.com', 'tiktok.com'
        ]
        
        for i, query in enumerate(queries, 1):
            logger.info(f"🔍 Search {i}/{len(queries)}: {query}")
            
            query_start = time.time()
            results = await self.web_search.search_with_filters(
                query, 
                num_results=max_results_per_query,
                exclude_domains=exclude_domains
            )
            query_time = time.time() - query_start
            
            # Filter out duplicate URLs
            new_results = []
            for result in results:
                if result['link'] not in seen_urls:
                    seen_urls.add(result['link'])
                    new_results.append(result)
            
            all_results.extend(new_results)
            logger.info(f"  ✅ Query {i} completed: {len(new_results)} new results in {query_time:.2f}s")
            
            # Add delay between searches
            await asyncio.sleep(1)
        
        total_time = time.time() - start_time
        self.metrics.total_results = len(all_results)
        
        logger.info(f"✅ Comprehensive search completed: {len(all_results)} unique results in {total_time:.2f}s")
        return all_results
    
    async def extract_and_analyze_content(self, search_results: List[Dict[str, Any]], queries: List[str]) -> List[Dict[str, Any]]:
        """Extract content with MongoDB caching and quality filtering"""
        start_time = time.time()
        
        logger.info(f"📄 Extracting content from {len(search_results)} sources")
        
        # Extract keywords for caching
        keywords = []
        for query in queries:
            query_keywords = [word.strip().lower() for word in re.split(r'[^\w]+', query) if len(word.strip()) > 2]
            keywords.extend(query_keywords)
        keywords = list(dict.fromkeys(keywords))  # Remove duplicates
        
        # Sort by domain quality for better extraction order
        def get_quality_score(result):
            domain_info = self.content_extractor.assess_domain_quality(result['link'])
            return domain_info['quality_score']
        
        sorted_results = sorted(search_results, key=get_quality_score, reverse=True)
        
        extracted_contents = []
        source_type_counts = defaultdict(int)
        
        for i, result in enumerate(sorted_results):
            logger.info(f"📄 Extracting {i+1}/{len(sorted_results)}: {result['displayLink']}")
            
            content = await self.content_extractor.extract_article_content(result['link'], keywords)
            content['search_result'] = result
            
            # Track cache hits/misses
            if content.get('from_cache', False):
                self.metrics.cache_hits += 1
            else:
                self.metrics.cache_misses += 1
            
            # Track source types for diversification
            source_type = content.get('domain_info', {}).get('source_type', 'unknown')
            source_type_counts[source_type] += 1
            self.metrics.source_types[source_type] += 1
            
            if content['success']:
                self.metrics.successful_extractions += 1
                cache_status = "💾 Cache" if content.get('from_cache') else "🌐 Live"
                logger.info(f"  ✅ Success ({cache_status}): {content['word_count']} words, quality: {content.get('domain_info', {}).get('quality_score', 0)}/10")
            else:
                self.metrics.failed_extractions += 1
                logger.warning(f"  ❌ Failed: {content.get('error', 'Unknown error')}")
            
            extracted_contents.append(content)
            
            # Add delay only for live scraping (not cached content)
            if not content.get('from_cache', False):
                await asyncio.sleep(1)
        
        extraction_time = time.time() - start_time
        logger.info(f"✅ Content extraction completed in {extraction_time:.2f}s")
        logger.info(f"💾 Cache performance: {self.metrics.cache_hits} hits, {self.metrics.cache_misses} misses")
        logger.info(f"📊 Source distribution: {dict(source_type_counts)}")
        
        return extracted_contents
    
    async def analyze_content_with_gaps(self, original_question: str, extracted_contents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhanced analysis with gap identification, relevance scoring, and token limit management (v3.07)"""
        start_time = time.time()
        
        logger.info("🧠 Starting comprehensive content analysis with token optimization")
        
        try:
            # v3.07: Optimize content for token limits
            logger.info(f"📊 Processing {len(extracted_contents)} sources for analysis")
            
            # Sort by quality and limit to target number of sources
            successful_contents = [c for c in extracted_contents if c.get('success', False)]
            
            # Sort by quality score
            def get_quality_score(content):
                return content.get('domain_info', {}).get('quality_score', 5)
            
            successful_contents.sort(key=get_quality_score, reverse=True)
            
            # Limit to optimal number of sources for analysis
            limited_contents = successful_contents[:TARGET_SOURCES_PER_ITERATION]
            
            if len(limited_contents) < len(successful_contents):
                logger.info(f"📉 Limited analysis to top {len(limited_contents)} sources (from {len(successful_contents)}) for token optimization")
            
            # Prepare content with summarization
            content_summaries = []
            total_tokens = 0
            
            for i, content in enumerate(limited_contents):
                domain_info = content.get('domain_info', {})
                quality_score = domain_info.get('quality_score', 5)
                source_type = domain_info.get('source_type', 'unknown')
                cache_status = "CACHED" if content.get('from_cache') else "LIVE"
                
                # Summarize content to fit token limits
                original_content = content['content']
                summarized_content = summarize_content(original_content, MAX_CONTENT_LENGTH)
                
                if len(summarized_content) < len(original_content):
                    self.metrics.content_summarized += 1
                    tokens_saved = count_tokens(original_content) - count_tokens(summarized_content)
                    self.metrics.tokens_saved += tokens_saved
                    logger.info(f"📝 Summarized source {i+1}: {len(original_content)} → {len(summarized_content)} chars, saved ~{tokens_saved} tokens")
                
                summary = f"""
Source {i+1}: {content['title']} [{cache_status}]
URL: {content['url']}
Quality Score: {quality_score}/10
Source Type: {source_type}
Extraction Method: {content['method']}
Content: {summarized_content}
Word Count: {content['word_count']}
"""
                
                # Check total token count
                summary_tokens = count_tokens(summary)
                total_tokens += summary_tokens
                
                if total_tokens > MAX_TOTAL_TOKENS:
                    logger.warning(f"⚠️ Token limit approaching ({total_tokens}/{MAX_TOTAL_TOKENS}), stopping at {len(content_summaries)} sources")
                    break
                
                content_summaries.append(summary)
            
            logger.info(f"📊 Final analysis input: {len(content_summaries)} sources, ~{total_tokens} tokens")
            
            system_message = """You are an expert research analyst with advanced reasoning capabilities. Analyze extracted web content for relevance and identify any gaps in information.

Your Analysis Tasks:
1. RELEVANCE ASSESSMENT: For each source, provide:
   - Individual relevance score (1-10)
   - Key insights that answer the original question
   - Data quality assessment (factual, recent, authoritative)
   - Any limitations or concerns

2. SYNTHESIS: Combine information across sources to:
   - Answer the original question comprehensively
   - Identify patterns and trends
   - Resolve conflicts between sources
   - Highlight the most reliable findings

3. GAP ANALYSIS: Identify what information is missing:
   - Key aspects of the question not fully addressed
   - Data that would strengthen the analysis
   - Sources that would provide better coverage

4. QUALITY ASSESSMENT: Evaluate the overall research quality:
   - Source diversity and authority
   - Data completeness and accuracy
   - Timeliness of information

CRITICAL: End your response with "OVERALL_RELEVANCE_SCORE: X" where X (1-10) represents how well all sources combined answer the original question.
- Score 9-10: Comprehensive answer with authoritative sources
- Score 7-8: Good answer with solid sources, minor gaps
- Score 5-6: Partial answer, significant gaps or quality issues  
- Score 3-4: Limited answer, major gaps or unreliable sources
- Score 1-2: Poor answer, mostly irrelevant or unreliable information"""

            user_prompt = f"""Original Research Question: {original_question}

Extracted Content from {len(content_summaries)} High-Quality Web Sources (Token-Optimized):
{''.join(content_summaries)}

Please provide a comprehensive analysis including:
1. Individual source relevance scores and assessment
2. Synthesized answer to the original question
3. Gap analysis - what's missing
4. Overall quality assessment
5. Your final overall relevance score (1-10)

Note: Content has been optimized for analysis within token limits."""

            print(f"\n🧠 Starting DeepSeek Reasoning Analysis...")
            print("🔄 [REASONING] Analyzing content relevance and gaps...", flush=True)
            
            analysis_start = time.time()
            
            response = await self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                stream=True,
                timeout=90.0
            )
            
            # Process streaming response with enhanced logging
            reasoning_content = ""
            analysis_content = ""
            reasoning_buffer = ""
            analysis_buffer = ""
            
            async for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    delta = choice.delta
                    
                    # Handle reasoning content streaming
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        reasoning_chunk = delta.reasoning_content
                        reasoning_content += reasoning_chunk
                        reasoning_buffer += reasoning_chunk
                        
                        if len(reasoning_buffer) > 100:
                            print(f"\n🔄 [REASONING] ...{reasoning_buffer[-50:]}", end="", flush=True)
                            reasoning_buffer = ""
                    
                    # Handle regular content streaming
                    if hasattr(delta, 'content') and delta.content:
                        content_chunk = delta.content
                        analysis_content += content_chunk
                        analysis_buffer += content_chunk
                        
                        if len(analysis_buffer) > 100:
                            print(f"\n🔄 [ANALYSIS] ...{analysis_buffer[-50:]}", end="", flush=True)
                            analysis_buffer = ""
            
            # Display remaining content
            if reasoning_buffer:
                print(f"\n🔄 [REASONING] ...{reasoning_buffer}", end="", flush=True)
            if analysis_buffer:
                print(f"\n🔄 [ANALYSIS] ...{analysis_buffer}", end="", flush=True)
            
            analysis_time = time.time() - analysis_start
            total_time = time.time() - start_time
            
            print(f"\n✅ Analysis completed in {analysis_time:.2f}s")
            
            # Extract overall relevance score
            overall_relevance_score = 0
            score_match = re.search(r'OVERALL_RELEVANCE_SCORE:\s*(\d+)', analysis_content)
            if score_match:
                overall_relevance_score = int(score_match.group(1))
                logger.info(f"📊 Extracted relevance score: {overall_relevance_score}/10")
                print(f"📊 Final relevance score: {overall_relevance_score}/10")
            else:
                logger.warning("⚠️ Could not extract overall relevance score")
                print("⚠️ Warning: Could not extract relevance score")
            
            self.metrics.relevance_scores.append(overall_relevance_score)
            
            return {
                'original_question': original_question,
                'analysis_content': analysis_content,
                'reasoning_content': reasoning_content,
                'overall_relevance_score': overall_relevance_score,
                'sources_analyzed': len(extracted_contents),
                'successful_extractions': self.metrics.successful_extractions,
                'cache_hits': self.metrics.cache_hits,
                'cache_misses': self.metrics.cache_misses,
                'analysis_time': analysis_time,
                'total_analysis_time': total_time,
                'model': 'deepseek-reasoner',
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            analysis_time = time.time() - start_time
            error_str = str(e)
            
            # v3.07: Handle token limit errors specifically
            if "max input limit" in error_str or "too long" in error_str.lower():
                logger.error(f"❌ Token limit exceeded in analysis (v3.07 optimization needed): {error_str}")
                logger.info(f"📊 Consider reducing MAX_CONTENT_LENGTH ({MAX_CONTENT_LENGTH}) or TARGET_SOURCES_PER_ITERATION ({TARGET_SOURCES_PER_ITERATION})")
            else:
                logger.error(f"❌ Analysis failed in {analysis_time:.2f}s: {e}")
            
            return {
                'original_question': original_question,
                'error': error_str,
                'error_type': 'token_limit' if "max input limit" in error_str else 'general',
                'analysis_time': analysis_time,
                'cache_hits': self.metrics.cache_hits,
                'cache_misses': self.metrics.cache_misses,
                'content_summarized': self.metrics.content_summarized,
                'tokens_saved': self.metrics.tokens_saved,
                'timestamp': datetime.utcnow().isoformat()
            }

    async def generate_statistical_summary(self, original_question: str, extracted_contents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate statistical summary using DeepSeek API reasoning based on actual data found in sources (v3.07)"""
        start_time = time.time()
        
        logger.info("📊 Starting statistical summary generation (v3.07)")
        
        try:
            # Filter successful extractions
            successful_contents = [c for c in extracted_contents if c.get('success', False)]
            
            if not successful_contents:
                logger.warning("⚠️ No successful extractions available for statistical summary")
                return {
                    'summary_type': 'no_data',
                    'summary_text': 'No data available for statistical summary generation.',
                    'source_urls': [],
                    'generation_time': time.time() - start_time,
                    'error': 'No successful content extractions'
                }
            
            # Extract content for statistical analysis
            content_for_analysis = []
            source_urls = []
            
            for i, content in enumerate(successful_contents):
                # Collect source URLs
                source_urls.append(content['url'])
                
                # Prepare content with focus on numerical data
                analysis_content = f"""
Source {i+1}: {content['title']}
URL: {content['url']}
Content: {content['content'][:1500]}...
"""
                content_for_analysis.append(analysis_content)
            
            self.metrics.statistical_data_found = len(successful_contents)
            
            system_message = """You are an expert data analyst specializing in extracting and analyzing statistical information from web content.

Your task is to create a data-driven summary based ONLY on actual metrics found in the provided sources.

CRITICAL RULES:
1. ONLY use actual numbers, statistics, or data explicitly mentioned in the sources
2. If sources contain numerical data (revenue, users, market share, etc.), use DeepSeek reasoning to analyze and rank
3. If NO numerical data is found, provide qualitative ranking based on source mentions
4. DO NOT fabricate or estimate statistics that aren't clearly stated in sources
5. Always specify the basis for your analysis (statistical vs qualitative)

OUTPUT FORMAT:
For Statistical Data Available:
"STATISTICAL SUMMARY (Based on Found Data):
1. [Product/Company]: [Specific metric from source] - [Additional context]
2. [Product/Company]: [Specific metric from source] - [Additional context]
..."

For No Statistical Data:
"QUALITATIVE SUMMARY (No Statistical Data Found):
Based on research findings, the most mentioned solutions are:
1. [Product/Company] - [Reason for ranking]
2. [Product/Company] - [Reason for ranking]
..."

Always end with source attribution."""

            user_prompt = f"""Original Question: {original_question}

Content from {len(successful_contents)} sources:
{''.join(content_for_analysis)}

TASK: Analyze this content and create a statistical summary. Focus on finding actual numerical data (revenue, users, market share, etc.). If no statistical data is available, provide a qualitative ranking based on source mentions.

Remember: Only use data explicitly mentioned in the sources. Do not estimate or fabricate statistics."""

            logger.info("🔄 Generating statistical summary with DeepSeek reasoning...")
            
            response = await self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                stream=False,
                timeout=60.0
            )
            
            summary_content = response.choices[0].message.content.strip()
            generation_time = time.time() - start_time
            
            # Determine summary type
            summary_type = 'statistical' if 'STATISTICAL SUMMARY' in summary_content else 'qualitative'
            
            # Format source URLs
            formatted_sources = []
            for i, url in enumerate(source_urls, 1):
                formatted_sources.append(f"{i}: {url}")
            
            # Combine summary with source URLs
            final_summary = f"""{summary_content}

SOURCES:
{chr(10).join(formatted_sources)}"""
            
            logger.info(f"✅ Statistical summary generated in {generation_time:.2f}s (Type: {summary_type})")
            
            return {
                'summary_type': summary_type,
                'summary_text': final_summary,
                'source_urls': source_urls,
                'source_count': len(source_urls),
                'generation_time': generation_time,
                'model': 'deepseek-reasoner',
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(f"❌ Statistical summary generation failed in {generation_time:.2f}s: {e}")
            
            return {
                'summary_type': 'error',
                'summary_text': f'Error generating statistical summary: {str(e)}',
                'source_urls': [c['url'] for c in successful_contents if c.get('success', False)],
                'generation_time': generation_time,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def conduct_enhanced_research(self, original_question: str, target_relevance: int = 7, max_iterations: int = 3) -> Dict[str, Any]:
        """Optimized research with time management, token optimization, and progressive responses"""
        # Initialize optimization components
        session_start = self.time_manager.start_timer()
        self.current_session = OptimizedResearchSession(
            session_id=f"research_{int(session_start)}",
            question=original_question,
            start_time=session_start,
            target_sources=10
        )
        
        # Initialize progressive response
        progress_session = self.progressive_response.initialize_response(original_question)
        session_id = progress_session['session_id']
        
        logger.info(f"🚀 Starting OPTIMIZED research v3.07 for: {original_question}")
        logger.info(f"⏰ Time limit: {self.time_manager.max_duration}s with optimization")
        logger.info(f"🎯 Target sources: {self.current_session.target_sources}")
        logger.info(f"🎯 Target relevance: {target_relevance}/10 with max {max_iterations} iterations")
        
        results = {
            'original_question': original_question,
            'timestamp': datetime.now().isoformat(),
            'research_type': 'optimized_research_v305',
            'session_id': self.current_session.session_id,
            'optimization_features': ['time_management', 'token_optimization', 'content_prioritization', 'progressive_response', 'statistical_summary'],
            'target_relevance': target_relevance,
            'phases': [],
            'final_metrics': {},
            'iterations': []
        }
        
        # Track all sources across iterations
        all_sources = []
        current_iteration = 0
        current_relevance = 0
        statistical_summary = None
        
        # Perform iterations until target relevance is met or max iterations reached
        while current_iteration < max_iterations and current_relevance < target_relevance:
            if self.time_manager.should_terminate():
                logger.warning(f"⏰ Time limit approaching, stopping iterations at {current_iteration}")
                break
                
            current_iteration += 1
            logger.info(f"🔄 Starting iteration {current_iteration}/{max_iterations}")
            
            # Track iteration results
            iteration_results = {
                'iteration': current_iteration,
                'relevance_achieved': 0,
                'steps': {},
                'statistical_summary': None
            }
        
        all_sources = []
        
        try:
            # Phase 1: Query Generation (Time Budget: 30s)
            phase_start = time.time()
            self.current_session.current_phase = "query_generation"
            
            if self.time_handler.should_skip_phase("query_generation", self.time_manager.check_time_remaining()):
                logger.warning("⏰ Skipping query generation due to time constraints")
                queries = [original_question]  # Fallback
            else:
                try:
                    queries = await asyncio.wait_for(
                        self.generate_multi_angle_queries(original_question),
                        timeout=self.time_manager.get_phase_time_budget("query_generation")
                    )
                except asyncio.TimeoutError:
                    logger.warning("⏰ Query generation timeout, using fallback")
                    queries = [original_question, f"statistics about {original_question}", f"latest data on {original_question}"]
            
            phase_duration = time.time() - phase_start
            results['phases'].append({
                'phase': 'query_generation',
                'duration': phase_duration,
                'queries_generated': len(queries),
                'success': True
            })
            
            # Phase 2: Web Search (Time Budget: 60s)
            if self.time_manager.should_terminate():
                return self.time_handler.emergency_termination({'sources': all_sources, 'analysis': 'Time limited before search'})
            
            phase_start = time.time()
            self.current_session.current_phase = "web_search"
            
            if self.time_handler.should_skip_phase("web_search", self.time_manager.check_time_remaining()):
                search_results = []
            else:
                try:
                    search_results = await asyncio.wait_for(
                        self.perform_comprehensive_search(queries, max_results_per_query=5),
                        timeout=self.time_manager.get_phase_time_budget("web_search")
                    )
                except asyncio.TimeoutError:
                    logger.warning("⏰ Search timeout, using available results")
                    search_results = []
            
            phase_duration = time.time() - phase_start
            results['phases'].append({
                'phase': 'web_search',
                'duration': phase_duration,
                'results_found': len(search_results),
                'success': len(search_results) > 0
            })
            
            # Phase 3: Prioritized Content Extraction (Time Budget: 360s)
            if self.time_manager.should_terminate():
                return self.time_handler.emergency_termination({'sources': all_sources, 'analysis': 'Time limited before extraction'})
            
            phase_start = time.time()
            self.current_session.current_phase = "content_extraction"
            
            # Apply time-aware filtering
            remaining_time = self.time_manager.check_time_remaining()
            prioritized_results = self.content_prioritizer.filter_by_time_budget(search_results, remaining_time - 150)  # Reserve 150s for analysis
            
            if prioritized_results:
                try:
                    # Extract content with circuit breaker protection
                    extracted_sources = await self.circuit_breaker.call(
                        self.extract_and_analyze_content_optimized, 
                        prioritized_results, 
                        queries,
                        remaining_time - 150
                    )
                    all_sources.extend(extracted_sources)
                    
                    # Update progressive response
                    self.progressive_response.update_with_sources(session_id, extracted_sources)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Content extraction error: {e}")
                    extracted_sources = []
            
            phase_duration = time.time() - phase_start
            results['phases'].append({
                'phase': 'content_extraction',
                'duration': phase_duration,
                'sources_processed': len(all_sources),
                'success': len(all_sources) > 0
            })
            
            # Phase 4: Token-Optimized Analysis (Time Budget: 120s)
            if self.time_manager.should_terminate():
                return self.time_handler.emergency_termination({'sources': all_sources, 'analysis': 'Time limited before analysis'})
            
            phase_start = time.time()
            self.current_session.current_phase = "analysis"
            
            # Optimize content for token limits
            optimized_sources = self.token_optimizer.optimize_content(all_sources)
            
            if optimized_sources:
                try:
                    analysis = await self.circuit_breaker.call(
                        self.analyze_content_with_token_optimization,
                        original_question,
                        optimized_sources,
                        self.time_manager.check_time_remaining()
                    )
                except Exception as e:
                    logger.error(f"❌ Analysis failed: {e}")
                    # Try token limit recovery
                    reduced_sources = self.token_handler.handle_token_error(optimized_sources, e)
                    if reduced_sources:
                        try:
                            analysis = await self.analyze_content_with_token_optimization(
                                original_question, reduced_sources, 60  # Emergency time limit
                            )
                        except Exception as retry_error:
                            logger.error(f"❌ Retry analysis failed: {retry_error}")
                            analysis = self.generate_fallback_analysis(original_question, reduced_sources)
                    else:
                        analysis = self.generate_fallback_analysis(original_question, all_sources)
            else:
                analysis = self.generate_fallback_analysis(original_question, all_sources)
            
            phase_duration = time.time() - phase_start
            results['phases'].append({
                'phase': 'analysis',
                'duration': phase_duration,
                'sources_analyzed': len(optimized_sources),
                'token_optimization_applied': len(optimized_sources) < len(all_sources),
                'success': 'error' not in analysis
            })
            
            # Phase 5: Final Response Generation
            self.current_session.current_phase = "summary_generation"
            
            # Finalize progressive response
            time_constrained = self.time_manager.should_terminate()
            final_response = self.progressive_response.finalize_response(session_id, time_constrained)
            
            # Complete session tracking
            total_duration = time.time() - session_start
            self.current_session.completion_status = "time_limited" if time_constrained else "completed"
            self.current_session.confidence_score = final_response.get('confidence_score', 0.5)
            
            # Build comprehensive results
            results.update({
                'final_analysis': analysis,
                'progressive_response': final_response,
                'optimization_summary': {
                    'total_sources_found': len(all_sources),
                    'sources_after_optimization': len(optimized_sources),
                    'time_constrained': time_constrained,
                    'total_duration': total_duration,
                    'phases_completed': len(results['phases']),
                    'cache_hits': self.metrics.cache_hits,
                    'circuit_breaker_triggered': self.circuit_breaker.failure_count > 0
                },
                'session_metadata': {
                    'session_id': self.current_session.session_id,
                    'completion_status': self.current_session.completion_status,
                    'confidence_score': self.current_session.confidence_score
                }
            })
            
            logger.info(f"✅ Optimized research completed in {total_duration:.1f}s")
            logger.info(f"📊 Processed {len(all_sources)} sources, analyzed {len(optimized_sources)}")
            logger.info(f"🎯 Confidence: {self.current_session.confidence_score:.2f}")
            
            # Mark research as successful
            results['success'] = True
            
            # Add iterations structure for compatibility with display code
            results['iterations'] = [{
                'iteration': 1,
                'relevance_achieved': min(10, max(1, int(self.current_session.confidence_score * 10))),
                'steps': {
                    'step1': {
                        'description': 'Query Generation',
                        'success': True,
                        'time_taken': sum(p['duration'] for p in results['phases'] if p['phase'] == 'query_generation'),
                        'query_count': len(queries)
                    },
                    'step2': {
                        'description': 'Web Search',
                        'success': True,
                        'time_taken': sum(p['duration'] for p in results['phases'] if p['phase'] == 'web_search'),
                        'total_results': sum(p.get('results_found', 0) for p in results['phases'] if p['phase'] == 'web_search')
                    },
                    'step3': {
                        'description': 'Content Extraction',
                        'success': True,
                        'time_taken': sum(p['duration'] for p in results['phases'] if p['phase'] == 'content_extraction'),
                        'sources_extracted': len(all_sources)
                    },
                    'step4': {
                        'description': 'Analysis',
                        'success': bool(analysis),
                        'time_taken': sum(p['duration'] for p in results['phases'] if p['phase'] == 'analysis'),
                        'analysis': analysis or {'analysis_content': 'Analysis not available'}
                    }
                },
                'statistical_summary': analysis.get('statistical_summary') if analysis else None
            }]
            
            # Extract and track relevance score from analysis
            if analysis and 'overall_relevance_score' in analysis:
                analysis_score = analysis['overall_relevance_score']
                self.score_calculator.add_iteration_score(analysis_score)
                logger.info(f"📊 Extracted relevance score from analysis: {analysis_score}/10")
            else:
                # Fallback scoring based on sources and success
                fallback_score = min(8, max(3, len(all_sources))) if all_sources else 0
                self.score_calculator.add_iteration_score(fallback_score)
                logger.warning(f"⚠️ Using fallback relevance score: {fallback_score}/10")
            
            # Calculate final metrics using enhanced score calculator and success validator
            total_duration = time.time() - session_start
            score_summary = self.score_calculator.get_calculation_summary()
            
            # Enhanced success determination
            success_evaluation = self.success_validator.determine_research_success(
                score_summary, analysis, all_sources
            )
            
            results['final_metrics'] = {
                'final_relevance_score': score_summary['final_relevance_score'],
                'target_achieved': score_summary['target_achieved'],
                'iterations_completed': current_iteration,
                'total_duration': total_duration,
                'score_progression': score_summary['score_progression'],
                'calculation_method': score_summary['calculation_method'],
                'success_evaluation': success_evaluation
            }
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Research failed: {e}")
            # Emergency response generation
            emergency_results = self.time_handler.emergency_termination({
                'sources': all_sources,
                'analysis': f'Research failed due to error: {str(e)}'
            })
            results['emergency_termination'] = emergency_results
            results['error'] = str(e)
            results['success'] = False
            return results
    
    async def extract_and_analyze_content_optimized(self, search_results: List[Dict[str, Any]], queries: List[str], time_budget: int) -> List[Dict[str, Any]]:
        """Optimized content extraction with time constraints and prioritization"""
        logger.info(f"🔍 Starting optimized content extraction for {len(search_results)} sources (budget: {time_budget}s)")
        
        # Prioritize sources
        prioritized_sources = self.content_prioritizer.rank_sources(search_results)
        
        extracted_sources = []
        extraction_start = time.time()
        
        for i, source in enumerate(prioritized_sources):
            # Check time budget
            elapsed = time.time() - extraction_start
            if elapsed >= time_budget:
                logger.warning(f"⏰ Content extraction time budget exceeded after {i} sources")
                break
            
            try:
                # Extract with timeout based on remaining budget
                remaining_time = time_budget - elapsed
                source_timeout = min(45, remaining_time // max(1, len(prioritized_sources) - i))
                
                # Normalize URL key (Google search uses 'link', but extraction expects 'url')
                url = source.get('url') or source.get('link', '')
                if not url:
                    logger.warning(f"⚠️ No URL found in source: {source}")
                    continue
                
                extracted = await asyncio.wait_for(
                    self.content_extractor.extract_article_content(url, queries),
                    timeout=source_timeout
                )
                
                if extracted and extracted.get('success'):
                    extracted['priority_score'] = source.get('priority_score', 0.5)
                    extracted_sources.append(extracted)
                    
                    # Update metrics
                    self.metrics.successful_extractions += 1
                    if extracted.get('from_cache'):
                        self.metrics.cache_hits += 1
                    else:
                        self.metrics.cache_misses += 1
                
            except asyncio.TimeoutError:
                url = source.get('url') or source.get('link', 'unknown')
                logger.warning(f"⏰ Extraction timeout for {url}")
                self.metrics.failed_extractions += 1
            except Exception as e:
                url = source.get('url') or source.get('link', 'unknown')
                logger.warning(f"⚠️ Extraction failed for {url}: {e}")
                self.metrics.failed_extractions += 1
        
        logger.info(f"✅ Extracted {len(extracted_sources)} sources in {time.time() - extraction_start:.1f}s")
        return extracted_sources
    
    async def analyze_content_with_token_optimization(self, question: str, sources: List[Dict[str, Any]], time_budget: int) -> Dict[str, Any]:
        """Token-optimized content analysis with fallback strategies"""
        logger.info(f"🧠 Starting token-optimized analysis of {len(sources)} sources")
        
        try:
            # Prepare content for analysis
            source_contents = []
            for i, source in enumerate(sources[:10]):  # Limit to top 10 sources
                content_summary = f"Source {i+1}: {source.get('title', 'Unknown')}\n"
                content_summary += f"URL: {source.get('url', 'Unknown')}\n"
                content_summary += f"Content: {source.get('content', '')}\n"
                content_summary += f"Quality Score: {source.get('domain_info', {}).get('quality_score', 5)}/10\n\n"
                source_contents.append(content_summary)
            
            combined_content = '\n'.join(source_contents)
            
            # Count tokens and optimize if needed
            token_count = self.token_optimizer.count_tokens(combined_content)
            if token_count > self.token_optimizer.max_tokens:
                logger.warning(f"⚠️ Content exceeds token limit: {token_count} > {self.token_optimizer.max_tokens}")
                # Use optimization to reduce content
                optimized_sources = self.token_optimizer.optimize_content(sources)
                if optimized_sources:
                    # Rebuild content with optimized sources
                    source_contents = []
                    for i, source in enumerate(optimized_sources):
                        content_summary = f"Source {i+1}: {source.get('title', 'Unknown')}\n"
                        content_summary += f"Content: {source.get('content', '')}\n\n"
                        source_contents.append(content_summary)
                    combined_content = '\n'.join(source_contents)
            
            # Create analysis prompt
            analysis_prompt = f"""Analyze the following research content and provide a comprehensive answer to the question.

Question: {question}

Research Sources:
{combined_content}

Instructions:
1. Provide a comprehensive answer based on the sources
2. Include specific data, statistics, and factual information where available
3. Rate the overall relevance of the sources (1-10 scale)
4. Identify any gaps in the research
5. Provide confidence score (0.0-1.0) based on source quality and completeness

Format your response as JSON with the following structure:
{{
    "comprehensive_answer": "Your detailed answer here",
    "key_findings": ["finding 1", "finding 2", "finding 3"],
    "statistics_found": ["stat 1", "stat 2"],
    "overall_relevance_score": 8,
    "confidence_score": 0.85,
    "gaps_identified": ["gap 1", "gap 2"],
    "source_quality_assessment": "assessment of source quality"
}}"""

            # Make API call with timeout
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": analysis_prompt}],
                    stream=False,
                    temperature=0.3
                ),
                timeout=min(time_budget, 90)
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Use robust API response parser
            analysis_result = self.response_parser.parse_analysis_response(result_text)
            analysis_result['token_optimization_applied'] = token_count > self.token_optimizer.max_tokens
            analysis_result['sources_analyzed'] = len(sources)
            
            # Add statistical summary if available
            if 'comprehensive_answer' in analysis_result:
                analysis_result['analysis_content'] = analysis_result.get('comprehensive_answer', result_text)
            else:
                analysis_result['analysis_content'] = result_text
            
            logger.info(f"📊 Analysis parsed using method: {analysis_result.get('parsing_method', 'unknown')}")
            return analysis_result
                
        except asyncio.TimeoutError:
            logger.warning("⏰ Analysis timeout, generating emergency response")
            return self.generate_fallback_analysis(question, sources)
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            return self.generate_fallback_analysis(question, sources)
    
    def generate_fallback_analysis(self, question: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate fallback analysis when main analysis fails"""
        logger.info("🆘 Generating fallback analysis")
        
        # Extract key information from sources
        source_titles = [s.get('title', 'Unknown') for s in sources[:5]]
        source_count = len(sources)
        successful_sources = len([s for s in sources if s.get('success', False)])
        
        # Calculate basic confidence
        confidence = min(0.7, successful_sources * 0.1)
        
        fallback_answer = f"""Based on the available research sources, here is a summary for: {question}

Research Summary:
- Analyzed {source_count} sources, with {successful_sources} successful extractions
- Key sources include: {', '.join(source_titles)}

This analysis was generated under constraints and may be incomplete. The research process encountered limitations that prevented full analysis.

Recommendations:
1. Consider running the research again with more time
2. Verify findings with additional sources
3. Focus on the highest-quality sources identified
"""
        
        return {
            'comprehensive_answer': fallback_answer,
            'overall_relevance_score': 5,
            'confidence_score': confidence,
            'sources_analyzed': source_count,
            'successful_extractions': successful_sources,
            'fallback_analysis': True,
            'limitation_note': 'This is a fallback analysis due to processing constraints'
        }
    
    async def generate_followup_queries(self, original_question: str, previous_iteration: Dict[str, Any]) -> List[str]:
        """Generate follow-up queries based on gaps identified in previous iteration"""
        start_time = time.time()
        
        try:
            logger.info("🎯 Generating follow-up queries based on identified gaps")
            
            previous_analysis = previous_iteration['steps']['step4']['analysis']
            previous_score = previous_iteration['relevance_achieved']
            
            system_message = """You are an expert research strategist. Based on a previous research iteration and its gaps, generate targeted follow-up search queries to fill the missing information.

Instructions:
1. Analyze the previous analysis to identify specific gaps and weaknesses
2. Generate 3-4 targeted search queries that address these gaps
3. Focus on missing data types, unexplored angles, or contradictory information
4. Make queries specific and different from previous searches
5. Format response as: Query1="...", Query2="...", Query3="...", etc."""

            user_prompt = f"""Original Question: {original_question}

Previous Iteration Results:
- Relevance Score: {previous_score}/10
- Analysis Summary: {previous_analysis.get('analysis_content', '')[:1000]}...

Generate 3-4 targeted follow-up search queries to address the gaps and improve relevance."""

            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                stream=False,
                timeout=30.0
            )
            
            response_text = response.choices[0].message.content
            generation_time = time.time() - start_time
            
            # Extract queries from response
            queries = []
            query_pattern = r'Query\d*="([^"]+)"'
            matches = re.findall(query_pattern, response_text)
            
            if matches:
                queries = matches
                logger.info(f"✅ Generated {len(queries)} follow-up queries in {generation_time:.2f}s")
                for i, query in enumerate(queries, 1):
                    logger.info(f"  {i}. {query}")
            else:
                # Fallback: modify original question
                queries = [f"{original_question} latest data", f"{original_question} market analysis", f"{original_question} revenue statistics"]
                logger.warning("⚠️ Could not extract follow-up queries, using modified versions")
            
            self.metrics.total_queries += len(queries)
            return queries
                
        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(f"❌ Follow-up query generation failed in {generation_time:.2f}s: {e}")
            return [f"{original_question} additional information"]
    
    async def cleanup(self):
        """Clean up resources"""
        await self.cache_service.close()
    
    async def conduct_relevance_enhanced_research(self, original_question: str, relevance_threshold: float = 0.7) -> Dict[str, Any]:
        """
        関連性評価強化版リサーチ実行
        
        Args:
            original_question: 元の質問
            relevance_threshold: 関連性閾値 (0.7 = 70%)
            
        Returns:
            Dict[str, Any]: 関連性評価結果を含む研究結果
        """
        logger.info(f"🚀 関連性評価強化版リサーチ開始: {original_question}")
        logger.info(f"🎯 関連性閾値: {relevance_threshold * 100}%")
        
        start_time = time.time()
        
        # 基本的な研究を実行
        base_results = await self.conduct_enhanced_research(original_question, target_relevance=7, max_iterations=1)
        
        if not base_results.get('success'):
            logger.error("❌ 基本研究が失敗しました")
            return base_results
        
        # 抽出されたコンテンツに対して関連性評価を実行
        extracted_contents = []
        for iteration in base_results.get('iterations', []):
            for step_key, step_data in iteration.get('steps', {}).items():
                if 'extracted_contents' in step_data:
                    extracted_contents.extend(step_data['extracted_contents'])
        
        if not extracted_contents:
            logger.warning("⚠️ 抽出されたコンテンツがありません")
            return base_results
        
        # 関連性評価を実行
        evaluations = await self.relevance_evaluator.batch_evaluate(original_question, extracted_contents)
        
        # 高関連性コンテンツをフィルタリング
        high_relevance_evaluations = self.relevance_evaluator.filter_high_relevance(evaluations)
        
        if not high_relevance_evaluations:
            logger.warning("❌ 70%以上の関連性を持つコンテンツが見つかりませんでした")
            logger.info("🔄 追加の検索クエリを生成して再検索を実行します")
            
            # 追加検索クエリ生成（フォールバック）
            additional_queries = await self._generate_additional_queries(original_question, evaluations)
            if additional_queries:
                logger.info(f"🔍 追加クエリ: {additional_queries}")
                # 追加検索の実装は次のタスクで実装
            
            return {
                **base_results,
                'relevance_enhancement': {
                    'evaluations': [eval.__dict__ for eval in evaluations],
                    'high_relevance_count': 0,
                    'threshold_met': False,
                    'additional_queries_generated': additional_queries
                }
            }
        
        # 高関連性回答を集計・ランキング
        aggregated_answers = self.answer_aggregator.aggregate_answers(evaluations)
        
        # 最も関連性の高い回答を取得
        top_answer = self.answer_aggregator.get_top_answer()
        
        # 最高関連性回答の要約を生成
        summary_result = None
        if top_answer:
            try:
                summary_result = await self.summary_generator.generate_summary(original_question, top_answer)
                logger.info(f"📝 要約生成成功: 品質グレード {summary_result.confidence_metrics.get('quality_grade', 'N/A')}")
            except Exception as e:
                logger.error(f"❌ 要約生成エラー: {e}")
        
        # 関連性評価結果を統合
        processing_time = time.time() - start_time
        
        enhanced_results = {
            **base_results,
            'relevance_enhancement': {
                'total_evaluations': len(evaluations),
                'high_relevance_count': len(high_relevance_evaluations),
                'threshold_met': True,
                'relevance_threshold': relevance_threshold,
                'evaluations': [eval.__dict__ for eval in evaluations],
                'high_relevance_evaluations': [eval.__dict__ for eval in high_relevance_evaluations],
                'evaluation_stats': self.relevance_evaluator.get_evaluation_stats(),
                'processing_time': processing_time
            },
            'answer_aggregation': {
                'aggregated_answers': [answer.__dict__ for answer in aggregated_answers],
                'top_answer': top_answer.__dict__ if top_answer else None,
                'aggregation_stats': self.answer_aggregator.get_aggregation_stats(),
                'total_aggregated': len(aggregated_answers)
            },
            'summary_generation': {
                'summary_result': summary_result.__dict__ if summary_result else None,
                'summary_generated': summary_result is not None,
                'summary_text': summary_result.summary_text if summary_result else None,
                'confidence_metrics': summary_result.confidence_metrics if summary_result else None
            }
        }
        
        logger.info(f"✅ 関連性評価強化版リサーチ完了:")
        logger.info(f"   処理時間: {processing_time:.2f}秒")
        logger.info(f"   評価件数: {len(evaluations)}件")
        logger.info(f"   高関連性: {len(high_relevance_evaluations)}件")
        logger.info(f"   集計回答: {len(aggregated_answers)}件")
        logger.info(f"   最高ランク: {top_answer.relevance_score:.1f}/10" if top_answer else "   最高ランク: なし")
        
        # 結果の構造化表示を生成・表示
        formatted_result = self.result_formatter.format_final_result(
            summary_result=summary_result,
            aggregated_answers=aggregated_answers, 
            original_question=original_question
        )
        
        self.result_formatter.display_formatted_result(formatted_result)
        
        # 結果にフォーマット済み表示も追加
        enhanced_results['formatted_display'] = {
            'summary_text': formatted_result.summary_text,
            'metadata_table': formatted_result.metadata_table,
            'confidence_display': formatted_result.confidence_display,
            'source_list': formatted_result.source_list,
            'error_message': formatted_result.error_message,
            'fallback_data': formatted_result.fallback_data
        }
        
        return enhanced_results
    
    async def _generate_additional_queries(self, original_question: str, evaluations: List) -> List[str]:
        """
        関連性が不足している場合の追加検索クエリ生成
        
        Args:
            original_question: 元の質問
            evaluations: 関連性評価結果
            
        Returns:
            List[str]: 追加検索クエリ
        """
        try:
            # 低関連性評価の理由を分析して改善クエリを生成
            low_relevance_reasons = [
                eval.evaluation_reason for eval in evaluations 
                if not eval.meets_threshold
            ]
            
            prompt = f"""
元の質問: {original_question}

低関連性と判定された理由:
{chr(10).join(low_relevance_reasons[:3])}

上記の問題を解決するため、より具体的で関連性の高い情報を得られる検索クエリを3つ生成してください。
形式: Query1="...", Query2="...", Query3="..."
"""
            
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "あなたは検索クエリ最適化の専門家です。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300
            )
            
            queries_text = response.choices[0].message.content
            queries = re.findall(r'Query\d+="([^"]+)"', queries_text)
            
            return queries[:3] if queries else [f"{original_question} 詳細情報"]
            
        except Exception as e:
            logger.error(f"❌ 追加クエリ生成エラー: {e}")
            return [f"{original_question} 詳細情報"]

def print_separator(char="=", length=80):
    """Print a separator line"""
    print("\n" + char * length + "\n")

def print_step_header(step_num: int, description: str):
    """Print a step header"""
    print(f"\n🎯 STEP {step_num}: {description}")
    print("-" * 60)

async def test_enhanced_research():
    """Test the iterative enhanced research process with MongoDB caching and statistical summaries (v3.07)"""
    print_separator()
    print("🚀 DEEPSEEK ENHANCED WEB RESEARCH v3.07 - WITH STATISTICAL SUMMARIES")
    print("Enhanced Multi-Query Research Process with Statistical Summary Generation (v3.07):")
    print("1. Initialize MongoDB connection and check cache statistics")
    print("2. Search existing cached content based on generated queries")
    print("3. Generate multiple search queries from different angles")
    print("4. Perform comprehensive search with smart URL deduplication")
    print("5. Extract content with Bright Data API and MongoDB caching")
    print("6. Analyze with gap identification and relevance scoring (with content summarization)")
    print("7. Generate statistical summary using DeepSeek API reasoning")
    print("8. If target relevance not met, generate follow-up queries and repeat")
    print("9. Comprehensive performance tracking including cache and API metrics")
    print("10. 10-minute time limit with intelligent content management")
    print_separator()
    
    # Initialize service
    try:
        service = EnhancedDeepSeekResearchService()
        await service.initialize()
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    # Display initial cache statistics
    cache_stats = await service.cache_service.get_cache_stats()
    print(f"💾 MongoDB Cache Status:")
    print(f"   📊 Total Entries: {cache_stats.get('total_entries', 0)}")
    print(f"   🔥 Fresh Entries (≤7 days): {cache_stats.get('fresh_entries', 0)}")
    print(f"   🔌 Connected: {'Yes' if cache_stats.get('connected') else 'No'}")
    
    # Test question
    original_question = "Find the CRM/SFA software available in Japan and make the rank by their revenues"
    target_relevance = 7
    print(f"\n🎯 Research Question: {original_question}")
    print(f"📊 Target Relevance Score: {target_relevance}/10")
    print(f"⏰ Time Limit: {MAX_RESEARCH_TIME//60} minutes")
    
    # Conduct iterative research with MongoDB caching and statistical summaries
    start_time = time.time()
    results = await service.conduct_enhanced_research(original_question, target_relevance=target_relevance, max_iterations=3)
    total_time = time.time() - start_time
    
    if not results.get('success'):
        print(f"❌ Research failed: {results.get('error', 'Unknown error')}")
        await service.cleanup()
        return
    
    # Display iteration results
    print_separator("=", 100)
    print("📊 ITERATION RESULTS WITH STATISTICAL SUMMARY GENERATION (v3.07)")
    
    for iteration_data in results['iterations']:
        iteration_num = iteration_data['iteration']
        relevance = iteration_data['relevance_achieved']
        
        print(f"\n🔄 ITERATION {iteration_num}:")
        print(f"   📊 Relevance Score: {relevance}/10")
        
        # Show key metrics for each step
        for step_key, step_data in iteration_data['steps'].items():
            step_name = step_data['description']
            time_taken = step_data.get('time_taken', 0)
            success = "✅" if step_data.get('success', False) else "❌"
            
            if 'query_count' in step_data:
                print(f"   {success} {step_name}: {step_data['query_count']} queries in {time_taken:.2f}s")
            elif 'cached_results' in step_data:
                print(f"   {success} {step_name}: {step_data['cached_results']} cached results in {time_taken:.2f}s")
            elif 'total_results' in step_data:
                print(f"   {success} {step_name}: {step_data['total_results']} results in {time_taken:.2f}s")
            elif 'new_extractions' in step_data:
                cache_hits = step_data.get('cache_hits', 0)
                cache_misses = step_data.get('cache_misses', 0)
                cached_extractions = step_data.get('cached_extractions', 0)
                print(f"   {success} {step_name}: {step_data['successful_new']}/{step_data['new_extractions']} successful + {cached_extractions} cached in {time_taken:.2f}s")
                if cache_hits + cache_misses > 0:
                    print(f"      💾 Cache: {cache_hits} hits, {cache_misses} misses")
            elif 'sources_analyzed' in step_data:
                print(f"   {success} {step_name}: {step_data['sources_analyzed']} sources in {time_taken:.2f}s")
    
    # Final comprehensive results with statistical summary
    final_metrics = results.get('final_metrics', {})
    final_score = final_metrics.get('final_relevance_score', 0)
    target_achieved = final_metrics.get('target_achieved', False)
    cache_performance = final_metrics.get('search_metrics', {}).get('cache_performance', {})
    statistical_summary = final_metrics.get('statistical_summary', {})
    
    print_separator("=", 100) 
    print("🎉 FINAL RESULTS WITH STATISTICAL SUMMARY (v3.07)")
    
    print(f"🎯 Target Achievement: {'SUCCESS' if target_achieved else 'PARTIAL'}")
    print(f"📊 Final Relevance Score: {final_score}/10 (Target: {target_relevance}/10)")
    print(f"🔄 Iterations Completed: {final_metrics.get('iterations_completed', 0)}/3")
    print(f"⏱️ Total Research Time: {final_metrics.get('total_duration', 0):.2f}s")
    print(f"⏰ Time Limit: {MAX_RESEARCH_TIME}s ({'Exceeded' if final_metrics.get('total_duration', 0) > MAX_RESEARCH_TIME else 'Within Limit'})")
    
    # Statistical Summary Display (v3.07)
    print(f"\n📊 STATISTICAL SUMMARY GENERATION (v3.07):")
    print(f"   📈 Summary Type: {statistical_summary.get('summary_type', 'not_available')}")
    print(f"   📄 Sources Used: {statistical_summary.get('source_count', 0)}")
    print(f"   ⏱️ Generation Time: {statistical_summary.get('generation_time', 0):.2f}s")
    
    # API and Cache performance breakdown
    print(f"\n🌐 BRIGHT DATA API & MONGODB CACHE PERFORMANCE:")
    print(f"   🎯 Cache Hits: {cache_performance.get('cache_hits', 0)}")
    print(f"   🌐 Cache Misses: {cache_performance.get('cache_misses', 0)}")
    print(f"   📊 Cache Hit Rate: {cache_performance.get('cache_hit_rate', 0):.1f}%")
    print(f"   📚 Total Cache Entries: {cache_performance.get('total_cache_entries', 0)}")
    print(f"   🔥 Fresh Cache Entries: {cache_performance.get('fresh_cache_entries', 0)}")
    
    # Performance breakdown
    search_metrics = final_metrics.get('search_metrics', {})
    print(f"\n📈 PERFORMANCE BREAKDOWN:")
    print(f"   🔍 Total Queries: {search_metrics.get('total_queries', 0)}")
    print(f"   📄 Total Sources Found: {search_metrics.get('total_results', 0)}")
    print(f"   ✅ Successful Extractions: {search_metrics.get('successful_extractions', 0)}")
    print(f"   📊 Extraction Success Rate: {search_metrics.get('extraction_success_rate', 0):.1f}%")
    print(f"   📊 Statistical Data Found: {search_metrics.get('statistical_data_found', 0)} sources")
    
    # Relevance progression
    relevance_progression = search_metrics.get('relevance_progression', [])
    if relevance_progression:
        print(f"   📈 Relevance Progression: {' → '.join(map(str, relevance_progression))}")
    
    # Source distribution
    source_dist = search_metrics.get('source_distribution', {})
    if source_dist:
        print(f"   🏢 Source Types: {dict(source_dist)}")
    
    # Display statistical summary (v3.07)
    if statistical_summary and statistical_summary.get('summary_text'):
        print_separator("⭐", 100)
        print("📊 STATISTICAL SUMMARY (v3.07):")
        print(statistical_summary['summary_text'])
        print_separator("⭐", 100)
    
    # Display enhanced comprehensive analysis
    if results['iterations']:
        final_iteration = results['iterations'][-1]
        final_analysis = final_iteration['steps']['step4']['analysis']
        
        print_separator("-", 80)
        # Use enhanced display formatter for better analysis presentation
        formatted_analysis = service.display_formatter.format_comprehensive_analysis(final_analysis)
        print(formatted_analysis)
    
    # Display progressive answer
    progressive_answer = final_metrics.get('progressive_answer', {})
    if progressive_answer and progressive_answer.get('final_answer'):
        print_separator("-", 80)
        print("🎯 FINAL PROGRESSIVE ANSWER:")
        print(f"❓ Question: {progressive_answer['question']}")
        print(f"📊 Confidence: {progressive_answer['confidence_score']:.2f}")
        print(f"📄 Sources: {progressive_answer['sources_analyzed']}")
        print(f"🔄 Versions: {progressive_answer['versions_count']}")
        print_separator("-", 80)
        print("✅ ANSWER:")
        print(progressive_answer['final_answer'])
    
    print_separator("=", 100)
    
    # Enhanced Success/failure assessment
    success_evaluation = final_metrics.get('success_evaluation', {})
    if success_evaluation:
        success_level = success_evaluation.get('success_level', 'unknown')
        overall_success = success_evaluation.get('overall_success', False)
        success_reasoning = success_evaluation.get('success_reasoning', '')
        recommendations = success_evaluation.get('recommendations', [])
        
        # Display success status with enhanced information
        if overall_success:
            if success_level == 'excellent':
                print("🎉 EXCELLENT SUCCESS: Research exceeded expectations!")
            else:
                print("✅ SUCCESS: Research completed successfully!")
        else:
            if success_level == 'partial':
                print("⚠️ PARTIAL SUCCESS: Research partially completed")
            else:
                print("❌ RESEARCH INCOMPLETE: Significant issues identified")
        
        print(f"📊 Success Level: {success_level.title()}")
        print(f"🎯 Final Score: {final_score}/10 (Target: {target_relevance}/10)")
        
        # Show success reasoning
        if success_reasoning:
            print(f"📋 Evaluation: {success_reasoning}")
        
        # Show recommendations if any
        if recommendations:
            print("💡 Recommendations for improvement:")
            for i, rec in enumerate(recommendations[:3], 1):  # Show top 3
                print(f"   {i}. {rec}")
        
        # Show cache performance
        if cache_performance.get('cache_hits', 0) > 0:
            print(f"💾 Cache optimization: {cache_performance.get('cache_hits', 0)} URLs served from cache")
        cache_misses = cache_performance.get('cache_misses', 0)
        if cache_misses > 0:
            print(f"🌐 Bright Data API calls: {cache_misses} URLs extracted via API")
    else:
        # Fallback to original logic if success evaluation is missing
        if target_achieved:
            print("🎉 SUCCESS: Target relevance score achieved!")
            print(f"✅ Research completed with {final_score}/10 relevance (≥{target_relevance} required)")
        else:
            print("⚠️ PARTIAL SUCCESS: Target relevance not fully achieved")
            print(f"📊 Final score: {final_score}/10 (Target: {target_relevance}/10)")
            print("💡 Consider running additional iterations or refining the research question")
    
    # Save comprehensive results with statistical summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'enhanced_research_v3_07_optimized_results_{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {filename}")
    
    # Get final cache statistics
    final_cache_stats = await service.cache_service.get_cache_stats()
    print(f"📊 Final cache statistics: {final_cache_stats.get('total_entries', 0)} total entries")
    
    # Cleanup
    await service.cleanup()
    print("🚀 Enhanced research with Statistical Summary generation (v3.07) completed!")

def check_environment():
    """Check if required environment variables are set"""
    print("🔧 Checking environment variables...")
    
    required_vars = {
        'DEEPSEEK_API_KEY': os.environ.get('DEEPSEEK_API_KEY'),
        'GOOGLE_API_KEY': os.environ.get('GOOGLE_API_KEY'),
        'GOOGLE_CSE_ID': os.environ.get('GOOGLE_CSE_ID'),
        'BRIGHTDATA_API_KEY': os.environ.get('BRIGHTDATA_API_KEY'),
        'MONGODB_URI': os.environ.get('MONGODB_URI', 'mongodb://localhost:27017')
    }
    
    missing_vars = []
    for var, value in required_vars.items():
        if var == 'MONGODB_URI':
            print(f"✅ {var}: {value}")  # MongoDB URI is OK to show
        elif value and value != f"your_{var.lower()}_here":
            masked_value = f"{value[:5]}...{value[-5:]}" if len(value) > 10 else "***"
            print(f"✅ {var}: {masked_value}")
        else:
            print(f"❌ {var}: Not set")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n⚠️ Missing environment variables: {', '.join(missing_vars)}")
        print("\nPlease configure your .env file with the required API keys.")
        if 'MONGODB_URI' not in missing_vars:
            print("💡 MongoDB URI is set - cache functionality will be available")
        if 'BRIGHTDATA_API_KEY' not in missing_vars:
            print("💡 Bright Data API key is set - professional content extraction will be available")
        else:
            print("⚠️ Without BRIGHTDATA_API_KEY, the system will fall back to basic content extraction")
        return False
    
    return True

async def main():
    """Main test function"""
    print("🚀 DEEPSEEK ENHANCED WEB RESEARCH v3.07 - WITH STATISTICAL SUMMARIES")
    print("=" * 70)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment setup incomplete. Please configure required variables.")
        sys.exit(1)
    
    try:
        # Run the enhanced iterative research test with MongoDB caching and statistical summaries
        await test_enhanced_research()
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
