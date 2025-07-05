#!/usr/bin/env python3
"""
AI Reasoning Service for Research System
Enhanced DeepSeek integration with intelligent model selection and reasoning capabilities
"""

import logging
import re
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from openai import AsyncOpenAI
from .interfaces import IAIReasoningService, ExtractedContent, AnalysisResult
from .config import get_config_manager
from .metrics import get_metrics_collector

logger = logging.getLogger(__name__)


class DeepSeekReasoningService(IAIReasoningService):
    """Enhanced DeepSeek reasoning service with intelligent model selection"""
    
    def __init__(self):
        self.config = get_config_manager()
        self.metrics = get_metrics_collector()
        
        # Get DeepSeek credentials
        deepseek_creds = self.config.get_api_credentials('deepseek')
        self.api_key = deepseek_creds['api_key']
        self.api_url = deepseek_creds['api_url']
        
        if not self.api_key:
            logger.error("❌ DEEPSEEK_API_KEY not set")
            raise ValueError("DEEPSEEK_API_KEY environment variable is required")
        
        # Initialize async client
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.api_url)
        
        # Get model settings
        self.model_settings = self.config.get_model_settings()
        self.research_settings = self.config.get_research_settings()
        
        logger.info(f"DeepSeek reasoning service initialized with API URL: {self.api_url}")
    
    def _get_model_for_task(self, task_type: str, search_mode: str = "standard") -> str:
        """Get the appropriate model for a specific task"""
        if task_type in ["analysis", "reasoning", "complex_analysis"]:
            return self.model_settings['primary_reasoning_model']
        elif task_type in ["query_generation", "necessity_check", "simple_chat"]:
            return self.model_settings['primary_chat_model']
        else:
            # Use mode-based selection
            return self.config.get_model_for_mode(search_mode)
    
    def _get_fallback_models(self, current_model: str) -> List[str]:
        """Get fallback models excluding the current model"""
        return self.config.get_fallback_models(exclude_model=current_model)
    
    async def _make_api_call(self, messages: List[Dict[str, str]], task_type: str, 
                           search_mode: str = "standard", stream: bool = False, 
                           timeout: float = None) -> Any:
        """Make API call with retry logic and model fallback"""
        primary_model = self._get_model_for_task(task_type, search_mode)
        fallback_models = self._get_fallback_models(primary_model)
        models_to_try = [primary_model] + fallback_models
        
        timeout = timeout or self.research_settings['analysis_timeout']
        max_retries = self.research_settings['max_retries']
        
        timing_id = self.metrics.start_timing(f"ai_api_call_{task_type}", {
            "model": primary_model,
            "stream": str(stream)
        })
        
        for model_index, model in enumerate(models_to_try):
            for retry in range(max_retries):
                try:
                    logger.info(f"🤖 API call: {task_type} using {model} (attempt {retry+1}/{max_retries})")
                    
                    response = await self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        stream=stream,
                        timeout=timeout
                    )
                    
                    # Record successful API call
                    self.metrics.increment_counter("ai_api_calls_successful", tags={
                        "model": model,
                        "task_type": task_type
                    })
                    
                    logger.info(f"✅ API call successful: {task_type} with {model}")
                    return response
                    
                except Exception as e:
                    error_msg = str(e)
                    logger.warning(f"⚠️ API call failed: {task_type} with {model} (attempt {retry+1}): {error_msg}")
                    
                    self.metrics.increment_counter("ai_api_calls_failed", tags={
                        "model": model,
                        "task_type": task_type,
                        "error_type": "api_error"
                    })
                    
                    # Check if it's a 500 Internal Server Error or model-specific issue
                    if "500" in error_msg or "Internal Server Error" in error_msg:
                        if retry < max_retries - 1:
                            wait_time = (2 ** retry) * 1.0  # Exponential backoff
                            await asyncio.sleep(wait_time)
                            continue
                        elif model_index < len(models_to_try) - 1:
                            # Try next model
                            logger.info(f"🔄 Switching to fallback model: {models_to_try[model_index + 1]}")
                            break
                    
                    # For the last retry of the last model
                    if retry == max_retries - 1 and model_index == len(models_to_try) - 1:
                        logger.error(f"❌ All API call attempts failed for {task_type}")
                        self.metrics.end_timing(timing_id)
                        raise Exception(f"All API call attempts failed: {error_msg}")
                    
                    if retry < max_retries - 1:
                        await asyncio.sleep(1 * (retry + 1))
        
        self.metrics.end_timing(timing_id)
        raise Exception("All models and retries exhausted")
    
    async def check_web_search_necessity(self, question: str) -> Dict[str, Any]:
        """Check if web search is necessary for answering the question"""
        timing_id = self.metrics.start_timing("necessity_check")
        
        try:
            logger.info("🔍 Checking if web search is necessary")
            
            system_message = """You are an expert research assistant. Your task is to analyze a question and determine whether web search is necessary to provide a comprehensive and accurate answer.

Instructions:
1. Analyze the given question carefully
2. Determine if the question requires current/recent information, specific data, or real-time facts that would benefit from web search
3. If web search IS needed, respond with: Query="your_optimized_search_query_here"
4. If web search is NOT needed, provide a comprehensive answer directly

Questions that typically NEED web search:
- Current events, recent news, latest developments
- Specific company information, financial data, market rankings
- Real-time statistics, current prices, recent reports
- Location-specific information that changes frequently
- Technical specifications of recent products

Questions that typically DON'T need web search:
- General knowledge questions
- Historical facts (unless very recent)
- Mathematical calculations
- Theoretical concepts and explanations
- Programming concepts and syntax
- Well-established scientific principles"""

            user_prompt = f"""Question: {question}

Please analyze this question and determine if web search is necessary. If web search is needed, respond with Query="search_terms". If not needed, provide a comprehensive answer directly."""

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ]
            
            response = await self._make_api_call(messages, "necessity_check", timeout=30.0)
            response_text = response.choices[0].message.content
            
            logger.info(f"🔍 Necessity check response: {response_text[:200]}...")
            
            # Check if response contains a search query
            query_match = re.search(r'Query="([^"]+)"', response_text)
            if query_match:
                search_query = query_match.group(1)
                logger.info(f"✅ Web search needed. Extracted search query: {search_query}")
                result = {
                    'web_search_needed': True,
                    'search_query': search_query,
                    'response': response_text,
                    'model': response.model
                }
            else:
                logger.info("✅ Web search not needed. Direct answer provided.")
                result = {
                    'web_search_needed': False,
                    'direct_answer': response_text,
                    'response': response_text,
                    'model': response.model
                }
            
            self.metrics.increment_counter("necessity_checks", tags={
                "web_search_needed": str(result['web_search_needed'])
            })
            
            return result
                
        except Exception as e:
            logger.error(f"❌ Necessity check failed: {e}")
            self.metrics.increment_counter("necessity_check_errors")
            # Default to web search if check fails
            return {
                'web_search_needed': True,
                'search_query': question,
                'error': str(e)
            }
        finally:
            self.metrics.end_timing(timing_id)
    
    async def generate_search_queries(self, question: str, context: Dict[str, Any] = None) -> List[str]:
        """Generate optimized search queries from different angles"""
        timing_id = self.metrics.start_timing("query_generation")
        
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

            context_info = ""
            if context:
                context_info = f"\nAdditional Context: {context}"

            user_prompt = f"""Original Question: {question}{context_info}

Generate 4-5 comprehensive search queries that approach this question from different angles. Focus on finding authoritative, data-rich sources."""

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ]
            
            response = await self._make_api_call(messages, "query_generation", timeout=30.0)
            response_text = response.choices[0].message.content
            
            # Extract queries from response
            queries = []
            query_pattern = r'Query\d*="([^"]+)"'
            matches = re.findall(query_pattern, response_text)
            
            if matches:
                queries = matches
                logger.info(f"✅ Generated {len(queries)} search queries")
                for i, query in enumerate(queries, 1):
                    logger.info(f"  {i}. {query}")
            else:
                # Fallback: use the original question
                queries = [question]
                logger.warning("⚠️ Could not extract queries from response, using original question")
            
            self.metrics.increment_counter("query_generations")
            self.metrics.record_histogram_value("queries_generated", len(queries))
            
            return queries
                
        except Exception as e:
            logger.error(f"❌ Query generation failed: {e}")
            self.metrics.increment_counter("query_generation_errors")
            return [question]
        finally:
            self.metrics.end_timing(timing_id)
    
    async def analyze_content_relevance(self, question: str, contents: List[ExtractedContent]) -> AnalysisResult:
        """Analyze content relevance and provide comprehensive answer with streaming"""
        timing_id = self.metrics.start_timing("content_analysis")
        
        try:
            logger.info("🧠 Starting comprehensive content analysis")
            
            # Prepare content for analysis with quality indicators
            content_summaries = []
            for i, content in enumerate(contents):
                if content.success:
                    domain_info = content.domain_info or {}
                    quality_score = domain_info.get('quality_score', 5)
                    source_type = domain_info.get('source_type', 'unknown')
                    cache_status = "CACHED" if content.from_cache else "LIVE"
                    
                    summary = f"""
Source {i+1}: {content.title} [{cache_status}]
URL: {content.url}
Quality Score: {quality_score}/10
Source Type: {source_type}
Extraction Method: {content.method}
Content Preview: {content.content[:1500]}...
Word Count: {content.word_count}
"""
                else:
                    summary = f"""
Source {i+1}: {content.title} (Extraction Failed)
URL: {content.url}
Error: {content.error or 'Unknown error'}
"""
                content_summaries.append(summary)
            
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

            user_prompt = f"""Original Research Question: {question}

Extracted Content from {len(contents)} Web Sources:
{''.join(content_summaries)}

Please provide a comprehensive analysis including:
1. Individual source relevance scores and assessment
2. Synthesized answer to the original question
3. Gap analysis - what's missing
4. Overall quality assessment
5. Your final overall relevance score (1-10)"""

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ]
            
            logger.info("🧠 Starting DeepSeek Reasoning Analysis...")
            
            # Use streaming for better user experience
            response = await self._make_api_call(messages, "complex_analysis", stream=True, timeout=90.0)
            
            # Process streaming response
            reasoning_content = ""
            analysis_content = ""
            
            async for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    delta = choice.delta
                    
                    # Handle reasoning content streaming
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        reasoning_content += delta.reasoning_content
                    
                    # Handle regular content streaming
                    if hasattr(delta, 'content') and delta.content:
                        analysis_content += delta.content
            
            # Extract overall relevance score
            overall_relevance_score = 0
            score_match = re.search(r'OVERALL_RELEVANCE_SCORE:\s*(\d+)', analysis_content)
            if score_match:
                overall_relevance_score = int(score_match.group(1))
                logger.info(f"📊 Extracted relevance score: {overall_relevance_score}/10")
            else:
                logger.warning("⚠️ Could not extract overall relevance score")
            
            result = AnalysisResult(
                original_question=question,
                analysis_content=analysis_content,
                reasoning_content=reasoning_content,
                relevance_score=overall_relevance_score,
                sources_analyzed=len(contents),
                model=getattr(response, 'model', 'deepseek-reasoner'),
                timestamp=datetime.utcnow()
            )
            
            # Record metrics
            self.metrics.increment_counter("content_analyses")
            self.metrics.record_histogram_value("analysis_relevance_score", overall_relevance_score)
            self.metrics.record_histogram_value("sources_analyzed", len(contents))
            
            logger.info(f"✅ Content analysis completed with relevance score: {overall_relevance_score}/10")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Content analysis failed: {e}")
            self.metrics.increment_counter("content_analysis_errors")
            
            return AnalysisResult(
                original_question=question,
                analysis_content=f"Analysis failed: {str(e)}",
                sources_analyzed=len(contents),
                timestamp=datetime.utcnow(),
                error=str(e)
            )
        finally:
            self.metrics.end_timing(timing_id)
    
    async def generate_followup_queries(self, question: str, previous_analysis: AnalysisResult) -> List[str]:
        """Generate follow-up queries based on gaps identified in previous analysis"""
        timing_id = self.metrics.start_timing("followup_query_generation")
        
        try:
            logger.info("🎯 Generating follow-up queries based on identified gaps")
            
            system_message = """You are an expert research strategist. Based on a previous research iteration and its gaps, generate targeted follow-up search queries to fill the missing information.

Instructions:
1. Analyze the previous analysis to identify specific gaps and weaknesses
2. Generate 3-4 targeted search queries that address these gaps
3. Focus on missing data types, unexplored angles, or contradictory information
4. Make queries specific and different from previous searches
5. Format response as: Query1="...", Query2="...", Query3="...", etc."""

            user_prompt = f"""Original Question: {question}

Previous Analysis Results:
- Relevance Score: {previous_analysis.relevance_score}/10
- Analysis Summary: {previous_analysis.analysis_content[:1000]}...

Generate 3-4 targeted follow-up search queries to address the gaps and improve relevance."""

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ]
            
            response = await self._make_api_call(messages, "query_generation", timeout=30.0)
            response_text = response.choices[0].message.content
            
            # Extract queries from response
            queries = []
            query_pattern = r'Query\d*="([^"]+)"'
            matches = re.findall(query_pattern, response_text)
            
            if matches:
                queries = matches
                logger.info(f"✅ Generated {len(queries)} follow-up queries")
                for i, query in enumerate(queries, 1):
                    logger.info(f"  {i}. {query}")
            else:
                # Fallback: modify original question
                queries = [f"{question} latest data", f"{question} market analysis", f"{question} revenue statistics"]
                logger.warning("⚠️ Could not extract follow-up queries, using modified versions")
            
            self.metrics.increment_counter("followup_query_generations")
            
            return queries
                
        except Exception as e:
            logger.error(f"❌ Follow-up query generation failed: {e}")
            self.metrics.increment_counter("followup_query_generation_errors")
            return [f"{question} additional information"]
        finally:
            self.metrics.end_timing(timing_id)
    
    def get_reasoning_stats(self) -> Dict[str, Any]:
        """Get AI reasoning service statistics"""
        return {
            'api_configured': bool(self.api_key),
            'primary_chat_model': self.model_settings['primary_chat_model'],
            'primary_reasoning_model': self.model_settings['primary_reasoning_model'],
            'fallback_models': self.model_settings['fallback_models'],
            'model_selection_strategy': self.model_settings['model_selection_strategy']
        }


# Factory function for creating AI reasoning service
def create_ai_reasoning_service() -> IAIReasoningService:
    """Create and return an AI reasoning service instance"""
    config = get_config_manager()
    
    if config.is_service_configured('deepseek'):
        return DeepSeekReasoningService()
    else:
        logger.error("DeepSeek not configured")
        raise ValueError("DeepSeek API credentials not configured")
