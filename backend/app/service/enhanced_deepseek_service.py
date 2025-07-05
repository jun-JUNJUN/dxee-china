#!/usr/bin/env python3
"""
Enhanced DeepSeek Service - Integration with Modular Research System
This service integrates the new modular research system with the existing backend architecture
"""

import os
import json
import asyncio
import logging
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional
from ..research import (
    create_research_system, ResearchQuery, ResearchResult,
    get_config_manager, get_metrics_collector, validate_system_setup
)
from .message_formatter import MessageFormatter

logger = logging.getLogger(__name__)


class EnhancedDeepSeekService:
    """
    Enhanced DeepSeek service with modular research system integration
    """
    
    def __init__(self, input_queue, output_queue):
        self.input_queue = input_queue
        self.output_queue = output_queue
        
        # Initialize message formatter
        self.formatter = MessageFormatter()
        
        # Initialize research system
        self.research_orchestrator = None
        self.config = None
        self.metrics = None
        self._initialize_research_system()
        
        logger.info("Enhanced DeepSeek service initialized with modular research system")
    
    def _initialize_research_system(self):
        """Initialize the modular research system"""
        try:
            # Validate system setup
            validation = validate_system_setup()
            if not validation['valid']:
                logger.error(f"Research system validation failed: {validation['issues']}")
                for warning in validation['warnings']:
                    logger.warning(f"Research system warning: {warning}")
            
            # Create research system
            self.research_orchestrator, self.config, self.metrics = create_research_system()
            
            # Log configuration summary
            config_summary = self.config.get_configuration_summary()
            logger.info(f"Research system initialized:")
            logger.info(f"  - Services configured: {config_summary['services_configured']}")
            logger.info(f"  - Default search mode: {config_summary['research_settings']['default_search_mode']}")
            logger.info(f"  - Cache enabled: {config_summary['research_settings']['cache_enabled']}")
            
        except Exception as e:
            logger.error(f"Failed to initialize research system: {e}")
            logger.error(traceback.format_exc())
            raise
    
    async def process_message(self, message_data):
        """
        Process a message through the enhanced research system
        
        Args:
            message_data (dict): Message data containing query and metadata
            
        Returns:
            dict: Response with research results
        """
        try:
            query_text = message_data.get('message', '')
            chat_id = message_data.get('chat_id')
            message_id = message_data.get('message_id', 'unknown')
            search_mode = message_data.get('search_mode', 'standard')
            
            logger.info(f"Processing enhanced message: {query_text[:50]}... (ID: {message_id}, Chat: {chat_id}, Mode: {search_mode})")
            
            # Handle different search modes
            if search_mode in ["enhanced", "googleweb", "deep"]:
                return await self._handle_enhanced_research(query_text, chat_id, message_id, search_mode)
            else:
                return await self._handle_standard_research(query_text, chat_id, message_id, search_mode)
            
        except Exception as e:
            logger.error(f"Error processing enhanced message: {e}")
            logger.error(traceback.format_exc())
            
            # Return error response
            return {
                'chat_id': chat_id,
                'message_id': message_id,
                'message': f'Error processing your query: {str(e)}',
                'timestamp': datetime.utcnow().isoformat(),
                'search_results': [],
                'error': True
            }
    
    async def _handle_enhanced_research(self, query_text: str, chat_id: str, 
                                      message_id: str, search_mode: str) -> Dict[str, Any]:
        """Handle enhanced research using the modular research system"""
        try:
            logger.info(f"Starting enhanced research for: {query_text}")
            
            # Create research query
            research_settings = self.config.get_research_settings()
            
            research_query = ResearchQuery(
                question=query_text,
                query_id=f"{chat_id}_{message_id}",
                timestamp=datetime.utcnow(),
                search_mode=search_mode,
                target_relevance=research_settings['default_target_relevance'],
                max_iterations=research_settings['default_max_iterations']
            )
            
            # Conduct research
            research_result = await self.research_orchestrator.conduct_research(research_query)
            
            if research_result.success:
                # Format successful research result
                if research_result.research_type == 'direct_answer':
                    # Direct answer without web search
                    analysis_content = research_result.direct_answer
                else:
                    # Web search research
                    analysis_content = research_result.analysis.analysis_content if research_result.analysis else "No analysis available"
                
                formatted_message = self.formatter.format_message(analysis_content, "markdown")
                
                # Prepare search results for display
                search_results = []
                if research_result.search_results:
                    for result in research_result.search_results:
                        search_results.append({
                            'title': result.title,
                            'content': result.snippet,
                            'url': result.url,
                            'snippet': result.snippet,
                            'relevance_score': result.relevance_score
                        })
                
                # Create enhanced response
                response = {
                    'chat_id': chat_id,
                    'message_id': message_id,
                    'message': analysis_content,
                    'formatted_message': formatted_message,
                    'timestamp': datetime.utcnow().isoformat(),
                    'search_results': search_results,
                    'enhanced_research_data': {
                        'research_type': research_result.research_type,
                        'iterations_completed': len(research_result.iterations or []),
                        'final_relevance_score': research_result.metrics.get('final_relevance_score', 0) if research_result.metrics else 0,
                        'target_achieved': research_result.metrics.get('target_achieved', False) if research_result.metrics else False,
                        'sources_analyzed': len(research_result.extracted_contents or []),
                        'successful_extractions': research_result.metrics.get('search_metrics', {}).get('successful_extractions', 0) if research_result.metrics else 0,
                        'cache_performance': research_result.metrics.get('cache_performance', {}) if research_result.metrics else {},
                        'reasoning_content': research_result.analysis.reasoning_content if research_result.analysis else None
                    }
                }
                
                logger.info(f"Enhanced research completed successfully for chat_id: {chat_id}")
                return response
                
            else:
                # Research failed
                error_msg = research_result.error or "Enhanced research failed"
                logger.warning(f"Enhanced research failed: {error_msg}")
                
                # Fallback to standard processing
                return await self._handle_standard_research(query_text, chat_id, message_id, "standard")
                
        except Exception as e:
            logger.error(f"Enhanced research failed: {e}")
            logger.error(traceback.format_exc())
            
            # Fallback to standard processing
            return await self._handle_standard_research(query_text, chat_id, message_id, "standard")
    
    async def _handle_standard_research(self, query_text: str, chat_id: str, 
                                      message_id: str, search_mode: str) -> Dict[str, Any]:
        """Handle standard research using basic AI reasoning"""
        try:
            logger.info(f"Using standard research for: {query_text}")
            
            # Use the AI reasoning service directly for simple queries
            ai_reasoning = self.research_orchestrator.ai_reasoning_service
            
            # Check if web search is needed
            necessity_check = await ai_reasoning.check_web_search_necessity(query_text)
            
            if not necessity_check['web_search_needed']:
                # Direct answer
                raw_message = necessity_check['direct_answer']
                formatted_message = self.formatter.format_message(raw_message, "markdown")
                
                return {
                    'chat_id': chat_id,
                    'message_id': message_id,
                    'message': raw_message,
                    'formatted_message': formatted_message,
                    'timestamp': datetime.utcnow().isoformat(),
                    'search_results': [],
                    'research_type': 'direct_answer'
                }
            else:
                # Simple web search and analysis
                search_query = necessity_check['search_query']
                
                # Perform basic search
                web_search = self.research_orchestrator.web_search_service
                search_results = await web_search.search(search_query, num_results=3)
                
                if search_results:
                    # Extract content from top results
                    content_extractor = self.research_orchestrator.content_extractor
                    urls = [result.url for result in search_results[:2]]  # Limit to top 2 for speed
                    
                    extracted_contents = []
                    for url in urls:
                        content = await content_extractor.extract_content(url)
                        if content.success:
                            extracted_contents.append(content)
                    
                    if extracted_contents:
                        # Analyze content
                        analysis = await ai_reasoning.analyze_content_relevance(query_text, extracted_contents)
                        raw_message = analysis.analysis_content
                    else:
                        raw_message = f"Found search results but could not extract content. Search query used: {search_query}"
                else:
                    raw_message = f"No search results found for: {search_query}"
                
                formatted_message = self.formatter.format_message(raw_message, "markdown")
                
                # Prepare search results for display
                display_results = []
                for result in search_results:
                    display_results.append({
                        'title': result.title,
                        'content': result.snippet,
                        'url': result.url,
                        'snippet': result.snippet
                    })
                
                return {
                    'chat_id': chat_id,
                    'message_id': message_id,
                    'message': raw_message,
                    'formatted_message': formatted_message,
                    'timestamp': datetime.utcnow().isoformat(),
                    'search_results': display_results,
                    'research_type': 'simple_web_search'
                }
                
        except Exception as e:
            logger.error(f"Standard research failed: {e}")
            
            # Final fallback - basic AI response
            raw_message = f"I encountered an issue while researching your question: {str(e)}. Let me provide what I can based on my knowledge: {query_text}"
            formatted_message = self.formatter.format_message(raw_message, "markdown")
            
            return {
                'chat_id': chat_id,
                'message_id': message_id,
                'message': raw_message,
                'formatted_message': formatted_message,
                'timestamp': datetime.utcnow().isoformat(),
                'search_results': [],
                'error': True
            }
    
    async def process_message_stream(self, message_data, stream_queue):
        """
        Process a message with streaming support
        
        Args:
            message_data (dict): Message data containing query and metadata
            stream_queue: Queue to send streaming chunks to
        """
        try:
            query_text = message_data.get('message', '')
            chat_id = message_data.get('chat_id')
            message_id = message_data.get('message_id', 'unknown')
            search_mode = message_data.get('search_mode', 'standard')
            
            logger.info(f"Processing streaming enhanced message: {query_text[:50]}... (ID: {message_id}, Chat: {chat_id}, Mode: {search_mode})")
            
            # For now, use the regular processing and send as complete chunk
            # In the future, this could be enhanced to use the streaming research capability
            result = await self.process_message(message_data)
            
            # Send progress updates
            await stream_queue.put({
                'chat_id': chat_id,
                'message_id': message_id,
                'type': 'progress',
                'content': 'Starting research...',
                'timestamp': datetime.utcnow().isoformat()
            })
            
            # Send final result
            await stream_queue.put({
                'chat_id': chat_id,
                'message_id': message_id,
                'type': 'complete',
                'content': result.get('message', ''),
                'formatted_content': result.get('formatted_message', ''),
                'reasoning_content': result.get('enhanced_research_data', {}).get('reasoning_content'),
                'timestamp': datetime.utcnow().isoformat(),
                'search_results': result.get('search_results', []),
                'enhanced_data': result.get('enhanced_research_data', {})
            })
            
            logger.info(f"Streaming completed for chat_id: {chat_id}")
            
        except Exception as e:
            logger.error(f"Error in streaming enhanced message: {e}")
            
            # Send error chunk
            await stream_queue.put({
                'chat_id': chat_id,
                'message_id': message_id,
                'type': 'error',
                'content': f'Error processing your query: {str(e)}',
                'timestamp': datetime.utcnow().isoformat()
            })
    
    async def start_processing(self, stream_queues=None):
        """
        Start processing messages from the input queue
        
        Args:
            stream_queues: Optional dictionary of streaming queues
        """
        logger.info("Starting enhanced message processing")
        
        while True:
            try:
                # Check if there are messages to process
                if not self.input_queue:
                    await asyncio.sleep(0.1)
                    continue
                
                # Process messages
                while self.input_queue:
                    try:
                        message_data = self.input_queue.pop(0)
                        
                        # Check if this is a streaming request
                        message_id = message_data.get('message_id', 'unknown')
                        if stream_queues and message_id in stream_queues:
                            # Process with streaming
                            await self.process_message_stream(message_data, stream_queues[message_id])
                        else:
                            # Process normally
                            response = await self.process_message(message_data)
                            self.output_queue.append(response)
                            
                    except Exception as e:
                        logger.error(f"Error processing individual message: {e}")
                        logger.error(traceback.format_exc())
                        continue
                
                # Brief pause before checking for more messages
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in message processing loop: {e}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(1)  # Longer pause on error
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get the status of the enhanced research system"""
        try:
            config_summary = self.config.get_configuration_summary()
            metrics_summary = self.metrics.get_metrics_summary()
            
            return {
                'system_initialized': self.research_orchestrator is not None,
                'configuration': config_summary,
                'metrics': metrics_summary,
                'services_status': {
                    'deepseek_configured': self.config.is_service_configured('deepseek'),
                    'google_configured': self.config.is_service_configured('google'),
                    'mongodb_configured': self.config.is_service_configured('mongodb')
                }
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                'system_initialized': False,
                'error': str(e)
            }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get a comprehensive performance report"""
        try:
            return self.metrics.get_performance_report()
        except Exception as e:
            logger.error(f"Error getting performance report: {e}")
            return {'error': str(e)}
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.research_orchestrator:
                await self.research_orchestrator.cleanup()
            logger.info("Enhanced DeepSeek service cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
