import os
import json
import asyncio
import logging
import traceback
import requests
import re
from datetime import datetime
from openai import OpenAI
from openai import AsyncOpenAI
from typing import List, Dict, Any
from .message_formatter import MessageFormatter

# Web scraping libraries for Deep Search
try:
    from bs4 import BeautifulSoup
    import newspaper
    from newspaper import Article
    from readability import Document
except ImportError as e:
    logging.warning(f"Web scraping libraries not available: {e}")
    BeautifulSoup = None
    newspaper = None
    Article = None
    Document = None

# Get logger
logger = logging.getLogger(__name__)

class DeepSeekService:
    """
    Service for interacting with the DeepSeek API using OpenAI SDK
    """
    def __init__(self, input_queue, output_queue):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.api_key = os.environ.get('DEEPSEEK_API_KEY', '')
        self.api_url = os.environ.get('DEEPSEEK_API_URL', 'https://api.deepseek.com')
        
        # Initialize OpenAI clients
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_url)
        self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.api_url)
        
        # Initialize message formatter
        self.formatter = MessageFormatter()
        
        logger.info(f"DeepSeek service initialized with API URL: {self.api_url}")
        
        if not self.api_key:
            logger.warning("DEEPSEEK_API_KEY environment variable not set")
            logger.warning("Please set DEEPSEEK_API_KEY in your .env file")
        
        # Initialize Google Search API credentials for Deep Search with Google Search
        self.google_api_key = os.environ.get('GOOGLE_API_KEY', '')
        self.google_cse_id = os.environ.get('GOOGLE_CSE_ID', '')
        
        if not self.google_api_key or not self.google_cse_id:
            logger.warning("Google Search API credentials not configured for Deep Search with Google Search")
            logger.warning("Please set GOOGLE_API_KEY and GOOGLE_CSE_ID in your .env file")
    
    def chat_completion(self, query, system_message="You are a helpful assistant", max_retries=6, search_mode="search"):
            """
            Get a chat completion from the DeepSeek API using the synchronous client
            
            Args:
                query (str): The user query
                system_message (str): The system message to set the assistant's behavior
                max_retries (int): Maximum number of retry attempts for API calls
                search_mode (str): The search mode ('search', 'deep', 'doubao')
                
            Returns:
                dict: Chat completion response from DeepSeek
            """
            # Select model based on search mode
            if search_mode == "deep":
                models_to_try = ["deepseek-reasoner", "deepseek-chat"]
            else:
                models_to_try = ["deepseek-chat", "deepseek-reasoner"]
            
            # Track which model we're using
            current_model = models_to_try[0]
            
            for retry in range(max_retries):
                try:
                    logger.info(f"Sending chat completion request to DeepSeek API (attempt {retry+1}/{max_retries})")
                    logger.info(f"Using model: {current_model}")
                    logger.info(f"Query preview: {query[:30]}...")
                    
                    # Add timeout parameters to avoid hanging requests
                    response = self.client.chat.completions.create(
                        model=current_model,
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": query},
                        ],
                        stream=False,
                        timeout=30.0  # Add timeout parameter
                    )
                    
                    logger.info("Chat completion request successful")
                    return {
                        "model": response.model,
                        "id": response.id,
                        "created": response.created,
                        "content": response.choices[0].message.content
                    }
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error on attempt {retry+1}/{max_retries}: {error_msg}")
                    
                    # Check if it's a 500 Internal Server Error
                    if "500" in error_msg and "Internal Server Error" in error_msg:
                        logger.warning("Received 500 Internal Server Error from DeepSeek API")
                        
                        # If we have more models to try and this isn't the last retry
                        if retry < max_retries - 1:
                            # Try a different model if available
                            model_index = min(retry + 1, len(models_to_try) - 1)
                            current_model = models_to_try[model_index]
                            logger.info(f"Switching to alternative model: {current_model}")
                            
                            # Wait before retrying to avoid overwhelming the API
                            import time
                            time.sleep(1 * (retry + 1))
                            continue
                    
                    # For the last retry or non-500 errors, log the full traceback
                    if retry == max_retries - 1:
                        logger.error("All retry attempts failed")
                        logger.error(traceback.format_exc())
                        
                    # If it's not the last retry, wait and try again
                    if retry < max_retries - 1:
                        import time
                        time.sleep(1 * (retry + 1))
                        continue
                    
                    # Return error response after all retries fail
                    return {"content": f"Error: {error_msg}. Please try again later."}
    
    async def async_chat_completion(self, query, system_message="You are a helpful assistant", max_retries=6, search_mode="search"):
            """
            Get a chat completion from the DeepSeek API using the asynchronous client
            
            Args:
                query (str): The user query
                system_message (str): The system message to set the assistant's behavior
                max_retries (int): Maximum number of retry attempts for API calls
                search_mode (str): The search mode ('search', 'deep', 'doubao')
                
            Returns:
                dict: Chat completion response from DeepSeek
            """
            # Select model based on search mode
            if search_mode == "deep":
                models_to_try = ["deepseek-reasoner", "deepseek-chat"]
            else:
                models_to_try = ["deepseek-chat", "deepseek-reasoner"]
            
            # Track which model we're using
            current_model = models_to_try[0]
            
            for retry in range(max_retries):
                try:
                    logger.info(f"Sending async chat completion request to DeepSeek API (attempt {retry+1}/{max_retries})")
                    logger.info(f"Using model: {current_model}")
                    logger.info(f"Query preview: {query[:30]}...")
                    
                    # Add timeout parameters to avoid hanging requests
                    response = await self.async_client.chat.completions.create(
                        model=current_model,
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": query},
                        ],
                        stream=False,
                        timeout=30.0  # Add timeout parameter
                    )
                    
                    logger.info("Async chat completion request successful")
                    return {
                        "model": response.model,
                        "id": response.id,
                        "created": response.created,
                        "content": response.choices[0].message.content
                    }
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error on attempt {retry+1}/{max_retries}: {error_msg}")
                    
                    # Check if it's a 500 Internal Server Error
                    if "500" in error_msg and "Internal Server Error" in error_msg:
                        logger.warning("Received 500 Internal Server Error from DeepSeek API")
                        
                        # If we have more models to try and this isn't the last retry
                        if retry < max_retries - 1:
                            # Try a different model if available
                            model_index = min(retry + 1, len(models_to_try) - 1)
                            current_model = models_to_try[model_index]
                            logger.info(f"Switching to alternative model: {current_model}")
                            
                            # Wait before retrying to avoid overwhelming the API
                            await asyncio.sleep(1 * (retry + 1))
                            continue
                    
                    # For the last retry or non-500 errors, log the full traceback
                    if retry == max_retries - 1:
                        logger.error("All retry attempts failed")
                        logger.error(traceback.format_exc())
                        
                    # If it's not the last retry, wait and try again
                    if retry < max_retries - 1:
                        await asyncio.sleep(1 * (retry + 1))
                        continue
                    
                    # Return error response after all retries fail
                    return {"content": f"Error: {error_msg}. Please try again later."}
    
    async def async_chat_completion_stream(self, query, system_message="You are a helpful assistant", chat_history=None, max_retries=6, search_mode="search"):
            """
            Get a streaming chat completion from the DeepSeek API using the asynchronous client
            
            Args:
                query (str): The user query
                system_message (str): The system message to set the assistant's behavior
                chat_history (list): Previous chat messages for context
                max_retries (int): Maximum number of retry attempts for API calls
                search_mode (str): The search mode ('search', 'deep', 'doubao')
                
            Yields:
                dict: Streaming chunks from DeepSeek API
            """
            # Select model based on search mode
            if search_mode == "deep":
                models_to_try = ["deepseek-reasoner", "deepseek-chat"]
            else:
                models_to_try = ["deepseek-chat", "deepseek-reasoner"]
            
            # Track which model we're using
            current_model = models_to_try[0]
            
            for retry in range(max_retries):
                try:
                    logger.info(f"Sending async streaming chat completion request to DeepSeek API (attempt {retry+1}/{max_retries})")
                    logger.info(f"Using model: {current_model}")
                    logger.info(f"Query preview: {query[:30]}...")
                    
                    # Build conversation messages with chat history
                    messages = [{"role": "system", "content": system_message}]
                    
                    # Add chat history if provided
                    if chat_history:
                        for msg in chat_history:
                            messages.append({
                                "role": msg["type"],  # 'user' or 'assistant'
                                "content": msg["content"]
                            })
                        logger.info(f"Added {len(chat_history)} messages from chat history")
                    
                    # Add current user query
                    messages.append({"role": "user", "content": query})
                    
                    # Add timeout parameters to avoid hanging requests
                    stream = await self.async_client.chat.completions.create(
                        model=current_model,
                        messages=messages,
                        stream=True,
                        timeout=30.0  # Add timeout parameter
                    )
                    
                    logger.info("Async streaming chat completion request successful")
                    
                    reasoning_content = ""
                    content = ""
                    
                    async for chunk in stream:
                        # Handle reasoning content (specific to deepseek-reasoner model)
                        if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                            reasoning_content += chunk.choices[0].delta.reasoning_content
                            yield {
                                "model": chunk.model if hasattr(chunk, 'model') else current_model,
                                "id": chunk.id if hasattr(chunk, 'id') else None,
                                "created": chunk.created if hasattr(chunk, 'created') else None,
                                "reasoning_content": chunk.choices[0].delta.reasoning_content,
                                "content_type": "reasoning",
                                "finished": False
                            }
                        # Handle regular content
                        elif chunk.choices[0].delta.content:
                            content += chunk.choices[0].delta.content
                            yield {
                                "model": chunk.model if hasattr(chunk, 'model') else current_model,
                                "id": chunk.id if hasattr(chunk, 'id') else None,
                                "created": chunk.created if hasattr(chunk, 'created') else None,
                                "content": chunk.choices[0].delta.content,
                                "content_type": "answer",
                                "finished": False
                            }
                    
                    # Send finish signal with complete reasoning and content
                    yield {
                        "model": current_model,
                        "content": content,
                        "reasoning_content": reasoning_content,
                        "reasoning_chunks": reasoning_content.count('\n') + 1 if reasoning_content else 0,
                        "answer_chunks": content.count(' ') + 1 if content else 0,
                        "reasoning_length": len(reasoning_content) if reasoning_content else 0,
                        "answer_length": len(content) if content else 0,
                        "finished": True
                    }
                    return
                    
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error on attempt {retry+1}/{max_retries}: {error_msg}")
                    
                    # Check if it's a 500 Internal Server Error
                    if "500" in error_msg and "Internal Server Error" in error_msg:
                        logger.warning("Received 500 Internal Server Error from DeepSeek API")
                        
                        # If we have more models to try and this isn't the last retry
                        if retry < max_retries - 1:
                            # Try a different model if available
                            model_index = min(retry + 1, len(models_to_try) - 1)
                            current_model = models_to_try[model_index]
                            logger.info(f"Switching to alternative model: {current_model}")
                            
                            # Wait before retrying to avoid overwhelming the API
                            await asyncio.sleep(1 * (retry + 1))
                            continue
                    
                    # For the last retry or non-500 errors, log the full traceback
                    if retry == max_retries - 1:
                        logger.error("All retry attempts failed")
                        logger.error(traceback.format_exc())
                        
                    # If it's not the last retry, wait and try again
                    if retry < max_retries - 1:
                        await asyncio.sleep(1 * (retry + 1))
                        continue
                    
                    # Return error response after all retries fail
                    yield {
                        "content": f"Error: {error_msg}. Please try again later.",
                        "finished": True,
                        "error": True
                    }
                    return
    
    async def process_message(self, message_data):
        """
        Process a message through the DeepSeek API
        
        Args:
            message_data (dict): Message data containing query and metadata
            
        Returns:
            dict: Response with chat completion
        """
        try:
            query = message_data.get('message', '')
            chat_id = message_data.get('chat_id')
            message_id = message_data.get('message_id', 'unknown')
            search_mode = message_data.get('search_mode', 'search')
            
            logger.info(f"Processing message: {query[:30]}... (ID: {message_id}, Chat: {chat_id}, Mode: {search_mode})")
            
            # Handle Deep Search with Google Search for "googleweb" mode
            if search_mode == "googleweb":
                logger.info("Using Deep Search with Google Search functionality")
                deep_search_results = await self.conduct_deep_search_with_google(query)
                
                if deep_search_results.get('success'):
                    # Format the analysis message
                    analysis_content = deep_search_results.get('analysis', 'No analysis available')
                    formatted_message = self.formatter.format_message(analysis_content, "markdown")
                    
                    # Prepare search results for display
                    search_results = []
                    for result in deep_search_results.get('search_results', []):
                        search_results.append({
                            'title': result.get('title', ''),
                            'content': result.get('snippet', ''),
                            'url': result.get('link', ''),
                            'snippet': result.get('snippet', '')
                        })
                    
                    # Create response data with Deep Search results
                    response = {
                        'chat_id': chat_id,
                        'message_id': message_id,
                        'message': analysis_content,
                        'formatted_message': formatted_message,
                        'timestamp': datetime.utcnow().isoformat(),
                        'search_results': search_results,
                        'deep_search_data': {
                            'research_type': deep_search_results.get('research_type', 'web_search_research'),
                            'search_query': deep_search_results.get('search_query', ''),
                            'extracted_contents': deep_search_results.get('extracted_contents', []),
                            'sources_analyzed': len(deep_search_results.get('extracted_contents', [])),
                            'successful_extractions': sum(1 for c in deep_search_results.get('extracted_contents', []) if c.get('success'))
                        }
                    }
                    
                    logger.info(f"Deep Search with Google Search completed successfully for chat_id: {chat_id}")
                    return response
                else:
                    # Deep Search failed, fall back to regular processing
                    logger.warning("Deep Search with Google Search failed, falling back to regular processing")
                    error_msg = deep_search_results.get('error', 'Deep Search failed')
                    fallback_query = f"I attempted to search the web for information about '{query}' but encountered an issue: {error_msg}. Let me provide what I can based on my knowledge: {query}"
                    completion_response = await self.async_chat_completion(fallback_query, search_mode="search")
            else:
                # Regular processing for other search modes
                completion_response = await self.async_chat_completion(query, search_mode=search_mode)
            
            # Format the response message for non-googleweb modes or fallback
            if search_mode != "googleweb" or not deep_search_results.get('success'):
                raw_message = completion_response.get('content', 'No response from DeepSeek API.')
                formatted_message = self.formatter.format_message(raw_message, "markdown")
                
                # Create response with message_id to match the specific request
                response = {
                    'chat_id': chat_id,
                    'message_id': message_id,  # Include message_id for matching
                    'message': raw_message,  # Keep raw message for backward compatibility
                    'formatted_message': formatted_message,  # Add formatted version
                    'timestamp': datetime.utcnow().isoformat(),
                    'search_results': []  # Keep this for backward compatibility
                }
            
            logger.info(f"Response created for chat_id: {chat_id}")
            return response
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            logger.error(traceback.format_exc())
            
            # Return error response with message_id
            return {
                'chat_id': chat_id,
                'message_id': message_id,  # Include message_id for matching
                'message': f'Error processing your query: {str(e)}',
                'timestamp': datetime.utcnow().isoformat(),
                'search_results': []
            }
    
    async def process_message_stream(self, message_data, stream_queue):
        """
        Process a message through the DeepSeek API with streaming
        
        Args:
            message_data (dict): Message data containing query and metadata
            stream_queue: Queue to send streaming chunks to
        """
        try:
            query = message_data.get('message', '')
            chat_id = message_data.get('chat_id')
            message_id = message_data.get('message_id', 'unknown')
            search_mode = message_data.get('search_mode', 'search')
            
            logger.info(f"Processing streaming message: {query[:30]}... (ID: {message_id}, Chat: {chat_id}, Mode: {search_mode})")
            
            # Get chat history from message data
            chat_history = message_data.get('chat_history', [])
            
            # Process streaming chat completion with search mode
            accumulated_content = ""
            accumulated_reasoning = ""
            
            async for chunk in self.async_chat_completion_stream(query, chat_history=chat_history, search_mode=search_mode):
                if chunk.get('error'):
                    # Send error chunk
                    await stream_queue.put({
                        'chat_id': chat_id,
                        'message_id': message_id,
                        'type': 'error',
                        'content': chunk.get('content', 'Unknown error'),
                        'timestamp': datetime.utcnow().isoformat()
                    })
                    return
                
                if chunk.get('finished'):
                    # Format the complete accumulated content
                    formatted_message = self.formatter.format_message(accumulated_content, "markdown")
                    formatted_reasoning = self.formatter.format_message(accumulated_reasoning, "markdown") if accumulated_reasoning else None
                    
                    # Send final chunk with complete message and reasoning
                    await stream_queue.put({
                        'chat_id': chat_id,
                        'message_id': message_id,
                        'type': 'complete',
                        'content': accumulated_content,
                        'reasoning_content': accumulated_reasoning,
                        'formatted_content': formatted_message,
                        'formatted_reasoning': formatted_reasoning,
                        'timestamp': datetime.utcnow().isoformat(),
                        'search_results': []
                    })
                    logger.info(f"Streaming completed for chat_id: {chat_id}")
                    return
                else:
                    # Handle reasoning content chunks
                    if chunk.get('content_type') == 'reasoning':
                        reasoning_content = chunk.get('reasoning_content', '')
                        accumulated_reasoning += reasoning_content
                        await stream_queue.put({
                            'chat_id': chat_id,
                            'message_id': message_id,
                            'type': 'reasoning_chunk',
                            'reasoning_content': reasoning_content,
                            'timestamp': datetime.utcnow().isoformat()
                        })
                    # Handle regular content chunks
                    elif chunk.get('content_type') == 'answer':
                        content = chunk.get('content', '')
                        accumulated_content += content
                        await stream_queue.put({
                            'chat_id': chat_id,
                            'message_id': message_id,
                            'type': 'chunk',
                            'content': content,
                            'timestamp': datetime.utcnow().isoformat()
                        })
                    # Handle legacy chunks (for backward compatibility)
                    else:
                        content = chunk.get('content', '')
                        accumulated_content += content
                        await stream_queue.put({
                            'chat_id': chat_id,
                            'message_id': message_id,
                            'type': 'chunk',
                            'content': content,
                            'timestamp': datetime.utcnow().isoformat()
                        })
                    
        except Exception as e:
            logger.error(f"Error processing streaming message: {e}")
            logger.error(traceback.format_exc())
            
            # Send error response
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
            stream_queues (dict): Dictionary mapping chat_id -> stream queue for streaming responses
        """
        logger.info("Starting DeepSeek service processing loop")
        
        while True:
            try:
                # Check if there are messages in the input queue
                queue_size = len(self.input_queue) if self.input_queue else 0
                
                if queue_size > 0:
                    # Get the next message
                    message_data = self.input_queue.pop(0)
                    chat_id = message_data.get('chat_id', 'unknown')
                    message_id = message_data.get('message_id', 'unknown')
                    message_preview = message_data.get('message', '')[:30] if message_data.get('message') else 'empty'
                    is_streaming = message_data.get('streaming', False)
                    
                    logger.info(f"Processing message from queue: {message_preview}... (Chat: {chat_id}, Streaming: {is_streaming})")
                    logger.info(f"Queue size before processing: {queue_size}, after: {len(self.input_queue)}")
                    
                    try:
                        start_time = datetime.utcnow()
                        
                        if is_streaming and stream_queues and chat_id in stream_queues:
                            # Process with streaming
                            await self.process_message_stream(message_data, stream_queues[chat_id])
                        else:
                            # Process normally (non-streaming)
                            response = await self.process_message(message_data)
                            
                            # Add the response to the output queue
                            self.output_queue.append(response)
                            logger.info(f"Output queue size: {len(self.output_queue)}")
                        
                        processing_time = (datetime.utcnow() - start_time).total_seconds()
                        logger.info(f"Message processed successfully in {processing_time:.2f} seconds")
                        
                    except Exception as e:
                        logger.error(f"Error processing individual message: {e}")
                        logger.error(traceback.format_exc())
                        
                        if is_streaming and stream_queues and chat_id in stream_queues:
                            # Send error through streaming
                            try:
                                await stream_queues[chat_id].put({
                                    'chat_id': chat_id,
                                    'message_id': message_id,
                                    'type': 'error',
                                    'content': f'Sorry, there was an error processing your request: {str(e)}',
                                    'timestamp': datetime.utcnow().isoformat()
                                })
                            except Exception as stream_err:
                                logger.error(f"Error sending stream error: {stream_err}")
                        else:
                            # Create an error response with message_id for non-streaming
                            error_response = {
                                'chat_id': message_data.get('chat_id'),
                                'message_id': message_data.get('message_id'),  # Include message_id for matching
                                'message': f'Sorry, there was an error processing your request: {str(e)}',
                                'timestamp': datetime.utcnow().isoformat(),
                                'search_results': []
                            }
                            # Add the error response to the output queue
                            self.output_queue.append(error_response)
                            logger.info("Error response added to output queue")
                
                # Sleep to avoid high CPU usage
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(1)  # Sleep longer on error
    
    async def google_web_search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Perform Google web search for Deep Search with Google Search functionality
        
        Args:
            query: Search query
            num_results: Number of results to return (max 10)
            
        Returns:
            List of search results with title, link, snippet
        """
        if not self.google_api_key or not self.google_cse_id:
            logger.error("Google API credentials not configured")
            return []
        
        try:
            params = {
                'key': self.google_api_key,
                'cx': self.google_cse_id,
                'q': query,
                'num': min(num_results, 10),
                'safe': 'active'
            }
            
            logger.info(f"Performing Google search for: {query}")
            response = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            if 'items' in data:
                for item in data['items']:
                    results.append({
                        'title': item.get('title', ''),
                        'link': item.get('link', ''),
                        'snippet': item.get('snippet', ''),
                        'displayLink': item.get('displayLink', '')
                    })
            
            logger.info(f"Found {len(results)} search results")
            return results
            
        except Exception as e:
            logger.error(f"Google search failed: {e}")
            return []
    
    async def extract_web_content(self, url: str) -> Dict[str, Any]:
        """
        Extract content from a web page for Deep Search with Google Search
        
        Args:
            url: URL to extract content from
            
        Returns:
            Dictionary with extracted content, title, and metadata
        """
        if not BeautifulSoup or not Article or not Document:
            logger.warning("Web scraping libraries not available")
            return {
                'url': url,
                'title': 'Content extraction not available',
                'content': 'Web scraping libraries not installed',
                'method': 'none',
                'word_count': 0,
                'success': False,
                'error': 'Libraries not available'
            }
        
        try:
            logger.info(f"Extracting content from: {url}")
            
            # Method 1: Try newspaper3k first (best for articles)
            try:
                article = Article(url)
                article.download()
                article.parse()
                
                if article.text and len(article.text.strip()) > 100:
                    return {
                        'url': url,
                        'title': article.title or 'No title',
                        'content': article.text,
                        'method': 'newspaper3k',
                        'word_count': len(article.text.split()),
                        'success': True
                    }
            except Exception as e:
                logger.warning(f"Newspaper3k failed for {url}: {e}")
            
            # Method 2: Try readability + BeautifulSoup
            try:
                session = requests.Session()
                session.headers.update({
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                })
                response = session.get(url, timeout=10)
                response.raise_for_status()
                
                # Use readability to extract main content
                doc = Document(response.text)
                soup = BeautifulSoup(doc.content(), 'html.parser')
                
                # Extract text content
                content = soup.get_text(separator=' ', strip=True)
                title = doc.title() or soup.find('title')
                title = title.get_text() if hasattr(title, 'get_text') else str(title) if title else 'No title'
                
                if content and len(content.strip()) > 100:
                    return {
                        'url': url,
                        'title': title,
                        'content': content[:3000],  # Limit content length for processing
                        'method': 'readability+bs4',
                        'word_count': len(content.split()),
                        'success': True
                    }
            except Exception as e:
                logger.warning(f"Readability method failed for {url}: {e}")
            
            # All methods failed
            return {
                'url': url,
                'title': 'Extraction failed',
                'content': 'Could not extract content from this URL',
                'method': 'none',
                'word_count': 0,
                'success': False,
                'error': 'All extraction methods failed'
            }
            
        except Exception as e:
            logger.error(f"Content extraction failed for {url}: {e}")
            return {
                'url': url,
                'title': 'Error',
                'content': f'Error extracting content: {str(e)}',
                'method': 'error',
                'word_count': 0,
                'success': False,
                'error': str(e)
            }
    
    async def conduct_deep_search_with_google(self, original_question: str) -> Dict[str, Any]:
        """
        Conduct Deep Search with Google Search functionality - Runs BOTH intelligent and legacy modes
        
        Args:
            original_question: The original research question
            
        Returns:
            Complete research results with both intelligent and legacy mode results
        """
        logger.info(f"Starting Deep Search with Google Search (BOTH modes) for: {original_question}")
        
        results = {
            'original_question': original_question,
            'timestamp': datetime.utcnow().isoformat(),
            'success': False,
            'intelligent_results': {},
            'legacy_results': {},
            'comparison': {}
        }
        
        try:
            # Run both research modes concurrently for comparison
            logger.info("Running both intelligent and legacy research modes concurrently")
            
            # Execute both modes concurrently
            intelligent_task = asyncio.create_task(self.conduct_advanced_research(original_question))
            legacy_task = asyncio.create_task(self.conduct_legacy_research(original_question))
            
            # Wait for both to complete
            intelligent_results, legacy_results = await asyncio.gather(intelligent_task, legacy_task, return_exceptions=True)
            
            # Handle results
            if isinstance(intelligent_results, Exception):
                logger.error(f"Intelligent mode failed: {intelligent_results}")
                results['intelligent_results'] = {'error': str(intelligent_results), 'success': False}
            else:
                results['intelligent_results'] = intelligent_results
                
            if isinstance(legacy_results, Exception):
                logger.error(f"Legacy mode failed: {legacy_results}")
                results['legacy_results'] = {'error': str(legacy_results), 'success': False}
            else:
                results['legacy_results'] = legacy_results
            
            # Determine success
            intelligent_success = results['intelligent_results'].get('success', False)
            legacy_success = results['legacy_results'].get('success', False)
            results['success'] = intelligent_success or legacy_success
            
            # Create comparison summary
            results['comparison'] = {
                'intelligent_mode': {
                    'success': intelligent_success,
                    'research_type': results['intelligent_results'].get('research_type', 'unknown'),
                    'steps_completed': len(results['intelligent_results'].get('steps', {})),
                    'has_direct_answer': 'direct_answer' in results['intelligent_results'],
                    'sources_analyzed': 0
                },
                'legacy_mode': {
                    'success': legacy_success,
                    'research_type': results['legacy_results'].get('research_type', 'unknown'),
                    'steps_completed': len(results['legacy_results'].get('steps', {})),
                    'has_direct_answer': False,  # Legacy always searches
                    'sources_analyzed': 0
                }
            }
            
            # Extract source counts for comparison
            if intelligent_success and 'steps' in results['intelligent_results']:
                step4 = results['intelligent_results']['steps'].get('step4', {})
                if 'analysis' in step4:
                    results['comparison']['intelligent_mode']['sources_analyzed'] = step4['analysis'].get('sources_analyzed', 0)
            
            if legacy_success and 'steps' in results['legacy_results']:
                step4 = results['legacy_results']['steps'].get('step4', {})
                if 'analysis' in step4:
                    results['comparison']['legacy_mode']['sources_analyzed'] = step4['analysis'].get('sources_analyzed', 0)
            
            # Create unified analysis combining both modes for backward compatibility
            unified_analysis = self._create_unified_analysis(results['intelligent_results'], results['legacy_results'])
            results['analysis'] = unified_analysis
            results['search_results'] = self._extract_search_results(results['intelligent_results'], results['legacy_results'])
            results['extracted_contents'] = self._extract_contents(results['intelligent_results'], results['legacy_results'])
            results['research_type'] = 'dual_mode_research'
            
            logger.info("Deep Search with Google Search (BOTH modes) completed successfully")
            
        except Exception as e:
            logger.error(f"Deep Search with Google Search (BOTH modes) failed: {e}")
            results['error'] = str(e)
            results['success'] = False
        
        return results
    
    def _create_unified_analysis(self, intelligent_results: Dict, legacy_results: Dict) -> str:
        """Create a unified analysis combining both intelligent and legacy mode results"""
        analysis_parts = []
        
        analysis_parts.append("# 🔍 Google Deep Search Results - Dual Mode Analysis\n")
        analysis_parts.append("This analysis combines results from both **Intelligent Mode** (with web search necessity check) and **Legacy Mode** (always performs web search).\n")
        
        # Intelligent Mode Results
        analysis_parts.append("## 🧠 Intelligent Mode Results\n")
        if intelligent_results.get('success'):
            if intelligent_results.get('research_type') == 'direct_answer':
                analysis_parts.append("**Decision**: No web search needed - providing direct answer\n")
                analysis_parts.append(f"**Answer**: {intelligent_results.get('direct_answer', 'No answer available')}\n")
            else:
                analysis_parts.append("**Decision**: Web search needed\n")
                if 'steps' in intelligent_results and 'step4' in intelligent_results['steps']:
                    step4 = intelligent_results['steps']['step4']
                    if 'analysis' in step4:
                        analysis_parts.append(f"**Analysis**: {step4['analysis'].get('analysis_content', 'No analysis available')}\n")
        else:
            analysis_parts.append(f"**Error**: {intelligent_results.get('error', 'Unknown error')}\n")
        
        # Legacy Mode Results
        analysis_parts.append("## ⚙️ Legacy Mode Results\n")
        analysis_parts.append("**Decision**: Always performs web search (no necessity check)\n")
        if legacy_results.get('success'):
            if 'steps' in legacy_results and 'step4' in legacy_results['steps']:
                step4 = legacy_results['steps']['step4']
                if 'analysis' in step4:
                    analysis_parts.append(f"**Analysis**: {step4['analysis'].get('analysis_content', 'No analysis available')}\n")
        else:
            analysis_parts.append(f"**Error**: {legacy_results.get('error', 'Unknown error')}\n")
        
        # Comparison
        analysis_parts.append("## ⚖️ Mode Comparison\n")
        analysis_parts.append("- **Intelligent Mode**: Uses AI to determine if web search is necessary first\n")
        analysis_parts.append("- **Legacy Mode**: Always performs comprehensive web search\n")
        analysis_parts.append("- **Recommendation**: Use Intelligent Mode for efficiency, Legacy Mode for thoroughness\n")
        
        return '\n'.join(analysis_parts)
    
    def _extract_search_results(self, intelligent_results: Dict, legacy_results: Dict) -> List[Dict]:
        """Extract search results from either mode for backward compatibility"""
        # Prefer intelligent mode results if available, otherwise use legacy
        if intelligent_results.get('success') and 'steps' in intelligent_results:
            step2 = intelligent_results['steps'].get('step2', {})
            if 'search_results' in step2:
                return step2['search_results']
        
        if legacy_results.get('success') and 'steps' in legacy_results:
            step2 = legacy_results['steps'].get('step2', {})
            if 'search_results' in step2:
                return step2['search_results']
        
        return []
    
    def _extract_contents(self, intelligent_results: Dict, legacy_results: Dict) -> List[Dict]:
        """Extract extracted contents from either mode for backward compatibility"""
        # Prefer intelligent mode results if available, otherwise use legacy
        if intelligent_results.get('success') and 'steps' in intelligent_results:
            step3 = intelligent_results['steps'].get('step3', {})
            if 'extracted_contents' in step3:
                return step3['extracted_contents']
        
        if legacy_results.get('success') and 'steps' in legacy_results:
            step3 = legacy_results['steps'].get('step3', {})
            if 'extracted_contents' in step3:
                return step3['extracted_contents']
        
        return []
    
    async def step0_check_web_search_necessity(self, original_question: str) -> Dict[str, Any]:
        """
        Step 0: Check if web search is necessary for answering the question
        
        Args:
            original_question: The original research question
            
        Returns:
            Dictionary with decision and either search query or direct answer
        """
        try:
            logger.info("Step 0: Checking if web search is necessary")
            
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

            user_prompt = f"""Question: {original_question}

Please analyze this question and determine if web search is necessary. If web search is needed, respond with Query="search_terms". If not needed, provide a comprehensive answer directly."""

            response = await self.async_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                stream=False,
                timeout=30.0
            )
            
            response_text = response.choices[0].message.content
            logger.info(f"DeepSeek web search necessity check: {response_text[:200]}...")
            
            # Check if response contains a search query
            query_match = re.search(r'Query="([^"]+)"', response_text)
            if query_match:
                search_query = query_match.group(1)
                logger.info(f"Web search needed. Extracted search query: {search_query}")
                return {
                    'web_search_needed': True,
                    'search_query': search_query,
                    'response': response_text
                }
            else:
                logger.info("Web search not needed. Direct answer provided.")
                return {
                    'web_search_needed': False,
                    'direct_answer': response_text,
                    'response': response_text
                }
                
        except Exception as e:
            logger.error(f"Step 0 failed: {e}")
            # Default to web search if check fails
            return {
                'web_search_needed': True,
                'search_query': original_question,
                'error': str(e)
            }
    
    async def step1_generate_search_query(self, original_question: str) -> str:
        """
        Step 1: Use DeepSeek to generate optimal search query (legacy method)
        
        Args:
            original_question: The original research question
            
        Returns:
            Optimized search query
        """
        try:
            logger.info("Step 1: Generating optimal search query")
            
            system_message = """You are an expert research assistant. Your task is to analyze a research question and generate the most effective search query for finding relevant information on the web.

Instructions:
1. Analyze the given question carefully
2. Identify the key concepts and search terms
3. Create an optimized search query that will find the most relevant results
4. Return your response in the exact format: Query="your_search_query_here"
5. Make the search query specific enough to find relevant results but broad enough to capture comprehensive information
6. Consider using industry-specific terms, location modifiers, and relevant keywords"""

            user_prompt = f"""Original Question: {original_question}

Please generate the optimal search query for finding comprehensive information to answer this question. Remember to respond in the format Query="your_search_query_here"."""

            response = await self.async_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                stream=False,
                timeout=30.0
            )
            
            response_text = response.choices[0].message.content
            logger.info(f"DeepSeek query generation response: {response_text}")
            
            # Extract query from response
            query_match = re.search(r'Query="([^"]+)"', response_text)
            if query_match:
                search_query = query_match.group(1)
                logger.info(f"Extracted search query: {search_query}")
                return search_query
            else:
                # Fallback: use the original question
                logger.warning("Could not extract query from response, using original question")
                return original_question
                
        except Exception as e:
            logger.error(f"Step 1 failed: {e}")
            return original_question
    
    async def step2_web_search(self, search_query: str) -> List[Dict[str, Any]]:
        """
        Step 2: Perform Google web search
        
        Args:
            search_query: The optimized search query
            
        Returns:
            List of search results
        """
        logger.info("Step 2: Performing web search")
        return await self.google_web_search(search_query, num_results=5)
    
    async def step3_extract_content(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Step 3: Extract full content from search result pages
        
        Args:
            search_results: List of search results from Google
            
        Returns:
            List of extracted content with metadata
        """
        logger.info("Step 3: Extracting content from search results")
        
        extracted_contents = []
        for i, result in enumerate(search_results):
            logger.info(f"Extracting content from result {i+1}/{len(search_results)}: {result['link']}")
            
            content = await self.extract_web_content(result['link'])
            content['search_result'] = result  # Include original search result
            extracted_contents.append(content)
            
            # Add delay to be respectful to websites
            await asyncio.sleep(1)
        
        return extracted_contents
    
    async def step4_analyze_relevance(self, original_question: str, extracted_contents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Step 4: Use DeepSeek Reasoning to analyze content relevance
        
        Args:
            original_question: The original research question
            extracted_contents: List of extracted content from web pages
            
        Returns:
            Analysis results with relevance assessment
        """
        logger.info("Step 4: Analyzing content relevance with DeepSeek Reasoning")
        
        try:
            # Prepare content for analysis
            content_summaries = []
            for i, content in enumerate(extracted_contents):
                if content['success']:
                    summary = f"""
Source {i+1}: {content['title']}
URL: {content['url']}
Content Preview: {content['content'][:1000]}...
Word Count: {content['word_count']}
"""
                else:
                    summary = f"""
Source {i+1}: {content['title']} (Extraction Failed)
URL: {content['url']}
Error: {content.get('error', 'Unknown error')}
"""
                content_summaries.append(summary)
            
            system_message = """You are an expert research analyst with advanced reasoning capabilities. Your task is to analyze extracted web content and determine its relevance to answering a specific research question.

Instructions:
1. Carefully analyze each piece of extracted content
2. Determine how well each source answers the original question
3. Identify which sources provide the most relevant and accurate information
4. For each source, provide:
   - Relevance score (1-10, where 10 is highly relevant)
   - Key insights that help answer the question
   - Any limitations or concerns about the source
5. Synthesize the information to provide a comprehensive answer
6. If there are discrepancies between sources, analyze the causes
7. Provide your reasoning process clearly"""

            user_prompt = f"""Original Research Question: {original_question}

Extracted Content from Web Sources:
{''.join(content_summaries)}

Please analyze each source for relevance to the original question and provide:
1. Individual source analysis with relevance scores
2. Key findings that answer the original question
3. Synthesis of information across sources
4. Analysis of any discrepancies or conflicting information
5. Overall assessment of how well the sources answer the original question"""

            response = await self.async_client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                stream=False,
                timeout=60.0
            )
            
            # Extract reasoning and answer content
            message = response.choices[0].message
            reasoning_content = getattr(message, 'reasoning_content', '') if hasattr(message, 'reasoning_content') else ''
            analysis_content = message.content or ''
            
            return {
                'original_question': original_question,
                'analysis_content': analysis_content,
                'reasoning_content': reasoning_content,
                'sources_analyzed': len(extracted_contents),
                'successful_extractions': sum(1 for c in extracted_contents if c['success']),
                'model': response.model,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Step 4 analysis failed: {e}")
            return {
                'original_question': original_question,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def conduct_advanced_research(self, original_question: str) -> Dict[str, Any]:
        """
        Conduct the complete advanced research process with web search necessity check (Intelligent Mode)
        
        Args:
            original_question: The original research question
            
        Returns:
            Complete research results
        """
        logger.info(f"Starting advanced research (intelligent mode) for: {original_question}")
        
        results = {
            'original_question': original_question,
            'timestamp': datetime.utcnow().isoformat(),
            'steps': {},
            'research_mode': 'intelligent'
        }
        
        try:
            # Step 0: Check if web search is necessary
            necessity_check = await self.step0_check_web_search_necessity(original_question)
            results['steps']['step0'] = {
                'description': 'Check web search necessity',
                'web_search_needed': necessity_check['web_search_needed'],
                'response': necessity_check['response'],
                'success': True
            }
            
            # If web search is not needed, return direct answer
            if not necessity_check['web_search_needed']:
                results['direct_answer'] = necessity_check['direct_answer']
                results['success'] = True
                results['research_type'] = 'direct_answer'
                logger.info("Research completed with direct answer (no web search needed)")
                return results
            
            # If web search is needed, continue with the full process
            search_query = necessity_check['search_query']
            results['research_type'] = 'web_search_research'
            results['steps']['step0']['search_query'] = search_query
            
            # Step 2: Web search (Step 1 is now integrated into Step 0)
            search_results = await self.step2_web_search(search_query)
            results['steps']['step2'] = {
                'description': 'Perform web search',
                'search_results': search_results,
                'results_count': len(search_results),
                'success': len(search_results) > 0
            }
            
            if not search_results:
                results['error'] = 'No search results found'
                return results
            
            # Step 3: Extract content
            extracted_contents = await self.step3_extract_content(search_results)
            results['steps']['step3'] = {
                'description': 'Extract content from web pages',
                'extracted_contents': extracted_contents,
                'successful_extractions': sum(1 for c in extracted_contents if c['success']),
                'total_extractions': len(extracted_contents),
                'success': any(c['success'] for c in extracted_contents)
            }
            
            # Step 4: Analyze relevance
            analysis = await self.step4_analyze_relevance(original_question, extracted_contents)
            results['steps']['step4'] = {
                'description': 'Analyze content relevance with DeepSeek Reasoning',
                'analysis': analysis,
                'success': 'error' not in analysis
            }
            
            results['success'] = True
            logger.info("Advanced research (intelligent mode) completed successfully")
            
        except Exception as e:
            logger.error(f"Advanced research (intelligent mode) failed: {e}")
            results['error'] = str(e)
            results['success'] = False
        
        return results

    async def conduct_legacy_research(self, original_question: str) -> Dict[str, Any]:
        """
        Conduct the legacy 4-step research process (always performs web search) (Legacy Mode)
        
        Args:
            original_question: The original research question
            
        Returns:
            Complete research results
        """
        logger.info(f"Starting legacy research (legacy mode) for: {original_question}")
        
        results = {
            'original_question': original_question,
            'timestamp': datetime.utcnow().isoformat(),
            'steps': {},
            'research_type': 'legacy_web_search',
            'research_mode': 'legacy'
        }
        
        try:
            # Step 1: Generate search query
            search_query = await self.step1_generate_search_query(original_question)
            results['steps']['step1'] = {
                'description': 'Generate optimal search query',
                'search_query': search_query,
                'success': True
            }
            
            # Step 2: Web search
            search_results = await self.step2_web_search(search_query)
            results['steps']['step2'] = {
                'description': 'Perform web search',
                'search_results': search_results,
                'results_count': len(search_results),
                'success': len(search_results) > 0
            }
            
            if not search_results:
                results['error'] = 'No search results found'
                return results
            
            # Step 3: Extract content
            extracted_contents = await self.step3_extract_content(search_results)
            results['steps']['step3'] = {
                'description': 'Extract content from web pages',
                'extracted_contents': extracted_contents,
                'successful_extractions': sum(1 for c in extracted_contents if c['success']),
                'total_extractions': len(extracted_contents),
                'success': any(c['success'] for c in extracted_contents)
            }
            
            # Step 4: Analyze relevance
            analysis = await self.step4_analyze_relevance(original_question, extracted_contents)
            results['steps']['step4'] = {
                'description': 'Analyze content relevance with DeepSeek Reasoning',
                'analysis': analysis,
                'success': 'error' not in analysis
            }
            
            results['success'] = True
            logger.info("Legacy research (legacy mode) completed successfully")
            
        except Exception as e:
            logger.error(f"Legacy research (legacy mode) failed: {e}")
            results['error'] = str(e)
            results['success'] = False
        
        return results
