import os
import json
import asyncio
import logging
import traceback
from datetime import datetime
from openai import OpenAI
from openai import AsyncOpenAI

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
        
        logger.info(f"DeepSeek service initialized with API URL: {self.api_url}")
        
        if not self.api_key:
            logger.warning("DEEPSEEK_API_KEY environment variable not set")
            logger.warning("Please set DEEPSEEK_API_KEY in your .env file")
    
    def chat_completion(self, query, system_message="You are a helpful assistant", max_retries=6):
            """
            Get a chat completion from the DeepSeek API using the synchronous client
            
            Args:
                query (str): The user query
                system_message (str): The system message to set the assistant's behavior
                max_retries (int): Maximum number of retry attempts for API calls
                
            Returns:
                dict: Chat completion response from DeepSeek
            """
            # List of models to try if the primary model fails
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
    
    async def async_chat_completion(self, query, system_message="You are a helpful assistant", max_retries=6):
            """
            Get a chat completion from the DeepSeek API using the asynchronous client
            
            Args:
                query (str): The user query
                system_message (str): The system message to set the assistant's behavior
                max_retries (int): Maximum number of retry attempts for API calls
                
            Returns:
                dict: Chat completion response from DeepSeek
            """
            # List of models to try if the primary model fails
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
            
            logger.info(f"Processing message: {query[:30]}... (ID: {message_id}, Chat: {chat_id})")
            
            # Get chat completion using DeepSeek API
            completion_response = await self.async_chat_completion(query)
            
            # Create response with message_id to match the specific request
            response = {
                'chat_id': chat_id,
                'message_id': message_id,  # Include message_id for matching
                'message': completion_response.get('content', 'No response from DeepSeek API.'),
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
    
    async def start_processing(self):
        """
        Start processing messages from the input queue
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
                    message_preview = message_data.get('message', '')[:30] if message_data.get('message') else 'empty'
                    
                    logger.info(f"Processing message from queue: {message_preview}... (Chat: {chat_id})")
                    logger.info(f"Queue size before processing: {queue_size}, after: {len(self.input_queue)}")
                    
                    try:
                        # Process the message
                        start_time = datetime.utcnow()
                        response = await self.process_message(message_data)
                        processing_time = (datetime.utcnow() - start_time).total_seconds()
                        
                        # Add the response to the output queue
                        self.output_queue.append(response)
                        logger.info(f"Message processed successfully in {processing_time:.2f} seconds")
                        logger.info(f"Output queue size: {len(self.output_queue)}")
                    except Exception as e:
                        logger.error(f"Error processing individual message: {e}")
                        logger.error(traceback.format_exc())
                        
                        # Create an error response with message_id
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
