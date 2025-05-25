import os
import json
import asyncio
import logging
import traceback
from datetime import datetime
from openai import OpenAI
from openai import AsyncOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
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
    
    def chat_completion(self, query, system_message="You are a helpful assistant"):
        """
        Get a chat completion from the DeepSeek API using the synchronous client
        
        Args:
            query (str): The user query
            system_message (str): The system message to set the assistant's behavior
            
        Returns:
            dict: Chat completion response from DeepSeek
        """
        try:
            logger.info(f"Sending chat completion request to DeepSeek API: {query[:30]}...")
            
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": query},
                ],
                stream=False
            )
            
            logger.info("Chat completion request successful")
            return {
                "model": response.model,
                "id": response.id,
                "created": response.created,
                "content": response.choices[0].message.content
            }
        except Exception as e:
            logger.error(f"Error getting chat completion from DeepSeek API: {e}")
            logger.error(traceback.format_exc())
            return {"content": f"Error: {str(e)}"}
    
    async def async_chat_completion(self, query, system_message="You are a helpful assistant"):
        """
        Get a chat completion from the DeepSeek API using the asynchronous client
        
        Args:
            query (str): The user query
            system_message (str): The system message to set the assistant's behavior
            
        Returns:
            dict: Chat completion response from DeepSeek
        """
        try:
            logger.info(f"Sending async chat completion request to DeepSeek API: {query[:30]}...")
            
            response = await self.async_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": query},
                ],
                stream=False
            )
            
            logger.info("Async chat completion request successful")
            return {
                "model": response.model,
                "id": response.id,
                "created": response.created,
                "content": response.choices[0].message.content
            }
        except Exception as e:
            logger.error(f"Error getting async chat completion from DeepSeek API: {e}")
            logger.error(traceback.format_exc())
            return {"content": f"Error: {str(e)}"}
    
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
            
            # Create response
            response = {
                'chat_id': chat_id,
                'message': completion_response.get('content', 'No response from DeepSeek API.'),
                'timestamp': datetime.utcnow().isoformat(),
                'search_results': []  # Keep this for backward compatibility
            }
            
            logger.info(f"Response created for chat_id: {chat_id}")
            return response
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            logger.error(traceback.format_exc())
            
            # Return error response
            return {
                'chat_id': chat_id,
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
                        
                        # Create an error response
                        error_response = {
                            'chat_id': message_data.get('chat_id'),
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
