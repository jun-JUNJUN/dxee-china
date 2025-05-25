import tornado.web
import json
import uuid
import logging
import traceback
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatMessageHandler(tornado.web.RequestHandler):
    """
    Handler for chat messages
    """
    async def post(self):
        try:
            # Parse request body
            try:
                data = json.loads(self.request.body)
                message = data.get('message')
                chat_id = data.get('chat_id')
                
                logger.info(f"Received message: {message[:30]}... for chat: {chat_id or 'new chat'}")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in request: {e}")
                logger.error(f"Request body: {self.request.body}")
                self.set_status(400)
                self.write({'error': f'Invalid JSON: {str(e)}'})
                return
            
            if not message:
                logger.warning("Message is required but was not provided")
                self.set_status(400)
                self.write({'error': 'Message is required'})
                return
            
            # Generate a new chat ID if not provided
            if not chat_id:
                chat_id = str(uuid.uuid4())
                logger.info(f"Generated new chat_id: {chat_id}")
            
            # Create message document
            message_doc = {
                'id': str(uuid.uuid4()),
                'chat_id': chat_id,
                'message': message,
                'timestamp': datetime.utcnow().isoformat(),
                'type': 'user'
            }
            
            # Store the message in chat history
            await self.application.meilisearch.index('chat_history').add_documents([message_doc])
            
            # Add the message to the input queue for processing by DeepSeek
            self.application.input_queue.append({
                'message': message,
                'chat_id': chat_id,
                'message_id': message_doc['id']
            })
            
            # Store the message in chat history
            try:
                message_doc = {
                    'id': str(uuid.uuid4()),
                    'chat_id': chat_id,
                    'message': message,
                    'timestamp': datetime.utcnow().isoformat(),
                    'type': 'user'
                }
                
                logger.info(f"Storing message in chat history for chat_id: {chat_id}")
                await self.application.meilisearch.index('chat_history').add_documents([message_doc])
                logger.info("Message stored successfully")
            except Exception as e:
                logger.error(f"Error storing message in chat history: {e}")
                logger.error(traceback.format_exc())
                # Continue processing even if storage fails
            
            # Add the message to the input queue for processing by DeepSeek
            try:
                logger.info(f"Adding message to input queue for chat_id: {chat_id}")
                self.application.input_queue.append({
                    'message': message,
                    'chat_id': chat_id,
                    'message_id': message_doc['id']
                })
                logger.info(f"Queue size after adding: {len(self.application.input_queue)}")
            except Exception as e:
                logger.error(f"Error adding message to input queue: {e}")
                logger.error(traceback.format_exc())
                self.set_status(500)
                self.write({'error': f'Error adding message to queue: {str(e)}'})
                return
            
            # Wait for the response from the output queue
            response = None
            max_attempts = 100  # Maximum number of attempts (10 seconds)
            attempts = 0
            
            logger.info(f"Waiting for response from DeepSeek for chat_id: {chat_id}")
            while response is None and attempts < max_attempts:
                # Check if there's a response in the output queue for this chat
                for i, item in enumerate(self.application.output_queue):
                    if item.get('chat_id') == chat_id:
                        response = self.application.output_queue.pop(i)
                        logger.info(f"Found response in output queue for chat_id: {chat_id}")
                        break
                
                if response is None:
                    # Wait a bit before checking again
                    await tornado.gen.sleep(0.1)
                    attempts += 1
                    
                    # Log every 20 attempts
                    if attempts % 20 == 0:
                        logger.info(f"Waiting for response... Attempt {attempts}/{max_attempts}")
            
            if response is None:
                # If no response after timeout, return an error
                logger.error(f"Request timeout for chat_id: {chat_id}")
                self.set_status(408)
                self.write({'error': 'Request timeout. The DeepSeek API may be unavailable.'})
                return
                
            logger.info(f"Response received for chat_id: {chat_id}")
            
            # Create response document
            try:
                response_doc = {
                    'id': str(uuid.uuid4()),
                    'chat_id': chat_id,
                    'message': response.get('message', 'Here are the relevant results for your query.'),
                    'timestamp': datetime.utcnow().isoformat(),
                    'type': 'assistant',
                    'search_results': response.get('search_results', [])
                }
                
                # Store the response in chat history
                logger.info(f"Storing response in chat history for chat_id: {chat_id}")
                await self.application.meilisearch.index('chat_history').add_documents([response_doc])
                logger.info("Response stored successfully")
                
                # Return the response
                result = {
                    'chat_id': chat_id,
                    'response': response_doc
                }
                logger.info(f"Returning response to client for chat_id: {chat_id}")
                self.write(result)
            except Exception as e:
                logger.error(f"Error processing response: {e}")
                logger.error(traceback.format_exc())
                self.set_status(500)
                self.write({'error': f'Error processing response: {str(e)}'})
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            logger.error(f"Request body: {self.request.body}")
            self.set_status(400)
            self.write({'error': f'Invalid JSON: {str(e)}'})
        except Exception as e:
            logger.error(f"Unexpected error in ChatMessageHandler: {e}")
            logger.error(traceback.format_exc())
            self.set_status(500)
            self.write({'error': f'Server error: {str(e)}'})

class ChatHistoryHandler(tornado.web.RequestHandler):
    """
    Handler for retrieving chat history
    """
    async def get(self, chat_id):
        try:
            logger.info(f"Getting chat history for chat_id: {chat_id}")
            
            # Search for messages with the given chat ID
            try:
                results = await self.application.meilisearch.index('chat_history').search(
                    '',
                    {
                        'filter': f'chat_id = {chat_id}',
                        'sort': ['timestamp:asc'],
                        'limit': 100
                    }
                )
                
                message_count = len(results.get('hits', []))
                logger.info(f"Retrieved {message_count} messages for chat_id: {chat_id}")
                
                self.write({
                    'chat_id': chat_id,
                    'messages': results['hits']
                })
            except Exception as e:
                logger.error(f"Error searching Meilisearch: {e}")
                logger.error(traceback.format_exc())
                self.set_status(500)
                self.write({'error': f'Error retrieving chat history: {str(e)}'})
        except Exception as e:
            logger.error(f"Unexpected error in ChatHistoryHandler: {e}")
            logger.error(traceback.format_exc())
            self.set_status(500)
            self.write({'error': f'Server error: {str(e)}'})
