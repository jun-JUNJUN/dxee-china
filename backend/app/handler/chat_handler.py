import tornado.web
import json
import uuid
import logging
import traceback
import json
import asyncio
from datetime import datetime
from bson import ObjectId, json_util

# Get logger
logger = logging.getLogger(__name__)

# Custom JSON encoder for MongoDB objects
class MongoJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return json_util.default(obj)

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
            
            # Get user_id from request (would come from authentication in a real app)
            user_id = self.get_secure_cookie("user_id")
            if user_id:
                user_id = user_id.decode('utf-8')
            else:
                # For development/testing, use a default user ID
                user_id = "default_user"
            
            # Check if chat exists
            chat = None
            if chat_id:
                chat = await self.application.mongodb.get_chat_by_id(chat_id)
                
                # If chat doesn't exist, create a new one
                if not chat:
                    chat_id = str(uuid.uuid4())
                    chat_doc = {
                        'chat_id': chat_id,  # Keep the UUID format for compatibility
                        'user_id': user_id,
                        'title': message[:30] + "..." if len(message) > 30 else message
                    }
                    await self.application.mongodb.create_chat(chat_doc)
                    logger.info(f"Created new chat with ID: {chat_id}")
            else:
                # Create a new chat
                chat_id = str(uuid.uuid4())
                chat_doc = {
                    'chat_id': chat_id,  # Keep the UUID format for compatibility
                    'user_id': user_id,
                    'title': message[:30] + "..." if len(message) > 30 else message
                }
                await self.application.mongodb.create_chat(chat_doc)
                logger.info(f"Created new chat with ID: {chat_id}")
            
            # Create message document
            message_doc = {
                'message_id': str(uuid.uuid4()),  # Keep the UUID format for compatibility
                'chat_id': chat_id,
                'user_id': user_id,
                'message': message,
                'timestamp': datetime.utcnow(),
                'type': 'user',
                'shared': False
            }
            
            message_id = message_doc['message_id']
            
            # Store the message in MongoDB
            try:
                logger.info(f"Storing message in MongoDB for chat_id: {chat_id}")
                await self.application.mongodb.create_message(message_doc)
                logger.info("Message stored successfully in MongoDB")
            except Exception as e:
                logger.error(f"Error storing message in MongoDB: {e}")
                logger.error(traceback.format_exc())
                # Continue processing even if storage fails
            
            # Add the message to the input queue for processing by DeepSeek
            try:
                logger.info(f"Adding message to input queue for chat_id: {chat_id}, message_id: {message_id}")
                self.application.input_queue.append({
                    'message': message,
                    'chat_id': chat_id,
                    'message_id': message_id
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
            max_attempts = 300  # Maximum number of attempts (30 seconds)
            attempts = 0
            
            logger.info(f"Waiting for response from DeepSeek for chat_id: {chat_id}")
            while response is None and attempts < max_attempts:
                # Check if there's a response in the output queue for this specific message
                for i, item in enumerate(self.application.output_queue):
                    if item.get('chat_id') == chat_id and item.get('message_id') == message_id:
                        response = self.application.output_queue.pop(i)
                        logger.info(f"Found response in output queue for chat_id: {chat_id}, message_id: {message_id}")
                        break
                
                if response is None:
                    # Wait a bit before checking again
                    await tornado.gen.sleep(0.1)
                    attempts += 1
                    
                    # Log every 20 attempts
                    if attempts % 20 == 0:
                        logger.info(f"Waiting for response... Attempt {attempts}/{max_attempts}")
            
            if response is None:
                # If no response after timeout, create a fallback response instead of returning an error
                logger.error(f"Request timeout for chat_id: {chat_id}, message_id: {message_id}")
                
                # Create a fallback response document
                response_doc = {
                    'id': str(uuid.uuid4()),
                    'chat_id': chat_id,
                    'message': "No response received from the AI service. It may be temporarily unavailable.",
                    'timestamp': datetime.utcnow().isoformat(),
                    'type': 'assistant',
                    'search_results': []
                }
                
                # Return the fallback response
                result = {
                    'chat_id': chat_id,
                    'response': response_doc
                }
                logger.info(f"Returning fallback response to client for chat_id: {chat_id}")
                self.write(result)
                return
                
            logger.info(f"Response received for chat_id: {chat_id}")
            
            # Create response document
            try:
                response_doc = {
                    'message_id': str(uuid.uuid4()),  # Keep the UUID format for compatibility
                    'chat_id': chat_id,
                    'user_id': user_id,
                    'message': response.get('message', 'Here are the relevant results for your query.'),
                    'formatted_message': response.get('formatted_message'),  # Add formatted message
                    'timestamp': datetime.utcnow(),
                    'type': 'assistant',
                    'search_results': response.get('search_results', []),
                    'shared': False
                }
                
                # Store the response in MongoDB
                logger.info(f"Storing response in MongoDB for chat_id: {chat_id}")
                await self.application.mongodb.create_message(response_doc)
                logger.info("Response stored successfully in MongoDB")
                
                # Return the response
                # Convert MongoDB document to JSON-serializable format
                response_doc['_id'] = str(response_doc.get('_id', ''))
                response_doc['timestamp'] = response_doc['timestamp'].isoformat()
                
                result = {
                    'chat_id': chat_id,
                    'response': response_doc
                }
                logger.info(f"Returning response to client for chat_id: {chat_id}")
                self.write(json.dumps(result, cls=MongoJSONEncoder))
                self.set_header('Content-Type', 'application/json')
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
            
            # Get user_id from request (would come from authentication in a real app)
            user_id = self.get_secure_cookie("user_id")
            if user_id:
                user_id = user_id.decode('utf-8')
            else:
                # For development/testing, use a default user ID
                user_id = "default_user"
            
            # Get chat to verify ownership
            chat = await self.application.mongodb.get_chat_by_id(chat_id)
            
            if not chat:
                logger.warning(f"Chat not found: {chat_id}")
                self.set_status(404)
                self.write({'error': 'Chat not found'})
                return
            
            # In a real app, you would check if the user owns this chat
            # if chat.get('user_id') != user_id:
            #     logger.warning(f"Unauthorized access to chat: {chat_id}")
            #     self.set_status(403)
            #     self.write({'error': 'Unauthorized'})
            #     return
            
            # Get messages from MongoDB
            try:
                messages = await self.application.mongodb.get_chat_messages(chat_id)
                
                message_count = len(messages)
                logger.info(f"Retrieved {message_count} messages for chat_id: {chat_id}")
                
                # Convert MongoDB documents to JSON-serializable format
                for message in messages:
                    message['_id'] = str(message['_id'])
                    if isinstance(message.get('timestamp'), datetime):
                        message['timestamp'] = message['timestamp'].isoformat()
                
                self.write(json.dumps({
                    'chat_id': chat_id,
                    'messages': messages
                }, cls=MongoJSONEncoder))
                self.set_header('Content-Type', 'application/json')
            except Exception as e:
                logger.error(f"Error retrieving messages from MongoDB: {e}")
                logger.error(traceback.format_exc())
                self.set_status(500)
                self.write({'error': f'Error retrieving chat history: {str(e)}'})
        except Exception as e:
            logger.error(f"Unexpected error in ChatHistoryHandler: {e}")
            logger.error(traceback.format_exc())
            self.set_status(500)
            self.write({'error': f'Server error: {str(e)}'})

class UserChatsHandler(tornado.web.RequestHandler):
    """
    Handler for retrieving a user's chats
    """
    async def get(self):
        try:
            # Get user_id from request (would come from authentication in a real app)
            user_id = self.get_secure_cookie("user_id")
            if user_id:
                user_id = user_id.decode('utf-8')
            else:
                # For development/testing, use a default user ID
                user_id = "default_user"
            
            logger.info(f"Getting chats for user_id: {user_id}")
            
            # Get pagination parameters
            limit = int(self.get_argument('limit', 20))
            skip = int(self.get_argument('skip', 0))
            
            # Get chats from MongoDB
            try:
                chats = await self.application.mongodb.get_user_chats(user_id, limit, skip)
                
                chat_count = len(chats)
                logger.info(f"Retrieved {chat_count} chats for user_id: {user_id}")
                
                # Convert MongoDB documents to JSON-serializable format
                for chat in chats:
                    chat['_id'] = str(chat['_id'])
                    if isinstance(chat.get('created_at'), datetime):
                        chat['created_at'] = chat['created_at'].isoformat()
                    if isinstance(chat.get('updated_at'), datetime):
                        chat['updated_at'] = chat['updated_at'].isoformat()
                
                self.write(json.dumps({
                    'user_id': user_id,
                    'chats': chats
                }, cls=MongoJSONEncoder))
                self.set_header('Content-Type', 'application/json')
            except Exception as e:
                logger.error(f"Error retrieving chats from MongoDB: {e}")
                logger.error(traceback.format_exc())
                self.set_status(500)
                self.write({'error': f'Error retrieving chats: {str(e)}'})
        except Exception as e:
            logger.error(f"Unexpected error in UserChatsHandler: {e}")
            logger.error(traceback.format_exc())
            self.set_status(500)
            self.write({'error': f'Server error: {str(e)}'})

class ShareMessageHandler(tornado.web.RequestHandler):
    """
    Handler for sharing a message
    """
    async def post(self, message_id):
        try:
            # Get user_id from request (would come from authentication in a real app)
            user_id = self.get_secure_cookie("user_id")
            if user_id:
                user_id = user_id.decode('utf-8')
            else:
                # For development/testing, use a default user ID
                user_id = "default_user"
            
            logger.info(f"Sharing message: {message_id}")
            
            # Parse request body
            try:
                data = json.loads(self.request.body)
                share = data.get('share', True)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in request: {e}")
                self.set_status(400)
                self.write({'error': f'Invalid JSON: {str(e)}'})
                return
            
            # Update message in MongoDB
            try:
                result = await self.application.mongodb.share_message(message_id, share)
                
                if result > 0:
                    logger.info(f"Message {message_id} {'shared' if share else 'unshared'} successfully")
                    
                    # If sharing, also add to Meilisearch for search
                    if share:
                        # Get the message from MongoDB
                        message = await self.application.mongodb.messages.find_one({"_id": ObjectId(message_id)})
                        
                        if message:
                            # Convert MongoDB document to JSON-serializable format
                            message_doc = json.loads(json_util.dumps(message))
                            
                            # Add to Meilisearch
                            await self.application.meilisearch.index('chat_history').add_documents([message_doc])
                            logger.info(f"Message {message_id} added to Meilisearch for search")
                    
                    self.write({'success': True})
                else:
                    logger.warning(f"Message not found: {message_id}")
                    self.set_status(404)
                    self.write({'error': 'Message not found'})
            except Exception as e:
                logger.error(f"Error updating message in MongoDB: {e}")
                logger.error(traceback.format_exc())
                self.set_status(500)
                self.write({'error': f'Error sharing message: {str(e)}'})
        except Exception as e:
            logger.error(f"Unexpected error in ShareMessageHandler: {e}")
            logger.error(traceback.format_exc())
            self.set_status(500)
            self.write({'error': f'Server error: {str(e)}'})

class ChatStreamHandler(tornado.web.RequestHandler):
    """
    Handler for streaming chat messages using Server-Sent Events
    """
    async def post(self):
        try:
            # Parse request body
            try:
                data = json.loads(self.request.body)
                message = data.get('message')
                chat_id = data.get('chat_id')
                chat_history = data.get('chat_history', [])
                
                logger.info(f"Received streaming message: {message[:30]}... for chat: {chat_id or 'new chat'}")
                logger.info(f"Chat history provided: {len(chat_history)} messages")
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
            
            # Get user_id from request (would come from authentication in a real app)
            user_id = self.get_secure_cookie("user_id")
            if user_id:
                user_id = user_id.decode('utf-8')
            else:
                # For development/testing, use a default user ID
                user_id = "default_user"
            
            # Set headers for Server-Sent Events
            self.set_header('Content-Type', 'text/event-stream')
            self.set_header('Cache-Control', 'no-cache')
            self.set_header('Connection', 'keep-alive')
            self.set_header('Access-Control-Allow-Origin', '*')
            self.set_header('Access-Control-Allow-Headers', 'Content-Type')
            
            # Check if chat exists
            chat = None
            if chat_id:
                chat = await self.application.mongodb.get_chat_by_id(chat_id)
                
                # If chat doesn't exist, create a new one
                if not chat:
                    chat_id = str(uuid.uuid4())
                    chat_doc = {
                        'chat_id': chat_id,
                        'user_id': user_id,
                        'title': message[:30] + "..." if len(message) > 30 else message
                    }
                    await self.application.mongodb.create_chat(chat_doc)
                    logger.info(f"Created new chat with ID: {chat_id}")
            else:
                # Create a new chat
                chat_id = str(uuid.uuid4())
                chat_doc = {
                    'chat_id': chat_id,
                    'user_id': user_id,
                    'title': message[:30] + "..." if len(message) > 30 else message
                }
                await self.application.mongodb.create_chat(chat_doc)
                logger.info(f"Created new chat with ID: {chat_id}")
            
            # Send initial response with chat_id
            self.write(f"data: {json.dumps({'type': 'chat_id', 'chat_id': chat_id})}\n\n")
            await self.flush()
            
            # Create message document
            message_doc = {
                'message_id': str(uuid.uuid4()),
                'chat_id': chat_id,
                'user_id': user_id,
                'message': message,
                'timestamp': datetime.utcnow(),
                'type': 'user',
                'shared': False
            }
            
            message_id = message_doc['message_id']
            
            # Store the user message in MongoDB
            try:
                logger.info(f"Storing user message in MongoDB for chat_id: {chat_id}")
                await self.application.mongodb.create_message(message_doc)
                logger.info("User message stored successfully in MongoDB")
            except Exception as e:
                logger.error(f"Error storing user message in MongoDB: {e}")
                logger.error(traceback.format_exc())
            
            # Create a stream queue for this request
            stream_queue = asyncio.Queue()
            
            # Initialize stream_queues dict if it doesn't exist
            if not hasattr(self.application, 'stream_queues'):
                self.application.stream_queues = {}
            
            # Add stream queue for this chat
            self.application.stream_queues[chat_id] = stream_queue
            
            # Add the message to the input queue for processing by DeepSeek with streaming flag
            try:
                logger.info(f"Adding streaming message to input queue for chat_id: {chat_id}, message_id: {message_id}")
                self.application.input_queue.append({
                    'message': message,
                    'chat_id': chat_id,
                    'message_id': message_id,
                    'streaming': True,
                    'chat_history': chat_history
                })
                logger.info(f"Queue size after adding: {len(self.application.input_queue)}")
            except Exception as e:
                logger.error(f"Error adding message to input queue: {e}")
                logger.error(traceback.format_exc())
                self.write(f"data: {json.dumps({'type': 'error', 'content': f'Error adding message to queue: {str(e)}'})}\n\n")
                await self.flush()
                return
            
            # Stream the response
            accumulated_content = ""
            try:
                while True:
                    try:
                        # Wait for chunks from the stream queue with timeout
                        chunk = await asyncio.wait_for(stream_queue.get(), timeout=0.5)
                        
                        if chunk['type'] == 'error':
                            self.write(f"data: {json.dumps(chunk)}\n\n")
                            await self.flush()
                            break
                        elif chunk['type'] == 'complete':
                            # Store the complete AI response in MongoDB
                            response_doc = {
                                'message_id': str(uuid.uuid4()),
                                'chat_id': chat_id,
                                'user_id': user_id,
                                'message': chunk['content'],
                                'timestamp': datetime.utcnow(),
                                'type': 'assistant',
                                'search_results': chunk.get('search_results', []),
                                'shared': False
                            }
                            
                            try:
                                await self.application.mongodb.create_message(response_doc)
                                logger.info("AI response stored successfully in MongoDB")
                            except Exception as e:
                                logger.error(f"Error storing AI response in MongoDB: {e}")
                            
                            # Send final complete chunk
                            self.write(f"data: {json.dumps(chunk)}\n\n")
                            await self.flush()
                            break
                        elif chunk['type'] == 'chunk':
                            accumulated_content += chunk['content']
                            self.write(f"data: {json.dumps(chunk)}\n\n")
                            await self.flush()
                    
                    except asyncio.TimeoutError:
                        # Send heartbeat to keep connection alive
                        self.write(f"data: {json.dumps({'type': 'heartbeat'})}\n\n")
                        await self.flush()
                        continue
                    except Exception as e:
                        logger.error(f"Error in streaming loop: {e}")
                        self.write(f"data: {json.dumps({'type': 'error', 'content': f'Streaming error: {str(e)}'})}\n\n")
                        await self.flush()
                        break
            
            finally:
                # Clean up the stream queue
                if chat_id in self.application.stream_queues:
                    del self.application.stream_queues[chat_id]
                logger.info(f"Streaming completed for chat_id: {chat_id}")
        
        except Exception as e:
            logger.error(f"Unexpected error in ChatStreamHandler: {e}")
            logger.error(traceback.format_exc())
            try:
                self.write(f"data: {json.dumps({'type': 'error', 'content': f'Server error: {str(e)}'})}\n\n")
                await self.flush()
            except:
                pass

class SharedMessagesHandler(tornado.web.RequestHandler):
    """
    Handler for retrieving shared messages
    """
    async def get(self):
        try:
            logger.info("Getting shared messages")
            
            # Get pagination parameters
            limit = int(self.get_argument('limit', 20))
            skip = int(self.get_argument('skip', 0))
            
            # Get shared messages from MongoDB
            try:
                messages = await self.application.mongodb.get_shared_messages(limit, skip)
                
                message_count = len(messages)
                logger.info(f"Retrieved {message_count} shared messages")
                
                # Convert MongoDB documents to JSON-serializable format
                for message in messages:
                    message['_id'] = str(message['_id'])
                    if isinstance(message.get('timestamp'), datetime):
                        message['timestamp'] = message['timestamp'].isoformat()
                    if isinstance(message.get('shared_at'), datetime):
                        message['shared_at'] = message['shared_at'].isoformat()
                
                self.write(json.dumps({
                    'messages': messages
                }, cls=MongoJSONEncoder))
                self.set_header('Content-Type', 'application/json')
            except Exception as e:
                logger.error(f"Error retrieving shared messages from MongoDB: {e}")
                logger.error(traceback.format_exc())
                self.set_status(500)
                self.write({'error': f'Error retrieving shared messages: {str(e)}'})
        except Exception as e:
            logger.error(f"Unexpected error in SharedMessagesHandler: {e}")
            logger.error(traceback.format_exc())
            self.set_status(500)
            self.write({'error': f'Server error: {str(e)}'})
