import tornado.web
import json
import uuid
import logging
import traceback
import json
import asyncio
import os
from datetime import datetime, timezone
from bson import ObjectId, json_util

# Import the simplified deep-think orchestrator and enhanced streaming
from app.service.deep_think_orchestrator import DeepThinkOrchestrator
from app.service.progress_streaming_service import ProgressStreamingService
from app.service.session_manager_service import SessionManagerService
from app.service.enhanced_chat_storage import EnhancedChatStorageService

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
                search_mode = data.get('search_mode', 'search')
                
                logger.info(f"Received message: {message[:30]}... for chat: {chat_id or 'new chat'}, mode: {search_mode}")
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
                logger.info(f"Adding message to input queue for chat_id: {chat_id}, message_id: {message_id}, mode: {search_mode}")
                self.application.input_queue.append({
                    'message': message,
                    'chat_id': chat_id,
                    'message_id': message_id,
                    'search_mode': search_mode
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
                search_mode = data.get('search_mode', 'search')
                
                logger.info(f"Received streaming message: {message[:30]}... for chat: {chat_id or 'new chat'}, mode: {search_mode}")
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
            
            
            # Check if this is a deep-think request (new orchestrator)
            if search_mode == "deepthink":
                logger.info(f"Deep-think mode detected for chat_id: {chat_id}")
                await self._handle_deepthink_research(message, chat_id, user_id, message_id, stream_queue)
                return
            
            # Add the message to the input queue for processing by regular DeepSeek with streaming flag
            try:
                logger.info(f"Adding streaming message to input queue for chat_id: {chat_id}, message_id: {message_id}, mode: {search_mode}")
                self.application.input_queue.append({
                    'message': message,
                    'chat_id': chat_id,
                    'message_id': message_id,
                    'streaming': True,
                    'chat_history': chat_history,
                    'search_mode': search_mode
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
                        elif chunk['type'] == 'reasoning_chunk':
                            # Handle reasoning content chunks
                            self.write(f"data: {json.dumps(chunk)}\n\n")
                            await self.flush()
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
    
    async def _handle_deepthink_research(self, message: str, chat_id: str, user_id: str, message_id: str, stream_queue):
        """Handle deep-think research using enhanced SSE streaming and session management"""
        session_id = str(uuid.uuid4())
        
        try:
            # Initialize services
            session_manager = SessionManagerService(self.application.mongodb)
            await session_manager.initialize()
            
            streaming_service = ProgressStreamingService()
            await streaming_service.start()
            
            # Create session in session manager
            session = await session_manager.create_session(
                question=message,
                chat_id=chat_id,
                user_id=user_id,
                timeout_seconds=600
            )
            
            # Create streaming connection
            connection_id = await streaming_service.create_streaming_connection(
                session_id=session_id,
                user_id=user_id,
                chat_id=chat_id,
                handler=self
            )
            
            logger.info(f"Created Deep Think session {session_id} with streaming connection {connection_id}")
            
            # Initialize orchestrator with session management
            orchestrator = DeepThinkOrchestrator(
                deepseek_service=self.application.deepseek_service,
                serper_client=self.application.serper_client,
                html_cache_service=self.application.html_cache_service,
                mongodb_service=self.application.mongodb,
                session_manager=session_manager
            )
            
            # Start background research task
            asyncio.create_task(self._background_research_with_streaming(
                orchestrator, streaming_service, session, message, chat_id, user_id, message_id
            ))
            
            # Stream progress updates with enhanced error handling
            try:
                async for progress_update in orchestrator.start_research_session(
                    message, session_id, chat_id, user_id
                ):
                    # Progress is automatically handled by the streaming service
                    # through the session manager integration
                    
                    await streaming_service.handle_session_progress(
                        session_id=session_id,
                        step=progress_update.step.value,
                        progress=progress_update.progress,
                        description=progress_update.description,
                        metadata=progress_update.metadata
                    )
                    
                    # If research is complete, break from streaming
                    if progress_update.progress >= 100:
                        break
                        
            except asyncio.TimeoutError:
                logger.info(f"Client connection timed out for session {session_id}")
                await streaming_service.broadcast_progress(session_id, {
                    'type': 'timeout_info',
                    'message': "Deep Think analysis continues in background",
                    'details': "You can safely close this page - results will be saved to chat history"
                })
                
            except Exception as stream_error:
                logger.error(f"Streaming error for session {session_id}: {stream_error}")
                await streaming_service.broadcast_error(session_id, f"Streaming error: {str(stream_error)}")
            
        except Exception as e:
            logger.error(f"Deep Think setup error: {e}")
            logger.error(traceback.format_exc())
            
            # Send error through basic SSE if streaming service failed
            error_data = {'type': 'error', 'content': f'Deep Think failed: {str(e)}'}
            self.write(f"data: {json.dumps(error_data)}\n\n")
            await self.flush()

    async def _background_research_with_streaming(self, orchestrator: DeepThinkOrchestrator, 
                                                streaming_service: ProgressStreamingService,
                                                session, question: str, chat_id: str, 
                                                user_id: str, message_id: str):
        """Background research task with enhanced streaming integration"""
        session_id = session.session_id
        
        try:
            logger.info(f"🔄 Starting background Deep Think with streaming for session {session_id}")
            
            # Process complete research session with streaming updates
            final_result = None
            async for progress_update in orchestrator.start_research_session(question, session_id, chat_id, user_id):
                # Stream progress through streaming service
                await streaming_service.handle_session_progress(
                    session_id=session_id,
                    step=progress_update.step.value,
                    progress=progress_update.progress,
                    description=progress_update.description,
                    metadata=progress_update.metadata
                )
                
                # Capture final result
                if progress_update.progress >= 100 and progress_update.metadata.get('result'):
                    final_result = progress_update.metadata['result']
                    break
            
            # Handle completion
            if final_result:
                await self._handle_research_completion(
                    streaming_service, session_id, final_result, chat_id, user_id, message_id
                )
            else:
                error_msg = "Research completed but no result was generated"
                logger.error(f"❌ {error_msg} for session {session_id}")
                await streaming_service.handle_session_error(session_id, error_msg)
                
        except Exception as e:
            error_msg = f"Background research failed: {str(e)}"
            logger.error(f"❌ {error_msg} for session {session_id}")
            await streaming_service.handle_session_error(session_id, error_msg)
        
        finally:
            # Clean up streaming service
            try:
                await streaming_service.stop()
            except Exception as cleanup_error:
                logger.warning(f"Error during streaming cleanup: {cleanup_error}")

    async def _handle_research_completion(self, streaming_service: ProgressStreamingService,
                                        session_id: str, final_result: dict, 
                                        chat_id: str, user_id: str, message_id: str):
        """Handle successful research completion with enhanced storage"""
        try:
            # Initialize enhanced storage service
            enhanced_storage = EnhancedChatStorageService(self.application.mongodb)
            await enhanced_storage.initialize()
            
            # Initialize session manager to get session details
            session_manager = SessionManagerService(self.application.mongodb)
            await session_manager.initialize()
            
            # Get session details
            session = await session_manager.get_session(session_id)
            if not session:
                logger.error(f"Session {session_id} not found for completion")
                await streaming_service.handle_session_error(session_id, "Session not found")
                return
            
            # Convert final_result to SessionResult format
            from ..models.session_models import SessionResult
            
            result_answer = final_result['answer']
            session_result = SessionResult(
                answer_content=result_answer['content'],
                confidence=result_answer['confidence'],
                sources=result_answer.get('sources', []),
                statistics=result_answer.get('statistics', {}),
                generation_time=final_result.get('generation_time', 0.0),
                total_duration=final_result.get('total_duration', 0.0),
                queries_generated=final_result.get('queries_generated', 0),
                sources_analyzed=final_result.get('sources_analyzed', 0),
                cache_hits=final_result.get('cache_hits', 0),
                metadata=final_result.get('metadata', {})
            )
            
            # Store using enhanced storage service
            stored_message_id = await enhanced_storage.store_deepthink_result(session, session_result)
            
            # Complete session in session manager
            await session_manager.complete_session(session_id, session_result)
            
            # Notify completion through streaming service
            await streaming_service.handle_session_completion(session_id, {
                'message_stored': True,
                'message_id': stored_message_id,
                'confidence': result_answer['confidence'],
                'sources_count': len(result_answer.get('sources', [])),
                'processing_time': final_result['total_duration'],
                'enhanced_storage': True
            })
            
            logger.info(f"✅ Enhanced Deep Think storage completed for session {session_id} -> message {stored_message_id}")
            
        except Exception as e:
            error_msg = f"Failed to handle enhanced research completion: {str(e)}"
            logger.error(f"❌ {error_msg}")
            await streaming_service.handle_session_error(session_id, error_msg)

    async def _background_deepthink_session(self, orchestrator: DeepThinkOrchestrator, session_id: str, 
                                          question: str, chat_id: str, user_id: str, message_id: str):
        """Legacy background session method (kept for compatibility)"""
        try:
            logger.info(f"🔄 Starting background Deep Think session {session_id}")
            
            # Process the complete research session
            final_result = None
            async for progress_update in orchestrator.start_research_session(question, session_id):
                if progress_update.progress >= 100 and progress_update.metadata.get('result'):
                    final_result = progress_update.metadata['result']
                    break
            
            if final_result:
                # Extract research result data
                result_answer = final_result['answer']
                confidence = result_answer['confidence']
                confidence_emoji = "🟢" if confidence >= 0.8 else "🟡" if confidence >= 0.6 else "🔴"
                
                # Format result for chat storage
                formatted_parts = [
                    f"**Deep Think Research Result** {confidence_emoji}",
                    "",
                    "## 📋 Research Answer",
                    result_answer['content'],
                    "",
                    f"**Confidence:** {confidence:.1%}",
                ]
                
                # Add statistics if available
                if result_answer.get('statistics'):
                    stats = result_answer['statistics']
                    formatted_parts.extend([
                        "",
                        "## 📊 Research Statistics",
                        f"- **Numbers found:** {len(stats.get('numbers_found', []))}",
                        f"- **Percentages:** {len(stats.get('percentages', []))}",
                        f"- **Sources analyzed:** {final_result['sources_analyzed']}",
                        f"- **Cache hits:** {final_result['cache_hits']}",
                    ])
                
                # Add sources
                if result_answer.get('sources'):
                    formatted_parts.extend([
                        "",
                        "## 🔗 Top Sources",
                    ])
                    for i, source in enumerate(result_answer['sources'][:3], 1):
                        formatted_parts.append(f"{i}. {source}")
                
                # Add processing time
                formatted_parts.append(f"\n*Processing time: {final_result['total_duration']:.1f}s*")
                
                formatted_response = "\n".join(formatted_parts)
                
                # Store response in MongoDB
                response_doc = {
                    'message_id': str(uuid.uuid4()),
                    'chat_id': chat_id,
                    'user_id': user_id,
                    'message': formatted_response,
                    'timestamp': datetime.now(timezone.utc),
                    'type': 'assistant',
                    'search_results': result_answer.get('sources', []),
                    'deepthink_data': final_result,  # Store complete research data
                    'session_id': session_id,
                    'shared': False,
                    'deepthink_completed': True
                }
                
                try:
                    await self.application.mongodb.create_message(response_doc)
                    logger.info(f"✅ Deep Think response stored for session {session_id}")
                    
                    # Clean up session
                    orchestrator.cleanup_session(session_id)
                    
                except Exception as e:
                    logger.error(f"❌ Error storing Deep Think response: {e}")
            else:
                logger.error(f"❌ Deep Think session {session_id} failed to produce result")
                
        except Exception as e:
            logger.error(f"❌ Background Deep Think session {session_id} failed: {e}")
            # Ensure cleanup even on error
            try:
                orchestrator.cleanup_session(session_id)
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
