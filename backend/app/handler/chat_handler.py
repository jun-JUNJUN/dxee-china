import tornado.web
import json
import uuid
import logging
import traceback
import json
import asyncio
import os
from datetime import datetime
from bson import ObjectId, json_util

# Import the enhanced research service and new deep-think orchestrator
from app.service.enhanced_deepseek_research_service import EnhancedDeepSeekResearchService
from app.service.deepthink_orchestrator import DeepThinkOrchestrator
from app.service.deepthink_models import DeepThinkRequest

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
            
            # Check if this is a DeepSeek research request
            if search_mode == "deepseek":
                logger.info(f"DeepSeek research mode detected for chat_id: {chat_id}")
                await self._handle_deepseek_research(message, chat_id, user_id, message_id, stream_queue)
                return
            
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
    
    async def _handle_deepseek_research(self, message: str, chat_id: str, user_id: str, message_id: str, stream_queue):
        """Handle DeepSeek research mode with streaming progress updates"""
        try:
            # Initialize the enhanced research service
            research_service = EnhancedDeepSeekResearchService(
                mongodb_service=self.application.mongodb,
                cache_expiry_days=int(os.environ.get('CACHE_EXPIRY_DAYS', 30))
            )
            await research_service.initialize()
            
            # Send research starting event
            self.write(f"data: {json.dumps({'type': 'research_step', 'step': 'initializing', 'content': '🔬 Starting DeepSeek research...'})}\n\n")
            await self.flush()
            
            # Send query generation step
            self.write(f"data: {json.dumps({'type': 'research_step', 'step': 'query_generation', 'content': '📝 Generating search queries...'})}\n\n")
            await self.flush()
            
            # Send web search step
            self.write(f"data: {json.dumps({'type': 'research_step', 'step': 'web_search', 'content': '🔍 Searching the web for information...'})}\n\n")
            await self.flush()
            
            # Send content extraction step
            self.write(f"data: {json.dumps({'type': 'research_step', 'step': 'content_extraction', 'content': '📄 Extracting content from sources...'})}\n\n")
            await self.flush()
            
            # Send relevance evaluation step
            self.write(f"data: {json.dumps({'type': 'research_step', 'step': 'relevance_evaluation', 'content': '🎯 Evaluating content relevance...'})}\n\n")
            await self.flush()
            
            # Send analysis step
            self.write(f"data: {json.dumps({'type': 'research_step', 'step': 'analysis', 'content': '🧠 Generating comprehensive analysis...'})}\n\n")
            await self.flush()
            
            # Conduct the research
            research_results = await research_service.conduct_deepseek_research(message, chat_id)
            
            # Send progress updates based on research results
            if research_results.get('success'):
                steps = research_results.get('steps', {})
                
                # Update progress based on completed steps
                if 'query_generation' in steps:
                    queries = steps['query_generation'].get('queries', [])
                    self.write(f"data: {json.dumps({'type': 'research_progress', 'step': 'queries_generated', 'content': f'Generated {len(queries)} search queries'})}\n\n")
                    await self.flush()
                
                if 'web_search' in steps:
                    total_results = steps['web_search'].get('total_results', 0)
                    self.write(f"data: {json.dumps({'type': 'research_progress', 'step': 'search_completed', 'content': f'Found {total_results} search results'})}\n\n")
                    await self.flush()
                
                if 'content_extraction' in steps:
                    successful = steps['content_extraction'].get('successful_extractions', 0)
                    total = steps['content_extraction'].get('total_sources', 0)
                    cache_hits = steps['content_extraction'].get('cache_hits', 0)
                    self.write(f"data: {json.dumps({'type': 'research_progress', 'step': 'extraction_completed', 'content': f'Extracted content from {successful}/{total} sources ({cache_hits} from cache)'})}\n\n")
                    await self.flush()
                
                if 'relevance_evaluation' in steps:
                    high_relevance = steps['relevance_evaluation'].get('high_relevance_count', 0)
                    total_evaluated = steps['relevance_evaluation'].get('total_evaluated', 0)
                    avg_relevance = steps['relevance_evaluation'].get('average_relevance', 0)
                    self.write(f"data: {json.dumps({'type': 'research_progress', 'step': 'relevance_completed', 'content': f'{high_relevance}/{total_evaluated} sources meet relevance threshold (avg: {avg_relevance:.1f}/10)'})}\n\n")
                    await self.flush()
                
                # Prepare final analysis content
                analysis_content = research_results.get('analysis', {})
                if isinstance(analysis_content, dict):
                    final_content = analysis_content.get('analysis', 'Research completed successfully.')
                    confidence = analysis_content.get('confidence', 0.0)
                    sources_used = analysis_content.get('sources_used', 0)
                else:
                    final_content = str(analysis_content) if analysis_content else 'Research completed successfully.'
                    confidence = 0.7
                    sources_used = len(research_results.get('sources', []))
                
                # Add research metadata to the response
                metadata_summary = f"\n\n---\n**Research Summary:**\n"
                metadata_summary += f"- Sources analyzed: {sources_used}\n"
                metadata_summary += f"- Confidence level: {confidence*100:.0f}%\n"
                
                timing_metrics = research_results.get('timing_metrics', {})
                if timing_metrics:
                    total_duration = sum(v for k, v in timing_metrics.items() if k.endswith('_duration'))
                    metadata_summary += f"- Research time: {total_duration:.1f}s\n"
                
                cache_perf = research_results.get('search_metrics', {}).get('cache_performance', {})
                if cache_perf:
                    hit_rate = cache_perf.get('hit_rate', 0) * 100
                    metadata_summary += f"- Cache hit rate: {hit_rate:.1f}%\n"
                
                full_content = final_content + metadata_summary
                
            else:
                # Research failed
                error_msg = research_results.get('error', 'Research failed for unknown reason')
                full_content = f"❌ Research failed: {error_msg}\n\nFalling back to regular chat mode."
                logger.error(f"DeepSeek research failed for chat_id {chat_id}: {error_msg}")
            
            # Send final research complete event
            self.write(f"data: {json.dumps({'type': 'research_complete', 'content': '✅ Research completed!'})}\n\n")
            await self.flush()
            
            # Send the final analysis as regular content chunks (to simulate streaming)
            words = full_content.split()
            accumulated_content = ""
            
            for i, word in enumerate(words):
                accumulated_content += word + " "
                
                # Send chunks of 3-5 words at a time
                if i % 4 == 0 and i > 0:
                    chunk_content = " ".join(words[max(0, i-3):i+1])
                    self.write(f"data: {json.dumps({'type': 'chunk', 'content': chunk_content + ' ', 'chat_id': chat_id, 'message_id': message_id})}\n\n")
                    await self.flush()
                    await asyncio.sleep(0.05)  # Small delay for realistic streaming
            
            # Send final completion
            self.write(f"data: {json.dumps({'type': 'complete', 'content': accumulated_content.strip(), 'chat_id': chat_id, 'message_id': message_id, 'research_data': research_results}, cls=MongoJSONEncoder)}\n\n")
            await self.flush()
            
            # Store the complete AI response in MongoDB
            response_doc = {
                'message_id': str(uuid.uuid4()),
                'chat_id': chat_id,
                'user_id': user_id,
                'message': accumulated_content.strip(),
                'timestamp': datetime.utcnow(),
                'type': 'assistant',
                'search_results': research_results.get('sources', []),
                'research_data': research_results,  # Store full research data
                'shared': False
            }
            
            try:
                await self.application.mongodb.create_message(response_doc)
                logger.info(f"DeepSeek research response stored successfully for chat_id: {chat_id}")
            except Exception as e:
                logger.error(f"Error storing DeepSeek research response: {e}")
            
            # Cleanup resources
            await research_service.cleanup()
            
        except Exception as e:
            logger.error(f"Error in DeepSeek research: {e}")
            logger.error(traceback.format_exc())
            
            # Send error to client
            self.write(f"data: {json.dumps({'type': 'error', 'content': f'Research error: {str(e)}'})}\n\n")
            await self.flush()
    
    async def _handle_deepthink_research(self, message: str, chat_id: str, user_id: str, message_id: str, stream_queue):
        """Handle deep-think research mode with new orchestrator and streaming progress"""
        try:
            # Validate environment configuration first
            validation_errors = []
            
            # Check Serper API key
            serper_api_key = os.environ.get('SERPER_API_KEY')
            if not serper_api_key:
                validation_errors.append("SERPER_API_KEY not configured")
            
            # Check DeepSeek API key
            deepseek_api_key = os.environ.get('DEEPSEEK_API_KEY')
            if not deepseek_api_key:
                validation_errors.append("DEEPSEEK_API_KEY not configured")
            
            # Check DeepSeek API URL
            deepseek_api_url = os.environ.get('DEEPSEEK_API_URL', 'https://api.deepseek.com')
            if not deepseek_api_url.startswith(('http://', 'https://')):
                validation_errors.append("DEEPSEEK_API_URL is malformed")
            
            if validation_errors:
                error_msg = f"Configuration errors: {', '.join(validation_errors)}. Please check your environment variables."
                logger.error(error_msg)
                self.write(f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n")
                await self.flush()
                return
            
            # Send configuration validation success
            self.write(f"data: {json.dumps({'type': 'deepthink_progress', 'step': 0, 'total_steps': 10, 'description': '✅ Configuration validated', 'progress': 5})}\n\n")
            await self.flush()
            
            # Initialize the deep-think orchestrator
            orchestrator = DeepThinkOrchestrator(
                deepseek_service=self.application.deepseek_service,
                mongodb_service=self.application.mongodb,
                serper_api_key=serper_api_key,
                timeout=int(os.environ.get('DEEPSEEK_RESEARCH_TIMEOUT', 600)),
                max_concurrent_searches=int(os.environ.get('MAX_CONCURRENT_RESEARCH', 3)),
                cache_expiry_days=int(os.environ.get('CACHE_EXPIRY_DAYS', 30))
            )
            
            # Create deep-think request
            request = DeepThinkRequest(
                request_id=str(uuid.uuid4()),
                question=message,
                chat_id=chat_id,
                user_id=user_id,
                timestamp=datetime.utcnow(),
                timeout_seconds=int(os.environ.get('DEEPSEEK_RESEARCH_TIMEOUT', 600))
            )
            
            # Stream the deep-think process
            try:
                async for progress_update in orchestrator.stream_deep_think(request):
                    if progress_update.step == orchestrator.total_steps:
                        # Final result
                        result = progress_update.details.get('result')
                        if result:
                            # Format the result for chat interface using structured Answer object
                            confidence = result.get('confidence_score', 0.0)
                            confidence_emoji = "🟢" if confidence >= 0.8 else "🟡" if confidence >= 0.6 else "🔴"
                            
                            # Check if we have structured Answer object (matching test file)
                            answer_obj = result.get('answer')
                            
                            if answer_obj:
                                # Use structured format matching test file Answer structure
                                formatted_parts = [
                                    f"**Deep Think Research Result** {confidence_emoji}",
                                    "",
                                    "## 📋 Answer",
                                    answer_obj.get('content', 'No answer content available'),
                                    "",
                                    f"**Confidence:** {answer_obj.get('confidence', 0.0):.1%}",
                                    ""
                                ]
                                
                                # Add statistics if available
                                if answer_obj.get('statistics'):
                                    stats = answer_obj['statistics']
                                    formatted_parts.extend([
                                        "### 📊 Research Statistics",
                                        ""
                                    ])
                                    if 'sources_analyzed' in stats:
                                        formatted_parts.append(f"• Sources analyzed: {stats['sources_analyzed']}")
                                    if 'high_relevance_sources' in stats:
                                        formatted_parts.append(f"• High relevance sources: {stats['high_relevance_sources']}")
                                    if 'reasoning_chains_used' in stats:
                                        formatted_parts.append(f"• Reasoning chains: {stats['reasoning_chains_used']}")
                                    formatted_parts.append("")
                                
                                # Add gaps if available
                                if answer_obj.get('gaps'):
                                    formatted_parts.extend([
                                        "### ⚠️ Information Gaps",
                                        ""
                                    ])
                                    for gap in answer_obj['gaps'][:3]:
                                        formatted_parts.append(f"• {gap}")
                                    formatted_parts.append("")
                                
                                # Add Conclusion section - use summary_answer as conclusion
                                if result.get('summary_answer'):
                                    formatted_parts.extend([
                                        "## 🎯 Conclusion",
                                        result.get('summary_answer', 'No conclusion available'),
                                        ""
                                    ])
                                
                                # Add expandable comprehensive analysis
                                formatted_parts.extend([
                                    "<details>",
                                    "<summary><strong>📚 Detailed Analysis (Click to expand)</strong></summary>",
                                    "",
                                    result.get('comprehensive_answer', answer_obj.get('content', 'No detailed analysis available')),
                                    "</details>",
                                    ""
                                ])
                                
                                # Add source information
                                if answer_obj.get('sources'):
                                    formatted_parts.extend([
                                        "### 📖 Top Sources",
                                        ""
                                    ])
                                    for i, source in enumerate(answer_obj['sources'][:5], 1):
                                        formatted_parts.append(f"{i}. {source}")
                                    formatted_parts.append("")
                                
                            else:
                                # Fallback to old format if structured objects not available
                                formatted_parts = [
                                    f"**Deep Think Research Result** {confidence_emoji}",
                                    "",
                                    "## 📋 Summary",
                                    result.get('summary_answer', result.get('summary', 'No summary available')),
                                    "",
                                    "<details>",
                                    "<summary><strong>📚 Comprehensive Analysis (Click to expand)</strong></summary>",
                                    "",
                                    result.get('comprehensive_answer', 'No comprehensive answer available'),
                                    "</details>",
                                    ""
                                ]
                            
                            # Add metadata if available
                            if 'total_sources' in result:
                                formatted_parts.append(f"*Sources analyzed: {result['total_sources']}*")
                            if 'processing_time' in result:
                                formatted_parts.append(f"*Processing time: {result['processing_time']:.1f}s*")
                            
                            formatted_response = "\n".join(formatted_parts)
                            
                            # Stream the formatted response word by word
                            words = formatted_response.split()
                            accumulated_content = ""
                            
                            for i, word in enumerate(words):
                                accumulated_content += word + " "
                                
                                # Send chunks of 3-5 words at a time
                                if i % 4 == 0 and i > 0:
                                    chunk_content = " ".join(words[max(0, i-3):i+1])
                                    self.write(f"data: {json.dumps({'type': 'chunk', 'content': chunk_content + ' ', 'chat_id': chat_id, 'message_id': message_id})}\n\n")
                                    await self.flush()
                                    await asyncio.sleep(0.03)  # Small delay for realistic streaming
                            
                            # Send final completion - serialize result properly
                            result_dict = result.to_dict() if hasattr(result, 'to_dict') else result
                            self.write(f"data: {json.dumps({'type': 'complete', 'content': accumulated_content.strip(), 'chat_id': chat_id, 'message_id': message_id, 'deepthink_result': result_dict}, cls=MongoJSONEncoder)}\n\n")
                            await self.flush()
                            
                            # Store the complete AI response in MongoDB
                            response_doc = {
                                'message_id': str(uuid.uuid4()),
                                'chat_id': chat_id,
                                'user_id': user_id,
                                'message': accumulated_content.strip(),
                                'timestamp': datetime.utcnow(),
                                'type': 'assistant',
                                'search_results': result.get('scraped_content', []),
                                'deepthink_data': result,  # Store full deep-think data
                                'shared': False
                            }
                            
                            try:
                                await self.application.mongodb.create_message(response_doc)
                                logger.info(f"Deep-think response stored successfully for chat_id: {chat_id}")
                            except Exception as e:
                                logger.error(f"Error storing deep-think response: {e}")
                        else:
                            # No result in final update - handle error
                            error_detail = progress_update.details.get('error', 'Unknown error')
                            self.write(f"data: {json.dumps({'type': 'error', 'content': f'Deep-think failed: {error_detail}'})}\n\n")
                            await self.flush()
                    else:
                        # Progress update
                        progress_data = {
                            'type': 'deepthink_progress',
                            'step': progress_update.step,
                            'total_steps': progress_update.total_steps,
                            'description': progress_update.description,
                            'progress': progress_update.progress_percent,
                            'details': progress_update.details
                        }
                        self.write(f"data: {json.dumps(progress_data, cls=MongoJSONEncoder)}\n\n")
                        await self.flush()
                        
            except asyncio.TimeoutError:
                timeout_error = f"Deep-think research timed out after {request.timeout_seconds} seconds"
                logger.error(timeout_error)
                
                # Provide helpful suggestions for timeout
                timeout_details = {
                    'type': 'error',
                    'content': f"{timeout_error}\n\nThis usually happens when:\n"
                              "• API connections are slow or failing\n"
                              "• The research topic requires extensive analysis\n"
                              "• Network connectivity issues\n\n"
                              "💡 Try: Simplifying your question or checking your internet connection",
                    'error_type': 'timeout',
                    'timeout_seconds': request.timeout_seconds
                }
                self.write(f"data: {json.dumps(timeout_details)}\n\n")
                await self.flush()
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Deep-think processing error: {error_msg}")
                logger.error(traceback.format_exc())
                
                # Provide specific error handling
                if "Connection error" in error_msg or "nodename nor servname" in error_msg:
                    friendly_error = {
                        'type': 'error',
                        'content': "🔌 Connection Error: Cannot reach DeepSeek API\n\n"
                                  "This usually means:\n"
                                  "• API endpoint configuration is wrong\n"
                                  "• Network connectivity issues\n"
                                  "• DNS resolution problems\n\n"
                                  "💡 Try: Check your DEEPSEEK_API_URL environment variable",
                        'error_type': 'connection',
                        'technical_details': error_msg
                    }
                elif "timeout" in error_msg.lower():
                    friendly_error = {
                        'type': 'error',
                        'content': "⏱️ Request Timeout: API calls are taking too long\n\n"
                                  "This could be due to:\n"
                                  "• Slow network connection\n"
                                  "• API service overload\n"
                                  "• Complex research query\n\n"
                                  "💡 Try: Simplifying your question or trying again later",
                        'error_type': 'timeout',
                        'technical_details': error_msg
                    }
                else:
                    friendly_error = {
                        'type': 'error',
                        'content': f"⚠️ Processing Error: {error_msg}\n\n"
                                  "An unexpected error occurred during research.\n"
                                  "Please try again or check the logs for more details.",
                        'error_type': 'processing',
                        'technical_details': error_msg
                    }
                
                self.write(f"data: {json.dumps(friendly_error)}\n\n")
                await self.flush()
                
        except Exception as e:
            logger.error(f"Error in deep-think research: {e}")
            logger.error(traceback.format_exc())
            
            # Send error to client
            self.write(f"data: {json.dumps({'type': 'error', 'content': f'Deep-think error: {str(e)}'})}\n\n")
            await self.flush()

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
