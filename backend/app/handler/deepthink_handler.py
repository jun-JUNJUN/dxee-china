#!/usr/bin/env python3
"""
Deep-think API handlers for enhanced chat history and analytics
"""

import tornado.web
import json
import logging
import traceback
from datetime import datetime
from bson import ObjectId, json_util

# Get logger
logger = logging.getLogger(__name__)


class MongoJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for MongoDB objects"""
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return json_util.default(obj)


class DeepThinkChatHistoryHandler(tornado.web.RequestHandler):
    """
    Enhanced chat history handler that includes deep-think metadata
    """
    async def get(self, chat_id):
        try:
            logger.info(f"Getting enhanced chat history for chat_id: {chat_id}")
            
            # Get user_id from authentication
            user_id = self.get_secure_cookie("user_id")
            if user_id:
                user_id = user_id.decode('utf-8')
            else:
                user_id = "default_user"
            
            # Get pagination parameters
            limit = int(self.get_argument('limit', 100))
            skip = int(self.get_argument('skip', 0))
            
            # Get enhanced messages with deep-think data
            try:
                messages = await self.application.mongodb.get_chat_messages_with_deepthink(
                    chat_id, limit, skip
                )
                
                logger.info(f"Retrieved {len(messages)} enhanced messages for chat_id: {chat_id}")
                
                # Convert MongoDB documents to JSON-serializable format
                for message in messages:
                    message['_id'] = str(message['_id'])
                    if isinstance(message.get('timestamp'), datetime):
                        message['timestamp'] = message['timestamp'].isoformat()
                
                # Get deep-think summary for this chat
                deepthink_messages = [m for m in messages if m.get('has_deepthink')]
                deepthink_summary = {
                    'total_deepthink_messages': len(deepthink_messages),
                    'avg_confidence': (
                        sum(m['deepthink_summary']['confidence_score'] for m in deepthink_messages) / 
                        len(deepthink_messages) if deepthink_messages else 0.0
                    ),
                    'avg_sources': (
                        sum(m['deepthink_summary']['total_sources'] for m in deepthink_messages) /
                        len(deepthink_messages) if deepthink_messages else 0.0
                    ),
                    'avg_processing_time': (
                        sum(m['deepthink_summary']['processing_time'] for m in deepthink_messages) /
                        len(deepthink_messages) if deepthink_messages else 0.0
                    )
                }
                
                response = {
                    'chat_id': chat_id,
                    'messages': messages,
                    'deepthink_summary': deepthink_summary,
                    'total_messages': len(messages)
                }
                
                self.write(json.dumps(response, cls=MongoJSONEncoder))
                self.set_header('Content-Type', 'application/json')
                
            except Exception as e:
                logger.error(f"Error retrieving enhanced messages from MongoDB: {e}")
                logger.error(traceback.format_exc())
                self.set_status(500)
                self.write({'error': f'Error retrieving enhanced chat history: {str(e)}'})
        
        except Exception as e:
            logger.error(f"Unexpected error in DeepThinkChatHistoryHandler: {e}")
            logger.error(traceback.format_exc())
            self.set_status(500)
            self.write({'error': f'Server error: {str(e)}'})


class DeepThinkAnalyticsHandler(tornado.web.RequestHandler):
    """
    Handler for deep-think analytics and usage statistics
    """
    async def get(self):
        try:
            # Get user_id from authentication
            user_id = self.get_secure_cookie("user_id")
            if user_id:
                user_id = user_id.decode('utf-8')
            else:
                user_id = "default_user"
            
            # Get time period parameter
            days = int(self.get_argument('days', 30))
            
            logger.info(f"Getting deep-think analytics for user_id: {user_id}, period: {days} days")
            
            try:
                # Get user's deep-think usage summary
                usage_summary = await self.application.mongodb.get_user_deepthink_chat_summary(
                    user_id, days
                )
                
                if usage_summary:
                    logger.info(f"Retrieved analytics for user {user_id}: {usage_summary['total_deepthink_requests']} requests")
                    
                    self.write(json.dumps({
                        'user_id': user_id,
                        'analytics': usage_summary,
                        'generated_at': datetime.utcnow().isoformat()
                    }, cls=MongoJSONEncoder))
                    self.set_header('Content-Type', 'application/json')
                else:
                    # No data found
                    self.write(json.dumps({
                        'user_id': user_id,
                        'analytics': None,
                        'message': 'No deep-think usage data found for this period',
                        'generated_at': datetime.utcnow().isoformat()
                    }))
                    self.set_header('Content-Type', 'application/json')
                
            except Exception as e:
                logger.error(f"Error retrieving analytics from MongoDB: {e}")
                logger.error(traceback.format_exc())
                self.set_status(500)
                self.write({'error': f'Error retrieving analytics: {str(e)}'})
        
        except Exception as e:
            logger.error(f"Unexpected error in DeepThinkAnalyticsHandler: {e}")
            logger.error(traceback.format_exc())
            self.set_status(500)
            self.write({'error': f'Server error: {str(e)}'})


class DeepThinkSearchHandler(tornado.web.RequestHandler):
    """
    Handler for searching across deep-think results
    """
    async def get(self):
        try:
            # Get user_id from authentication
            user_id = self.get_secure_cookie("user_id")
            if user_id:
                user_id = user_id.decode('utf-8')
            else:
                user_id = "default_user"
            
            # Get search parameters
            query = self.get_argument('q', '').strip()
            limit = int(self.get_argument('limit', 10))
            user_filter = self.get_argument('user_filter', 'true').lower() == 'true'
            
            if not query:
                self.set_status(400)
                self.write({'error': 'Search query (q) is required'})
                return
            
            logger.info(f"Searching deep-think content: '{query}' for user: {user_id if user_filter else 'all'}")
            
            try:
                # Search deep-think content
                search_results = await self.application.mongodb.search_deepthink_content(
                    query=query,
                    user_id=user_id if user_filter else None,
                    limit=limit
                )
                
                logger.info(f"Found {len(search_results)} deep-think search results")
                
                # Convert timestamps to ISO format
                for result in search_results:
                    if isinstance(result.get('timestamp'), datetime):
                        result['timestamp'] = result['timestamp'].isoformat()
                
                response = {
                    'query': query,
                    'user_id': user_id if user_filter else None,
                    'results': search_results,
                    'total_results': len(search_results),
                    'searched_at': datetime.utcnow().isoformat()
                }
                
                self.write(json.dumps(response, cls=MongoJSONEncoder))
                self.set_header('Content-Type', 'application/json')
                
            except Exception as e:
                logger.error(f"Error searching deep-think content: {e}")
                logger.error(traceback.format_exc())
                self.set_status(500)
                self.write({'error': f'Error searching content: {str(e)}'})
        
        except Exception as e:
            logger.error(f"Unexpected error in DeepThinkSearchHandler: {e}")
            logger.error(traceback.format_exc())
            self.set_status(500)
            self.write({'error': f'Server error: {str(e)}'})


class DeepThinkResultHandler(tornado.web.RequestHandler):
    """
    Handler for retrieving specific deep-think results
    """
    async def get(self, request_id):
        try:
            logger.info(f"Getting deep-think result for request_id: {request_id}")
            
            # Get user_id from authentication
            user_id = self.get_secure_cookie("user_id")
            if user_id:
                user_id = user_id.decode('utf-8')
            else:
                user_id = "default_user"
            
            try:
                # Get the deep-think result
                result = await self.application.mongodb.get_deepthink_result(request_id)
                
                if result:
                    logger.info(f"Retrieved deep-think result for request_id: {request_id}")
                    
                    # Convert MongoDB document to JSON-serializable format
                    result['_id'] = str(result['_id'])
                    if isinstance(result.get('created_at'), datetime):
                        result['created_at'] = result['created_at'].isoformat()
                    if isinstance(result.get('updated_at'), datetime):
                        result['updated_at'] = result['updated_at'].isoformat()
                    if isinstance(result.get('timestamp'), datetime):
                        result['timestamp'] = result['timestamp'].isoformat()
                    
                    self.write(json.dumps({
                        'request_id': request_id,
                        'result': result
                    }, cls=MongoJSONEncoder))
                    self.set_header('Content-Type', 'application/json')
                else:
                    logger.warning(f"Deep-think result not found: {request_id}")
                    self.set_status(404)
                    self.write({'error': 'Deep-think result not found'})
                
            except Exception as e:
                logger.error(f"Error retrieving deep-think result from MongoDB: {e}")
                logger.error(traceback.format_exc())
                self.set_status(500)
                self.write({'error': f'Error retrieving result: {str(e)}'})
        
        except Exception as e:
            logger.error(f"Unexpected error in DeepThinkResultHandler: {e}")
            logger.error(traceback.format_exc())
            self.set_status(500)
            self.write({'error': f'Server error: {str(e)}'})


class ChatDeepThinkMessagesHandler(tornado.web.RequestHandler):
    """
    Handler for retrieving only deep-think messages from a specific chat
    """
    async def get(self, chat_id):
        try:
            logger.info(f"Getting deep-think messages for chat_id: {chat_id}")
            
            # Get user_id from authentication
            user_id = self.get_secure_cookie("user_id")
            if user_id:
                user_id = user_id.decode('utf-8')
            else:
                user_id = "default_user"
            
            try:
                # Get deep-think messages for this chat
                deepthink_messages = await self.application.mongodb.get_deepthink_messages_for_chat(chat_id)
                
                logger.info(f"Retrieved {len(deepthink_messages)} deep-think messages for chat_id: {chat_id}")
                
                # Convert MongoDB documents to JSON-serializable format
                for message in deepthink_messages:
                    message['_id'] = str(message['_id'])
                    if isinstance(message.get('timestamp'), datetime):
                        message['timestamp'] = message['timestamp'].isoformat()
                
                response = {
                    'chat_id': chat_id,
                    'deepthink_messages': deepthink_messages,
                    'total_deepthink_messages': len(deepthink_messages)
                }
                
                self.write(json.dumps(response, cls=MongoJSONEncoder))
                self.set_header('Content-Type', 'application/json')
                
            except Exception as e:
                logger.error(f"Error retrieving deep-think messages from MongoDB: {e}")
                logger.error(traceback.format_exc())
                self.set_status(500)
                self.write({'error': f'Error retrieving deep-think messages: {str(e)}'})
        
        except Exception as e:
            logger.error(f"Unexpected error in ChatDeepThinkMessagesHandler: {e}")
            logger.error(traceback.format_exc())
            self.set_status(500)
            self.write({'error': f'Server error: {str(e)}'})