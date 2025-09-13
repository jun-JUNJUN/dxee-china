#!/usr/bin/env python3
"""
Session-Chat Integration Handler - Bridge between sessions and chat history

Provides endpoints and functionality to integrate Deep Think sessions with
chat history, allowing users to view, search, and manage their research sessions.

Key Features:
- Session listing and filtering
- Session-to-chat conversion
- Session search and analytics  
- Chat context enhancement with session data
- Session management interface
"""

import tornado.web
import json
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List

from ..service.session_manager_service import SessionManagerService
from ..service.enhanced_chat_storage import EnhancedChatStorageService
from ..models.session_models import SessionStatus

logger = logging.getLogger(__name__)

# Custom JSON encoder for MongoDB objects
class MongoJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class UserSessionsHandler(tornado.web.RequestHandler):
    """
    Handler for user's Deep Think sessions
    
    Provides session listing, filtering, and status checking
    """
    
    async def get(self):
        """Get user's Deep Think sessions with filtering"""
        try:
            # Get user authentication
            user_id = self.get_secure_cookie("user_id")
            if user_id:
                user_id = user_id.decode('utf-8')
            else:
                user_id = "default_user"
            
            # Parse query parameters
            limit = int(self.get_argument('limit', 20))
            status_filter = self.get_argument('status', None)
            days_back = int(self.get_argument('days_back', 7))
            include_completed = self.get_argument('include_completed', 'true').lower() == 'true'
            include_failed = self.get_argument('include_failed', 'false').lower() == 'true'
            
            # Initialize session manager
            session_manager = SessionManagerService(self.application.mongodb)
            await session_manager.initialize()
            
            # Build filter criteria
            filter_status = None
            if status_filter and status_filter in ['pending', 'active', 'completed', 'failed', 'timeout']:
                filter_status = SessionStatus(status_filter)
            
            # Get sessions
            all_sessions = await session_manager.get_user_sessions(user_id, limit=limit * 2)  # Get more for filtering
            
            # Apply filters
            filtered_sessions = []
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
            
            for session in all_sessions:
                # Date filter
                if session.created_at < cutoff_date:
                    continue
                
                # Status filter
                if filter_status and session.status != filter_status:
                    continue
                
                # Include completed/failed filters
                if session.status == SessionStatus.COMPLETED and not include_completed:
                    continue
                if session.status in [SessionStatus.FAILED, SessionStatus.TIMEOUT] and not include_failed:
                    continue
                
                filtered_sessions.append(session)
                
                if len(filtered_sessions) >= limit:
                    break
            
            # Convert to response format
            session_list = []
            for session in filtered_sessions:
                session_data = {
                    'session_id': session.session_id,
                    'question': session.question[:100] + ('...' if len(session.question) > 100 else ''),
                    'full_question': session.question,
                    'status': session.status.value,
                    'created_at': session.created_at.isoformat(),
                    'updated_at': session.updated_at.isoformat(),
                    'is_active': session.is_active(),
                    'is_completed': session.is_completed(),
                    'chat_id': session.chat_id,
                    'duration': session.get_duration()
                }
                
                # Add progress info if available
                if session.current_progress:
                    session_data['current_progress'] = {
                        'step': session.current_progress.current_step.value,
                        'progress': session.current_progress.progress_percent,
                        'description': session.current_progress.description
                    }
                
                # Add result summary if completed
                if session.result:
                    session_data['result_summary'] = {
                        'confidence': session.result.confidence,
                        'sources_count': len(session.result.sources),
                        'has_statistics': len(session.result.statistics.get('numbers_found', [])) > 0,
                        'processing_time': session.result.total_duration
                    }
                
                # Add error if failed
                if session.error_message:
                    session_data['error_message'] = session.error_message
                
                session_list.append(session_data)
            
            response = {
                'sessions': session_list,
                'total_count': len(session_list),
                'filtered_count': len(session_list),
                'filters_applied': {
                    'status': status_filter,
                    'days_back': days_back,
                    'include_completed': include_completed,
                    'include_failed': include_failed
                }
            }
            
            self.write(json.dumps(response, cls=MongoJSONEncoder))
            
        except Exception as e:
            logger.error(f"Error getting user sessions: {e}")
            self.set_status(500)
            self.write({'error': f'Failed to get sessions: {str(e)}'})


class SessionChatConvertHandler(tornado.web.RequestHandler):
    """
    Handler for converting sessions to chat messages
    
    Allows users to convert completed sessions into chat messages
    """
    
    async def post(self):
        """Convert session result to chat message"""
        try:
            # Parse request
            data = json.loads(self.request.body)
            session_id = data.get('session_id')
            target_chat_id = data.get('chat_id')  # Optional: specify target chat
            
            user_id = self.get_secure_cookie("user_id")
            if user_id:
                user_id = user_id.decode('utf-8')
            else:
                user_id = "default_user"
            
            if not session_id:
                self.set_status(400)
                self.write({'error': 'session_id is required'})
                return
            
            # Initialize services
            session_manager = SessionManagerService(self.application.mongodb)
            await session_manager.initialize()
            
            enhanced_storage = EnhancedChatStorageService(self.application.mongodb)
            await enhanced_storage.initialize()
            
            # Get session
            session = await session_manager.get_session(session_id)
            if not session:
                self.set_status(404)
                self.write({'error': 'Session not found'})
                return
            
            if session.user_id != user_id:
                self.set_status(403)
                self.write({'error': 'Unauthorized'})
                return
            
            if not session.is_completed() or not session.result:
                self.set_status(400)
                self.write({'error': 'Session is not completed or has no result'})
                return
            
            # Use target chat or create new one
            if target_chat_id:
                # Verify chat exists and belongs to user
                existing_chat = await self.application.mongodb.get_chat_by_id(target_chat_id)
                if not existing_chat or existing_chat.get('user_id') != user_id:
                    self.set_status(403)
                    self.write({'error': 'Invalid or unauthorized target chat'})
                    return
                chat_id = target_chat_id
            else:
                # Create new chat
                chat_id = str(uuid.uuid4())
                chat_doc = {
                    'chat_id': chat_id,
                    'user_id': user_id,
                    'title': f"Deep Think: {session.question[:30]}..."
                }
                await self.application.mongodb.create_chat(chat_doc)
            
            # Store user question message first
            user_message_doc = {
                'message_id': str(uuid.uuid4()),
                'chat_id': chat_id,
                'user_id': user_id,
                'message': session.question,
                'timestamp': session.created_at,
                'type': 'user',
                'shared': False,
                'converted_from_session': session_id
            }
            await self.application.mongodb.create_message(user_message_doc)
            
            # Store enhanced Deep Think result
            message_id = await enhanced_storage.store_deepthink_result(session, session.result)
            
            # Update session metadata to track conversion
            session.session_metadata['converted_to_chat'] = {
                'chat_id': chat_id,
                'message_id': message_id,
                'converted_at': datetime.now(timezone.utc).isoformat()
            }
            
            response = {
                'success': True,
                'chat_id': chat_id,
                'message_id': message_id,
                'user_message_id': user_message_doc['message_id'],
                'session_id': session_id,
                'created_new_chat': target_chat_id is None
            }
            
            self.write(json.dumps(response, cls=MongoJSONEncoder))
            logger.info(f"Converted session {session_id} to chat {chat_id}")
            
        except Exception as e:
            logger.error(f"Error converting session to chat: {e}")
            self.set_status(500)
            self.write({'error': f'Conversion failed: {str(e)}'})


class SessionAnalyticsHandler(tornado.web.RequestHandler):
    """
    Handler for session analytics and statistics
    """
    
    async def get(self):
        """Get user's Deep Think analytics"""
        try:
            user_id = self.get_secure_cookie("user_id")
            if user_id:
                user_id = user_id.decode('utf-8')
            else:
                user_id = "default_user"
            
            # Initialize services
            session_manager = SessionManagerService(self.application.mongodb)
            await session_manager.initialize()
            
            enhanced_storage = EnhancedChatStorageService(self.application.mongodb)
            await enhanced_storage.initialize()
            
            # Get session analytics
            session_stats = await session_manager.get_session_statistics()
            
            # Get Deep Think chat analytics
            chat_analytics = await enhanced_storage.get_deepthink_analytics(user_id)
            
            # Get recent sessions summary
            recent_sessions = await session_manager.get_user_sessions(user_id, limit=10)
            recent_summary = {
                'total_recent': len(recent_sessions),
                'active_count': len([s for s in recent_sessions if s.is_active()]),
                'completed_count': len([s for s in recent_sessions if s.status == SessionStatus.COMPLETED]),
                'failed_count': len([s for s in recent_sessions if s.status in [SessionStatus.FAILED, SessionStatus.TIMEOUT]])
            }
            
            # Calculate usage patterns
            if recent_sessions:
                avg_processing_time = sum(
                    s.result.total_duration for s in recent_sessions 
                    if s.result and s.result.total_duration
                ) / len([s for s in recent_sessions if s.result and s.result.total_duration])
                
                usage_patterns = {
                    'avg_processing_time': avg_processing_time,
                    'most_recent_session': recent_sessions[0].created_at.isoformat(),
                    'session_frequency': len(recent_sessions) / 7,  # Sessions per day over last week
                    'avg_question_length': sum(len(s.question) for s in recent_sessions) / len(recent_sessions)
                }
            else:
                usage_patterns = {
                    'avg_processing_time': 0,
                    'most_recent_session': None,
                    'session_frequency': 0,
                    'avg_question_length': 0
                }
            
            response = {
                'user_id': user_id,
                'session_statistics': session_stats,
                'chat_analytics': chat_analytics,
                'recent_summary': recent_summary,
                'usage_patterns': usage_patterns,
                'generated_at': datetime.now(timezone.utc).isoformat()
            }
            
            self.write(json.dumps(response, cls=MongoJSONEncoder))
            
        except Exception as e:
            logger.error(f"Error getting session analytics: {e}")
            self.set_status(500)
            self.write({'error': f'Analytics failed: {str(e)}'})


class ChatSessionEnhancementHandler(tornado.web.RequestHandler):
    """
    Handler for enhancing chat views with session data
    """
    
    async def get(self):
        """Get enhanced chat history with session context"""
        try:
            chat_id = self.get_argument('chat_id', None)
            user_id = self.get_secure_cookie("user_id")
            
            if user_id:
                user_id = user_id.decode('utf-8')
            else:
                user_id = "default_user"
            
            if not chat_id:
                self.set_status(400)
                self.write({'error': 'chat_id is required'})
                return
            
            # Initialize enhanced storage
            enhanced_storage = EnhancedChatStorageService(self.application.mongodb)
            await enhanced_storage.initialize()
            
            # Get enhanced chat with Deep Think context
            enhanced_chat = await enhanced_storage.get_chat_with_deepthink_context(chat_id, user_id)
            
            self.write(json.dumps(enhanced_chat, cls=MongoJSONEncoder))
            
        except Exception as e:
            logger.error(f"Error getting enhanced chat: {e}")
            self.set_status(500)
            self.write({'error': f'Enhancement failed: {str(e)}'})


class SessionSearchHandler(tornado.web.RequestHandler):
    """
    Handler for searching sessions and Deep Think results
    """
    
    async def get(self):
        """Search sessions and results"""
        try:
            search_query = self.get_argument('q', '')
            search_type = self.get_argument('type', 'all')  # 'sessions', 'results', 'all'
            limit = int(self.get_argument('limit', 20))
            
            user_id = self.get_secure_cookie("user_id")
            if user_id:
                user_id = user_id.decode('utf-8')
            else:
                user_id = "default_user"
            
            if not search_query:
                self.set_status(400)
                self.write({'error': 'Search query (q) is required'})
                return
            
            results = {'query': search_query, 'type': search_type}
            
            # Search sessions
            if search_type in ['sessions', 'all']:
                session_manager = SessionManagerService(self.application.mongodb)
                await session_manager.initialize()
                
                # Get all user sessions and filter by search query
                all_sessions = await session_manager.get_user_sessions(user_id, limit=100)
                matching_sessions = []
                
                for session in all_sessions:
                    if search_query.lower() in session.question.lower():
                        session_data = {
                            'session_id': session.session_id,
                            'question': session.question,
                            'status': session.status.value,
                            'created_at': session.created_at.isoformat(),
                            'match_type': 'question'
                        }
                        
                        if session.result:
                            session_data['result_summary'] = {
                                'confidence': session.result.confidence,
                                'sources_count': len(session.result.sources)
                            }
                        
                        matching_sessions.append(session_data)
                
                results['sessions'] = matching_sessions[:limit]
            
            # Search Deep Think results in chat
            if search_type in ['results', 'all']:
                enhanced_storage = EnhancedChatStorageService(self.application.mongodb)
                await enhanced_storage.initialize()
                
                matching_results = await enhanced_storage.search_deepthink_results(
                    user_id, search_query, limit
                )
                
                # Format results
                formatted_results = []
                for result in matching_results:
                    formatted_result = {
                        'message_id': result['message_id'],
                        'chat_id': result['chat_id'],
                        'question': result.get('deepthink_data', {}).get('original_question', ''),
                        'confidence': result.get('deepthink_data', {}).get('confidence', 0),
                        'timestamp': result['timestamp'],
                        'match_score': result.get('score', 0),
                        'preview': result['message'][:200] + ('...' if len(result['message']) > 200 else '')
                    }
                    formatted_results.append(formatted_result)
                
                results['results'] = formatted_results
            
            self.write(json.dumps(results, cls=MongoJSONEncoder))
            
        except Exception as e:
            logger.error(f"Error searching sessions: {e}")
            self.set_status(500)
            self.write({'error': f'Search failed: {str(e)}'})


class SessionManagementHandler(tornado.web.RequestHandler):
    """
    Handler for session management operations
    """
    
    async def delete(self):
        """Delete a session"""
        try:
            session_id = self.get_argument('session_id', None)
            user_id = self.get_secure_cookie("user_id")
            
            if user_id:
                user_id = user_id.decode('utf-8')
            else:
                user_id = "default_user"
            
            if not session_id:
                self.set_status(400)
                self.write({'error': 'session_id is required'})
                return
            
            # Initialize session manager
            session_manager = SessionManagerService(self.application.mongodb)
            await session_manager.initialize()
            
            # Verify session belongs to user
            session = await session_manager.get_session(session_id)
            if not session:
                self.set_status(404)
                self.write({'error': 'Session not found'})
                return
            
            if session.user_id != user_id:
                self.set_status(403)
                self.write({'error': 'Unauthorized'})
                return
            
            # Cancel active session or delete completed session
            if session.is_active():
                await session_manager.cancel_session(session_id)
                action = 'cancelled'
            else:
                # Delete from database
                await self.application.mongodb.db.processing_sessions.delete_one(
                    {'session_id': session_id}
                )
                action = 'deleted'
            
            response = {
                'success': True,
                'session_id': session_id,
                'action': action
            }
            
            self.write(json.dumps(response))
            logger.info(f"Session {session_id} {action} by user {user_id}")
            
        except Exception as e:
            logger.error(f"Error managing session: {e}")
            self.set_status(500)
            self.write({'error': f'Session management failed: {str(e)}'})