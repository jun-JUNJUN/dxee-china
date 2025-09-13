#!/usr/bin/env python3
"""
Streaming Reconnection Handler - Handles SSE reconnection for Deep Think sessions

Provides endpoints and logic for clients to reconnect to ongoing Deep Think sessions
and receive buffered progress messages they may have missed.

Key Features:
- Session reconnection with message replay
- Progress continuation from last received message
- Graceful handling of expired sessions
- Connection validation and restoration
"""

import tornado.web
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

from ..service.progress_streaming_service import ProgressStreamingService
from ..service.session_manager_service import SessionManagerService
from ..models.session_models import SessionStatus

logger = logging.getLogger(__name__)

# Custom JSON encoder for MongoDB objects
class MongoJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class StreamingReconnectHandler(tornado.web.RequestHandler):
    """
    Handler for streaming reconnection requests
    
    Allows clients to reconnect to ongoing Deep Think sessions and receive
    buffered messages they may have missed during disconnection.
    """
    
    async def get(self):
        """Handle streaming reconnection via GET request with SSE"""
        try:
            # Parse query parameters
            session_id = self.get_argument('session_id', None)
            user_id = self.get_secure_cookie("user_id")
            last_message_timestamp = self.get_argument('last_timestamp', None)
            
            if user_id:
                user_id = user_id.decode('utf-8')
            else:
                user_id = "default_user"
            
            # Validate session_id
            if not session_id:
                self.set_status(400)
                self.write({'error': 'session_id is required'})
                return
            
            logger.info(f"Reconnection request for session {session_id} from user {user_id}")
            
            # Initialize services
            session_manager = SessionManagerService(self.application.mongodb)
            await session_manager.initialize()
            
            streaming_service = ProgressStreamingService()
            await streaming_service.start()
            
            # Verify session exists and belongs to user
            session = await session_manager.get_session(session_id)
            if not session:
                await self._send_error("Session not found")
                return
            
            if session.user_id != user_id:
                await self._send_error("Unauthorized: Session belongs to different user")
                return
            
            # Check session status
            if session.is_completed():
                await self._send_session_completed(session, streaming_service)
                return
            
            if not session.is_active():
                await self._send_error(f"Session is not active (status: {session.status.value})")
                return
            
            # Create new streaming connection
            connection_id = await streaming_service.create_streaming_connection(
                session_id=session_id,
                user_id=user_id,
                chat_id=session.chat_id,
                handler=self
            )
            
            logger.info(f"Created reconnection streaming connection {connection_id} for session {session_id}")
            
            # Send buffered messages if available
            await self._replay_buffered_messages(streaming_service, session_id, last_message_timestamp)
            
            # Keep connection alive for new messages
            await self._maintain_connection(streaming_service, connection_id, session_id)
            
        except Exception as e:
            logger.error(f"Streaming reconnection error: {e}")
            await self._send_error(f"Reconnection failed: {str(e)}")
    
    async def post(self):
        """Handle reconnection status check via POST"""
        try:
            # Parse request body
            data = json.loads(self.request.body)
            session_id = data.get('session_id')
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
            
            # Get session status
            session = await session_manager.get_session(session_id)
            if not session:
                self.write({
                    'exists': False,
                    'error': 'Session not found'
                })
                return
            
            if session.user_id != user_id:
                self.write({
                    'exists': False,
                    'error': 'Unauthorized'
                })
                return
            
            # Return session status
            response_data = {
                'exists': True,
                'session_id': session_id,
                'status': session.status.value,
                'is_active': session.is_active(),
                'is_completed': session.is_completed(),
                'created_at': session.created_at.isoformat(),
                'updated_at': session.updated_at.isoformat()
            }
            
            # Add progress info if available
            if session.current_progress:
                response_data['current_progress'] = {
                    'step': session.current_progress.current_step.value,
                    'progress': session.current_progress.progress_percent,
                    'description': session.current_progress.description,
                    'timestamp': session.current_progress.timestamp.isoformat()
                }
            
            # Add result info if completed
            if session.result:
                response_data['result'] = {
                    'confidence': session.result.confidence,
                    'sources_count': len(session.result.sources),
                    'processing_time': session.result.total_duration
                }
            
            self.write(json.dumps(response_data, cls=MongoJSONEncoder))
            
        except Exception as e:
            logger.error(f"Session status check error: {e}")
            self.set_status(500)
            self.write({'error': f'Status check failed: {str(e)}'})
    
    async def _send_error(self, error_message: str):
        """Send error message via SSE"""
        error_data = {
            'type': 'error',
            'error': error_message,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        self.set_header('Content-Type', 'text/event-stream')
        self.set_header('Cache-Control', 'no-cache')
        self.set_header('Connection', 'keep-alive')
        
        self.write(f"data: {json.dumps(error_data)}\n\n")
        await self.flush()
    
    async def _send_session_completed(self, session, streaming_service: ProgressStreamingService):
        """Send session completion information"""
        completion_data = {
            'type': 'session_completed',
            'session_id': session.session_id,
            'status': session.status.value,
            'completed_at': session.completed_at.isoformat() if session.completed_at else None,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Add result info if available
        if session.result:
            completion_data['result'] = {
                'confidence': session.result.confidence,
                'sources_count': len(session.result.sources),
                'processing_time': session.result.total_duration
            }
        
        self.set_header('Content-Type', 'text/event-stream')
        self.set_header('Cache-Control', 'no-cache')
        self.set_header('Connection', 'keep-alive')
        
        self.write(f"data: {json.dumps(completion_data, cls=MongoJSONEncoder)}\n\n")
        await self.flush()
    
    async def _replay_buffered_messages(self, streaming_service: ProgressStreamingService, 
                                      session_id: str, last_timestamp: Optional[str]):
        """Replay buffered messages from last timestamp"""
        try:
            if session_id not in streaming_service.message_buffer:
                logger.info(f"No buffered messages for session {session_id}")
                return
            
            buffered_messages = streaming_service.message_buffer[session_id]
            
            if not buffered_messages:
                logger.info(f"Empty message buffer for session {session_id}")
                return
            
            # Parse last timestamp
            replay_from_timestamp = None
            if last_timestamp:
                try:
                    replay_from_timestamp = datetime.fromisoformat(last_timestamp.replace('Z', '+00:00'))
                except ValueError:
                    logger.warning(f"Invalid last timestamp format: {last_timestamp}")
            
            # Replay messages newer than last timestamp
            replayed_count = 0
            for message in buffered_messages:
                if replay_from_timestamp is None or message.timestamp > replay_from_timestamp:
                    # Send replay marker for first message
                    if replayed_count == 0:
                        replay_marker = {
                            'type': 'replay_start',
                            'session_id': session_id,
                            'replayed_from': last_timestamp,
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        }
                        self.write(f"data: {json.dumps(replay_marker)}\n\n")
                        await self.flush()
                    
                    # Send buffered message
                    replay_message = {
                        'type': 'replay_message',
                        'original_type': message.message_type,
                        'session_id': message.session_id,
                        'timestamp': message.timestamp.isoformat(),
                        **message.data
                    }
                    self.write(f"data: {json.dumps(replay_message, cls=MongoJSONEncoder)}\n\n")
                    await self.flush()
                    replayed_count += 1
            
            # Send replay completion marker
            if replayed_count > 0:
                replay_end = {
                    'type': 'replay_end',
                    'session_id': session_id,
                    'messages_replayed': replayed_count,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                self.write(f"data: {json.dumps(replay_end)}\n\n")
                await self.flush()
                
                logger.info(f"Replayed {replayed_count} buffered messages for session {session_id}")
            else:
                logger.info(f"No new messages to replay for session {session_id}")
        
        except Exception as e:
            logger.error(f"Error replaying buffered messages for session {session_id}: {e}")
    
    async def _maintain_connection(self, streaming_service: ProgressStreamingService, 
                                 connection_id: str, session_id: str):
        """Maintain streaming connection for ongoing session"""
        try:
            # Send connection ready message
            ready_message = {
                'type': 'connection_ready',
                'connection_id': connection_id,
                'session_id': session_id,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            self.write(f"data: {json.dumps(ready_message)}\n\n")
            await self.flush()
            
            # Keep connection alive - streaming service will handle new messages
            # Connection will be cleaned up automatically when client disconnects
            # or session completes
            
            logger.info(f"Streaming connection {connection_id} ready for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error maintaining connection {connection_id}: {e}")
            await streaming_service.disconnect_client(connection_id)


class SessionProgressHandler(tornado.web.RequestHandler):
    """
    Handler for checking session progress without streaming
    """
    
    async def get(self):
        """Get current session progress"""
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
            
            # Build response
            response = {
                'session_id': session_id,
                'status': session.status.value,
                'question': session.question,
                'created_at': session.created_at.isoformat(),
                'updated_at': session.updated_at.isoformat(),
                'is_active': session.is_active(),
                'is_completed': session.is_completed()
            }
            
            # Add current progress
            if session.current_progress:
                response['current_progress'] = {
                    'step': session.current_progress.current_step.value,
                    'progress': session.current_progress.progress_percent,
                    'description': session.current_progress.description,
                    'timestamp': session.current_progress.timestamp.isoformat()
                }
            
            # Add result if completed
            if session.result:
                response['result'] = {
                    'confidence': session.result.confidence,
                    'answer_length': len(session.result.answer_content),
                    'sources_count': len(session.result.sources),
                    'statistics_count': sum(len(v) if isinstance(v, list) else 1 
                                          for v in session.result.statistics.values()),
                    'total_duration': session.result.total_duration,
                    'queries_generated': session.result.queries_generated,
                    'sources_analyzed': session.result.sources_analyzed,
                    'cache_hits': session.result.cache_hits
                }
            
            # Add error if failed
            if session.error_message:
                response['error_message'] = session.error_message
            
            self.write(json.dumps(response, cls=MongoJSONEncoder))
            
        except Exception as e:
            logger.error(f"Session progress check error: {e}")
            self.set_status(500)
            self.write({'error': f'Progress check failed: {str(e)}'})


class StreamingStatsHandler(tornado.web.RequestHandler):
    """
    Handler for streaming service statistics
    """
    
    async def get(self):
        """Get streaming service statistics"""
        try:
            # Initialize streaming service (temporary for stats)
            streaming_service = ProgressStreamingService()
            stats = streaming_service.get_streaming_stats()
            
            self.write(json.dumps(stats, cls=MongoJSONEncoder))
            
        except Exception as e:
            logger.error(f"Streaming stats error: {e}")
            self.set_status(500)
            self.write({'error': f'Stats retrieval failed: {str(e)}'})