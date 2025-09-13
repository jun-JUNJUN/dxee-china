#!/usr/bin/env python3
"""
Progress Streaming Service - Enhanced SSE streaming for Deep Think progress

Provides improved Server-Sent Events streaming with connection management,
heartbeat monitoring, and graceful disconnection handling.

Key Features:
- Enhanced SSE streaming with proper headers
- Connection heartbeat and keep-alive
- Graceful disconnection detection
- Progress buffering for reconnection
- Streaming rate limiting and throttling
- Connection status monitoring
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass, field
from collections import deque
import tornado.web

from .streaming_error_handler import StreamingErrorHandler, StreamingError, ErrorSeverity

logger = logging.getLogger(__name__)

# =============================================================================
# Streaming Models
# =============================================================================

@dataclass
class StreamingConnection:
    """Represents a client streaming connection"""
    connection_id: str
    session_id: str
    user_id: str
    chat_id: str
    connected_at: datetime
    last_activity: datetime
    handler: tornado.web.RequestHandler
    is_active: bool = True
    messages_sent: int = 0
    heartbeat_count: int = 0
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now(timezone.utc)
    
    def is_stale(self, timeout_seconds: int = 300) -> bool:
        """Check if connection is stale (no activity for timeout period)"""
        elapsed = datetime.now(timezone.utc) - self.last_activity
        return elapsed.total_seconds() > timeout_seconds

@dataclass
class ProgressMessage:
    """Progress message for streaming"""
    session_id: str
    message_type: str  # 'progress', 'completion', 'error', 'heartbeat'
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    retry_count: int = 0
    
    def to_sse_format(self) -> str:
        """Convert to Server-Sent Events format"""
        message_data = {
            'type': self.message_type,
            'timestamp': self.timestamp.isoformat(),
            **self.data
        }
        return f"data: {json.dumps(message_data)}\n\n"

# =============================================================================
# Enhanced Progress Streaming Service
# =============================================================================

class ProgressStreamingService:
    """
    Enhanced streaming service for Deep Think progress updates
    
    Manages SSE connections with improved error handling, connection monitoring,
    and graceful disconnection handling.
    """
    
    def __init__(self, heartbeat_interval: int = 30, connection_timeout: int = 300):
        self.heartbeat_interval = heartbeat_interval  # seconds
        self.connection_timeout = connection_timeout  # seconds
        
        # Error handler for simplified error management
        self.error_handler = StreamingErrorHandler()
        
        # Active connections tracking
        self.active_connections: Dict[str, StreamingConnection] = {}
        self.session_connections: Dict[str, List[str]] = {}  # session_id -> connection_ids
        
        # Message buffering for reconnection
        self.message_buffer: Dict[str, deque] = {}  # session_id -> messages
        self.buffer_size = 50  # Keep last 50 messages per session
        
        # Background tasks
        self._heartbeat_task = None
        self._cleanup_task = None
        self._running = False
        
        # Metrics
        self.total_connections = 0
        self.active_sessions = 0
        self.messages_sent = 0
        self.disconnections = 0
        
        logger.info("ProgressStreamingService initialized with error handling")
    
    async def start(self):
        """Start background tasks"""
        if not self._running:
            self._running = True
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Progress streaming service started")
    
    async def stop(self):
        """Stop background tasks"""
        if self._running:
            self._running = False
            
            # Cancel tasks
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass
            
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Close all connections
            for connection in list(self.active_connections.values()):
                await self._disconnect_client(connection.connection_id)
            
            logger.info("Progress streaming service stopped")
    
    # =============================================================================
    # Connection Management
    # =============================================================================
    
    async def create_streaming_connection(self, session_id: str, user_id: str, chat_id: str,
                                        handler: tornado.web.RequestHandler) -> str:
        """Create a new streaming connection"""
        connection_id = f"{session_id}_{int(time.time() * 1000)}"
        
        # Set SSE headers
        handler.set_header('Content-Type', 'text/event-stream')
        handler.set_header('Cache-Control', 'no-cache')
        handler.set_header('Connection', 'keep-alive')
        handler.set_header('Access-Control-Allow-Origin', '*')
        handler.set_header('Access-Control-Allow-Headers', 'Cache-Control')
        
        # Create connection object
        connection = StreamingConnection(
            connection_id=connection_id,
            session_id=session_id,
            user_id=user_id,
            chat_id=chat_id,
            connected_at=datetime.now(timezone.utc),
            last_activity=datetime.now(timezone.utc),
            handler=handler
        )
        
        # Track connection
        self.active_connections[connection_id] = connection
        
        if session_id not in self.session_connections:
            self.session_connections[session_id] = []
        self.session_connections[session_id].append(connection_id)
        
        # Initialize message buffer for session
        if session_id not in self.message_buffer:
            self.message_buffer[session_id] = deque(maxlen=self.buffer_size)
        
        self.total_connections += 1
        self.active_sessions = len(self.session_connections)
        
        logger.info(f"Created streaming connection {connection_id} for session {session_id}")
        
        # Send initial connection message
        await self._send_to_connection(connection_id, ProgressMessage(
            session_id=session_id,
            message_type='connected',
            data={
                'connection_id': connection_id,
                'message': 'Streaming connection established',
                'server_time': datetime.now(timezone.utc).isoformat()
            }
        ))
        
        return connection_id
    
    async def disconnect_client(self, connection_id: str):
        """Disconnect a client"""
        await self._disconnect_client(connection_id)
    
    async def _disconnect_client(self, connection_id: str):
        """Internal method to disconnect client"""
        if connection_id in self.active_connections:
            connection = self.active_connections[connection_id]
            connection.is_active = False
            
            # Remove from session tracking
            if connection.session_id in self.session_connections:
                if connection_id in self.session_connections[connection.session_id]:
                    self.session_connections[connection.session_id].remove(connection_id)
                
                # Clean up empty session
                if not self.session_connections[connection.session_id]:
                    del self.session_connections[connection.session_id]
            
            # Remove connection
            del self.active_connections[connection_id]
            self.disconnections += 1
            self.active_sessions = len(self.session_connections)
            
            logger.info(f"Disconnected client {connection_id} for session {connection.session_id}")
    
    # =============================================================================
    # Message Streaming
    # =============================================================================
    
    async def broadcast_progress(self, session_id: str, progress_data: Dict[str, Any]):
        """Broadcast progress update to all connections for a session"""
        message = ProgressMessage(
            session_id=session_id,
            message_type='progress',
            data=progress_data
        )
        
        await self._broadcast_to_session(session_id, message)
    
    async def broadcast_completion(self, session_id: str, result_data: Dict[str, Any]):
        """Broadcast completion message to all connections for a session"""
        message = ProgressMessage(
            session_id=session_id,
            message_type='completion',
            data=result_data
        )
        
        await self._broadcast_to_session(session_id, message)
    
    async def broadcast_error(self, session_id: str, error_message: str):
        """Broadcast error message to all connections for a session"""
        message = ProgressMessage(
            session_id=session_id,
            message_type='error',
            data={'error': error_message}
        )
        
        await self._broadcast_to_session(session_id, message)
    
    async def send_heartbeat(self, connection_id: str):
        """Send heartbeat to specific connection"""
        if connection_id in self.active_connections:
            connection = self.active_connections[connection_id]
            connection.heartbeat_count += 1
            
            message = ProgressMessage(
                session_id=connection.session_id,
                message_type='heartbeat',
                data={
                    'heartbeat_count': connection.heartbeat_count,
                    'server_time': datetime.now(timezone.utc).isoformat()
                }
            )
            
            await self._send_to_connection(connection_id, message)
    
    async def _broadcast_to_session(self, session_id: str, message: ProgressMessage):
        """Broadcast message to all connections for a session"""
        # Add to message buffer
        if session_id in self.message_buffer:
            self.message_buffer[session_id].append(message)
        
        # Send to active connections
        if session_id in self.session_connections:
            connection_ids = self.session_connections[session_id].copy()
            
            for connection_id in connection_ids:
                await self._send_to_connection(connection_id, message)
    
    async def _send_to_connection(self, connection_id: str, message: ProgressMessage):
        """Send message to specific connection"""
        if connection_id not in self.active_connections:
            return
        
        connection = self.active_connections[connection_id]
        if not connection.is_active:
            return
        
        try:
            # Write SSE message
            sse_data = message.to_sse_format()
            connection.handler.write(sse_data)
            await connection.handler.flush()
            
            # Update connection activity
            connection.update_activity()
            connection.messages_sent += 1
            self.messages_sent += 1
            
            logger.debug(f"Sent message to connection {connection_id}: {message.message_type}")
            
        except Exception as e:
            # Handle error using error handler (no retry logic)
            error = self.error_handler.create_connection_error(
                message=f"Failed to send message to connection: {str(e)}",
                session_id=connection.session_id,
                connection_id=connection_id,
                context={'message_type': message.message_type}
            )
            
            error_response = self.error_handler.handle_error(error)
            
            # Mark connection as inactive based on error handler response
            if error_response['should_disconnect']:
                connection.is_active = False
                logger.info(f"Connection {connection_id} marked for cleanup due to error")
            else:
                logger.warning(f"Connection error handled, continuing: {connection_id}")
    
    # =============================================================================
    # Background Tasks
    # =============================================================================
    
    async def _heartbeat_loop(self):
        """Background task to send periodic heartbeats"""
        while self._running:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                if self._running:
                    # Send heartbeat to all active connections
                    for connection_id in list(self.active_connections.keys()):
                        await self.send_heartbeat(connection_id)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
    
    async def _cleanup_loop(self):
        """Background task to clean up stale connections"""
        while self._running:
            try:
                await asyncio.sleep(60)  # Run cleanup every minute
                
                if self._running:
                    await self._cleanup_stale_connections()
                    await self._cleanup_old_message_buffers()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _cleanup_stale_connections(self):
        """Clean up stale/inactive connections"""
        stale_connections = []
        
        for connection_id, connection in self.active_connections.items():
            if not connection.is_active or connection.is_stale(self.connection_timeout):
                stale_connections.append(connection_id)
        
        for connection_id in stale_connections:
            await self._disconnect_client(connection_id)
        
        if stale_connections:
            logger.info(f"Cleaned up {len(stale_connections)} stale connections")
    
    async def _cleanup_old_message_buffers(self):
        """Clean up message buffers for inactive sessions"""
        active_session_ids = set(self.session_connections.keys())
        all_buffered_sessions = set(self.message_buffer.keys())
        
        # Clean up buffers for sessions with no active connections
        inactive_sessions = all_buffered_sessions - active_session_ids
        for session_id in inactive_sessions:
            del self.message_buffer[session_id]
        
        if inactive_sessions:
            logger.info(f"Cleaned up message buffers for {len(inactive_sessions)} inactive sessions")
    
    # =============================================================================
    # Session Management Integration
    # =============================================================================
    
    async def handle_session_progress(self, session_id: str, step: str, progress: int, 
                                    description: str, metadata: Optional[Dict[str, Any]] = None):
        """Handle session progress update from session manager"""
        progress_data = {
            'session_id': session_id,
            'step': step,
            'progress': progress,
            'description': description,
            'metadata': metadata or {}
        }
        
        await self.broadcast_progress(session_id, progress_data)
    
    async def handle_session_completion(self, session_id: str, result: Dict[str, Any]):
        """Handle session completion from session manager"""
        completion_data = {
            'session_id': session_id,
            'completed': True,
            'result': result
        }
        
        await self.broadcast_completion(session_id, completion_data)
        
        # Clean up connections for completed session after a delay
        await asyncio.sleep(5)  # Allow clients to receive completion message
        await self._cleanup_session_connections(session_id)
    
    async def handle_session_error(self, session_id: str, error_message: str):
        """Handle session error from session manager"""
        try:
            # Create structured error
            error = self.error_handler.create_session_error(
                message=error_message,
                session_id=session_id,
                severity=ErrorSeverity.HIGH,
                context={'source': 'session_manager'}
            )
            
            # Process error (no retry logic)
            error_response = self.error_handler.handle_error(error)
            
            # Broadcast error to clients
            await self.broadcast_error(session_id, error.user_message)
            
            # Clean up connections based on error severity
            if error_response['should_disconnect']:
                await asyncio.sleep(5)  # Allow clients to receive error message
                await self._cleanup_session_connections(session_id)
                
        except Exception as e:
            logger.error(f"Error handling session error for {session_id}: {e}")
            # Fallback to simple cleanup
            await self._cleanup_session_connections(session_id)
    
    async def _cleanup_session_connections(self, session_id: str):
        """Clean up all connections for a session"""
        if session_id in self.session_connections:
            connection_ids = self.session_connections[session_id].copy()
            for connection_id in connection_ids:
                await self._disconnect_client(connection_id)
        
        # Clean up message buffer
        if session_id in self.message_buffer:
            del self.message_buffer[session_id]
    
    # =============================================================================
    # Metrics and Status
    # =============================================================================
    
    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get streaming service statistics"""
        return {
            'active_connections': len(self.active_connections),
            'active_sessions': len(self.session_connections),
            'total_connections_created': self.total_connections,
            'total_messages_sent': self.messages_sent,
            'total_disconnections': self.disconnections,
            'message_buffers': len(self.message_buffer),
            'service_running': self._running,
            'heartbeat_interval': self.heartbeat_interval,
            'connection_timeout': self.connection_timeout
        }
    
    def get_session_connections(self, session_id: str) -> List[Dict[str, Any]]:
        """Get connection info for a specific session"""
        connections = []
        
        if session_id in self.session_connections:
            for connection_id in self.session_connections[session_id]:
                if connection_id in self.active_connections:
                    conn = self.active_connections[connection_id]
                    connections.append({
                        'connection_id': connection_id,
                        'connected_at': conn.connected_at.isoformat(),
                        'last_activity': conn.last_activity.isoformat(),
                        'messages_sent': conn.messages_sent,
                        'heartbeat_count': conn.heartbeat_count,
                        'is_active': conn.is_active
                    })
        
        return connections
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on streaming service"""
        return {
            'service_running': self._running,
            'active_connections': len(self.active_connections),
            'active_sessions': len(self.session_connections),
            'background_tasks_running': (
                self._heartbeat_task is not None and not self._heartbeat_task.done(),
                self._cleanup_task is not None and not self._cleanup_task.done()
            ),
            'healthy': self._running and len(self.active_connections) >= 0
        }