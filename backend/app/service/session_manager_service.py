#!/usr/bin/env python3
"""
Session Manager Service - Manages Deep Think research sessions

Handles session lifecycle management, progress tracking, result storage,
and cleanup operations. Provides background session management that continues
even when frontend disconnects.

Key Features:
- Session creation, tracking, and cleanup
- Progress monitoring with real-time updates
- Result storage and retrieval
- Background task management
- Session statistics and analytics
- Automatic cleanup and maintenance
"""

import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

# Import session models
from ..models.session_models import (
    ProcessingSession, SessionStatus, SessionStep, SessionProgress, 
    SessionResult, SessionEvent, SessionCleanupConfig, SessionCleanupResult
)

# Import MongoDB service
from .mongodb_service import MongoDBService

logger = logging.getLogger(__name__)


# =============================================================================
# Session Manager Service
# =============================================================================

class SessionManagerService:
    """
    Manages Deep Think research sessions with MongoDB persistence
    
    Provides comprehensive session lifecycle management including:
    - Session creation and tracking
    - Progress updates and monitoring
    - Result storage and retrieval
    - Background cleanup operations
    - Performance metrics and analytics
    """
    
    def __init__(self, mongodb_service: MongoDBService, 
                 cleanup_config: Optional[SessionCleanupConfig] = None):
        self.mongodb = mongodb_service
        self.cleanup_config = cleanup_config or SessionCleanupConfig()
        
        # In-memory session tracking for active sessions
        self.active_sessions: Dict[str, ProcessingSession] = {}
        self.session_events: Dict[str, List[SessionEvent]] = defaultdict(list)
        
        # Performance metrics
        self.total_sessions_created = 0
        self.successful_completions = 0
        self.failed_sessions = 0
        self.cleanup_runs = 0
        
        # Background tasks
        self._cleanup_task = None
        self._cleanup_running = False
        
        logger.info("SessionManagerService initialized")
    
    async def initialize(self) -> None:
        """Initialize the session manager and create MongoDB indexes"""
        initialization_errors = []
        
        # Try to create MongoDB indexes
        try:
            await self._create_session_indexes()
            logger.info("Session indexes processed successfully")
        except Exception as e:
            error_msg = f"Index creation failed: {e}"
            logger.error(error_msg)
            initialization_errors.append(error_msg)
        
        # Try to start background cleanup task
        try:
            await self.start_cleanup_task()
            logger.info("Background cleanup task started")
        except Exception as e:
            error_msg = f"Cleanup task failed to start: {e}"
            logger.error(error_msg)
            initialization_errors.append(error_msg)
        
        if initialization_errors:
            logger.warning(f"SessionManagerService initialized with {len(initialization_errors)} warnings")
            for error in initialization_errors:
                logger.warning(f"  - {error}")
        else:
            logger.info("SessionManagerService initialized successfully")
    
    async def _create_session_indexes(self) -> None:
        """Create MongoDB indexes for optimal session queries"""
        indexes_created = []
        
        # Simple approach: use MongoDB's default naming to avoid conflicts
        index_specs = [
            {"keys": "session_id", "options": {"unique": True}},
            {"keys": "user_id", "options": {}},
            {"keys": "chat_id", "options": {}},
            {"keys": "status", "options": {}},
            {"keys": "updated_at", "options": {}},
            {
                "keys": "created_at",
                "options": {"expireAfterSeconds": self.cleanup_config.max_age_days * 24 * 60 * 60},
                "name": "created_at_ttl"  # Only TTL index needs custom name
            },
            {"keys": [("user_id", 1), ("status", 1), ("created_at", -1)], "options": {}}
        ]
        
        # Create each index with graceful error handling
        for i, spec in enumerate(index_specs):
            try:
                # Use custom name only for TTL index, let MongoDB auto-name others
                if "name" in spec and spec["name"]:
                    if spec["options"]:
                        await self.mongodb.db.processing_sessions.create_index(
                            spec["keys"],
                            name=spec["name"],
                            **{k: v for k, v in spec["options"].items() if k != "name"}
                        )
                    else:
                        await self.mongodb.db.processing_sessions.create_index(
                            spec["keys"],
                            name=spec["name"]
                        )
                    indexes_created.append(spec["name"])
                else:
                    # Let MongoDB use default naming
                    if spec["options"]:
                        await self.mongodb.db.processing_sessions.create_index(
                            spec["keys"],
                            **spec["options"]
                        )
                    else:
                        await self.mongodb.db.processing_sessions.create_index(spec["keys"])
                    
                    # Generate expected name for logging
                    if isinstance(spec["keys"], str):
                        expected_name = f"{spec['keys']}_1"
                    else:
                        expected_name = f"compound_{i}"
                    indexes_created.append(expected_name)
                
            except Exception as e:
                error_msg = str(e).lower()
                if "already exists" in error_msg or "equivalent index" in error_msg:
                    index_name = spec.get("name", f"index_{i}")
                    logger.info(f"Index {index_name} already exists, skipping")
                    indexes_created.append(index_name)
                else:
                    logger.warning(f"Failed to create index {i}: {e}")
        
        logger.info(f"Session MongoDB indexes processed. Created/verified: {indexes_created}")
    
    # =============================================================================
    # Session Lifecycle Management
    # =============================================================================
    
    async def create_session(self, question: str, chat_id: str, user_id: str, 
                           timeout_seconds: int = 600) -> ProcessingSession:
        """Create a new processing session"""
        try:
            # Create new session
            session = ProcessingSession.create_new(
                question=question,
                chat_id=chat_id,
                user_id=user_id,
                timeout_seconds=timeout_seconds
            )
            
            # Store in MongoDB
            await self.mongodb.db.processing_sessions.insert_one(session.to_dict())
            
            # Track in memory
            self.active_sessions[session.session_id] = session
            self.total_sessions_created += 1
            
            # Create session event
            event = SessionEvent(
                session_id=session.session_id,
                event_type='created',
                timestamp=datetime.now(timezone.utc),
                data={
                    'question_length': len(question),
                    'timeout_seconds': timeout_seconds
                }
            )
            self.session_events[session.session_id].append(event)
            
            logger.info(f"Created session {session.session_id} for user {user_id}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise
    
    async def start_session(self, session_id: str) -> Optional[ProcessingSession]:
        """Start processing a session"""
        try:
            session = await self.get_session(session_id)
            if not session:
                logger.warning(f"Cannot start session {session_id}: not found")
                return None
            
            if session.status != SessionStatus.PENDING:
                logger.warning(f"Cannot start session {session_id}: status is {session.status}")
                return None
            
            # Update session status
            session.start_processing()
            
            # Update in MongoDB
            await self._update_session_in_db(session)
            
            # Update in memory
            self.active_sessions[session_id] = session
            
            # Create session event
            event = SessionEvent(
                session_id=session_id,
                event_type='started',
                timestamp=datetime.now(timezone.utc)
            )
            self.session_events[session_id].append(event)
            
            logger.info(f"Started session {session_id}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to start session {session_id}: {e}")
            return None
    
    async def update_session_progress(self, session_id: str, step: SessionStep, 
                                    progress_percent: int, description: str,
                                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update session progress"""
        try:
            session = await self.get_session(session_id)
            if not session:
                logger.warning(f"Cannot update progress for session {session_id}: not found")
                return False
            
            # Create progress object
            progress = SessionProgress(
                current_step=step,
                progress_percent=progress_percent,
                description=description,
                metadata=metadata or {}
            )
            
            # Update session
            session.update_progress(progress)
            
            # Update in MongoDB
            await self._update_session_in_db(session)
            
            # Update in memory
            self.active_sessions[session_id] = session
            
            # Create session event
            event = SessionEvent(
                session_id=session_id,
                event_type='progress',
                timestamp=datetime.now(timezone.utc),
                data={
                    'step': step.value,
                    'progress': progress_percent,
                    'description': description
                }
            )
            self.session_events[session_id].append(event)
            
            logger.debug(f"Updated progress for session {session_id}: {progress_percent}% ({step.value})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update session progress {session_id}: {e}")
            return False
    
    async def complete_session(self, session_id: str, result: SessionResult) -> bool:
        """Complete a session with results"""
        try:
            session = await self.get_session(session_id)
            if not session:
                logger.warning(f"Cannot complete session {session_id}: not found")
                return False
            
            # Update session with result
            session.complete_session(result)
            
            # Update in MongoDB
            await self._update_session_in_db(session)
            
            # Update metrics
            self.successful_completions += 1
            
            # Remove from active sessions
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            # Create session event
            event = SessionEvent(
                session_id=session_id,
                event_type='completed',
                timestamp=datetime.now(timezone.utc),
                data={
                    'confidence': result.confidence,
                    'sources_count': len(result.sources),
                    'duration': result.total_duration
                }
            )
            self.session_events[session_id].append(event)
            
            logger.info(f"Completed session {session_id} with confidence {result.confidence:.1%}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to complete session {session_id}: {e}")
            return False
    
    async def fail_session(self, session_id: str, error_message: str) -> bool:
        """Mark a session as failed"""
        try:
            session = await self.get_session(session_id)
            if not session:
                logger.warning(f"Cannot fail session {session_id}: not found")
                return False
            
            # Update session with error
            session.fail_session(error_message)
            
            # Update in MongoDB
            await self._update_session_in_db(session)
            
            # Update metrics
            self.failed_sessions += 1
            
            # Remove from active sessions
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            # Create session event
            event = SessionEvent(
                session_id=session_id,
                event_type='failed',
                timestamp=datetime.now(timezone.utc),
                data={'error_message': error_message}
            )
            self.session_events[session_id].append(event)
            
            logger.warning(f"Failed session {session_id}: {error_message}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to mark session {session_id} as failed: {e}")
            return False
    
    async def timeout_session(self, session_id: str) -> bool:
        """Mark a session as timed out"""
        try:
            session = await self.get_session(session_id)
            if not session:
                logger.warning(f"Cannot timeout session {session_id}: not found")
                return False
            
            # Update session with timeout
            session.timeout_session()
            
            # Update in MongoDB
            await self._update_session_in_db(session)
            
            # Update metrics
            self.failed_sessions += 1
            
            # Remove from active sessions
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            # Create session event
            event = SessionEvent(
                session_id=session_id,
                event_type='timeout',
                timestamp=datetime.now(timezone.utc),
                data={'timeout_seconds': session.timeout_seconds}
            )
            self.session_events[session_id].append(event)
            
            logger.warning(f"Session {session_id} timed out after {session.timeout_seconds}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to timeout session {session_id}: {e}")
            return False
    
    async def cancel_session(self, session_id: str) -> bool:
        """Cancel a session"""
        try:
            session = await self.get_session(session_id)
            if not session:
                logger.warning(f"Cannot cancel session {session_id}: not found")
                return False
            
            # Update session as cancelled
            session.cancel_session()
            
            # Update in MongoDB
            await self._update_session_in_db(session)
            
            # Remove from active sessions
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            # Create session event
            event = SessionEvent(
                session_id=session_id,
                event_type='cancelled',
                timestamp=datetime.now(timezone.utc)
            )
            self.session_events[session_id].append(event)
            
            logger.info(f"Cancelled session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel session {session_id}: {e}")
            return False
    
    # =============================================================================
    # Session Query and Retrieval
    # =============================================================================
    
    async def get_session(self, session_id: str) -> Optional[ProcessingSession]:
        """Get session by ID (checks memory first, then MongoDB)"""
        try:
            # Check in-memory first
            if session_id in self.active_sessions:
                return self.active_sessions[session_id]
            
            # Query MongoDB
            doc = await self.mongodb.db.processing_sessions.find_one({"session_id": session_id})
            if doc:
                return ProcessingSession.from_dict(doc)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None
    
    async def get_user_sessions(self, user_id: str, limit: int = 20, 
                              status_filter: Optional[SessionStatus] = None) -> List[ProcessingSession]:
        """Get sessions for a user"""
        try:
            query = {"user_id": user_id}
            if status_filter:
                query["status"] = status_filter.value
            
            cursor = self.mongodb.db.processing_sessions.find(query).sort("created_at", -1).limit(limit)
            sessions = []
            
            async for doc in cursor:
                sessions.append(ProcessingSession.from_dict(doc))
            
            return sessions
            
        except Exception as e:
            logger.error(f"Failed to get sessions for user {user_id}: {e}")
            return []
    
    async def get_active_sessions(self) -> List[ProcessingSession]:
        """Get all currently active sessions"""
        try:
            # Return in-memory active sessions
            active_sessions = []
            for session in self.active_sessions.values():
                if session.is_active():
                    active_sessions.append(session)
                elif session.is_expired():
                    # Handle expired sessions
                    await self.timeout_session(session.session_id)
            
            return active_sessions
            
        except Exception as e:
            logger.error(f"Failed to get active sessions: {e}")
            return []
    
    async def get_session_statistics(self) -> Dict[str, Any]:
        """Get comprehensive session statistics"""
        try:
            # Query MongoDB for session counts
            pipeline = [
                {
                    "$group": {
                        "_id": "$status",
                        "count": {"$sum": 1},
                        "avg_duration": {
                            "$avg": {
                                "$divide": [
                                    {"$subtract": ["$completed_at", "$started_at"]},
                                    1000  # Convert to seconds
                                ]
                            }
                        }
                    }
                }
            ]
            
            status_stats = {}
            async for doc in self.mongodb.db.processing_sessions.aggregate(pipeline):
                status_stats[doc["_id"]] = {
                    "count": doc["count"],
                    "avg_duration": doc.get("avg_duration", 0.0) or 0.0
                }
            
            # Calculate overall metrics
            total_sessions = sum(stat["count"] for stat in status_stats.values())
            completed_sessions = status_stats.get("completed", {}).get("count", 0)
            failed_sessions = sum(
                status_stats.get(status, {}).get("count", 0) 
                for status in ["failed", "timeout", "cancelled"]
            )
            
            success_rate = (completed_sessions / max(total_sessions, 1)) * 100
            
            return {
                "total_sessions": total_sessions,
                "active_sessions": len(self.active_sessions),
                "completed_sessions": completed_sessions,
                "failed_sessions": failed_sessions,
                "success_rate": success_rate,
                "average_duration": status_stats.get("completed", {}).get("avg_duration", 0.0),
                "status_breakdown": status_stats,
                "cleanup_runs": self.cleanup_runs
            }
            
        except Exception as e:
            logger.error(f"Failed to get session statistics: {e}")
            return {
                "total_sessions": 0,
                "active_sessions": len(self.active_sessions),
                "completed_sessions": 0,
                "failed_sessions": 0,
                "success_rate": 0.0,
                "average_duration": 0.0,
                "status_breakdown": {},
                "cleanup_runs": self.cleanup_runs
            }
    
    # =============================================================================
    # Session Cleanup and Maintenance
    # =============================================================================
    
    async def start_cleanup_task(self) -> None:
        """Start background cleanup task"""
        if not self._cleanup_running:
            self._cleanup_running = True
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Session cleanup task started")
    
    async def stop_cleanup_task(self) -> None:
        """Stop background cleanup task"""
        if self._cleanup_running:
            self._cleanup_running = False
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            logger.info("Session cleanup task stopped")
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop"""
        while self._cleanup_running:
            try:
                await asyncio.sleep(self.cleanup_config.cleanup_interval_hours * 3600)
                if self._cleanup_running:  # Check again after sleep
                    await self.cleanup_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def cleanup_sessions(self) -> SessionCleanupResult:
        """Clean up old and expired sessions"""
        try:
            start_time = time.time()
            self.cleanup_runs += 1
            
            # Calculate cutoff date
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.cleanup_config.max_age_days)
            
            # Find sessions to clean up
            cleanup_query = {
                "$or": [
                    {"created_at": {"$lt": cutoff_date}},
                    {"status": {"$in": ["completed", "failed", "timeout", "cancelled"]}}
                ]
            }
            
            # Count sessions to be cleaned
            sessions_to_clean = await self.mongodb.db.processing_sessions.count_documents(cleanup_query)
            
            if sessions_to_clean == 0:
                return SessionCleanupResult(
                    sessions_cleaned=0,
                    space_freed_mb=0.0,
                    cleanup_duration=time.time() - start_time
                )
            
            # Calculate approximate space
            sample_docs = await self.mongodb.db.processing_sessions.find(
                cleanup_query, {"session_id": 1, "result": 1, "progress_history": 1}
            ).limit(100).to_list(length=100)
            
            avg_size = 0
            if sample_docs:
                total_size = sum(len(str(doc).encode('utf-8')) for doc in sample_docs)
                avg_size = total_size / len(sample_docs)
            
            estimated_space_mb = (sessions_to_clean * avg_size) / (1024 * 1024)
            
            # Remove old sessions
            delete_result = await self.mongodb.db.processing_sessions.delete_many(cleanup_query)
            
            # Clean up in-memory sessions
            cleaned_memory = 0
            for session_id in list(self.active_sessions.keys()):
                session = self.active_sessions[session_id]
                if session.is_expired() or session.is_completed():
                    del self.active_sessions[session_id]
                    cleaned_memory += 1
            
            # Clean up session events
            for session_id in list(self.session_events.keys()):
                if session_id not in self.active_sessions:
                    del self.session_events[session_id]
            
            # Get oldest remaining session
            oldest_doc = await self.mongodb.db.processing_sessions.find_one(
                {}, sort=[("created_at", 1)]
            )
            oldest_remaining = oldest_doc.get("created_at") if oldest_doc else None
            
            cleanup_duration = time.time() - start_time
            
            result = SessionCleanupResult(
                sessions_cleaned=delete_result.deleted_count + cleaned_memory,
                space_freed_mb=estimated_space_mb,
                oldest_remaining=oldest_remaining,
                cleanup_duration=cleanup_duration
            )
            
            logger.info(f"Session cleanup completed: {result.sessions_cleaned} sessions cleaned, "
                       f"{result.space_freed_mb:.2f} MB freed in {cleanup_duration:.1f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Session cleanup failed: {e}")
            return SessionCleanupResult(
                sessions_cleaned=0,
                space_freed_mb=0.0,
                cleanup_duration=time.time() - start_time
            )
    
    # =============================================================================
    # Helper Methods
    # =============================================================================
    
    async def _update_session_in_db(self, session: ProcessingSession) -> None:
        """Update session in MongoDB"""
        try:
            await self.mongodb.db.processing_sessions.replace_one(
                {"session_id": session.session_id},
                session.to_dict(),
                upsert=True
            )
        except Exception as e:
            logger.error(f"Failed to update session {session.session_id} in MongoDB: {e}")
            raise
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get session manager performance metrics"""
        total_requests = self.total_sessions_created
        success_rate = (self.successful_completions / max(total_requests, 1)) * 100
        
        return {
            "total_sessions_created": self.total_sessions_created,
            "successful_completions": self.successful_completions,
            "failed_sessions": self.failed_sessions,
            "success_rate": round(success_rate, 1),
            "active_sessions_count": len(self.active_sessions),
            "cleanup_runs": self.cleanup_runs,
            "cleanup_running": self._cleanup_running
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on session manager"""
        try:
            # Check MongoDB connection
            await self.mongodb.db.admin.command('ping')
            mongodb_healthy = True
        except:
            mongodb_healthy = False
        
        # Check active sessions
        active_count = len(self.active_sessions)
        
        # Check for expired sessions that haven't been cleaned up
        expired_sessions = 0
        for session in self.active_sessions.values():
            if session.is_expired():
                expired_sessions += 1
        
        return {
            "mongodb_connection": mongodb_healthy,
            "active_sessions": active_count,
            "expired_sessions": expired_sessions,
            "cleanup_task_running": self._cleanup_running,
            "session_events_cached": len(self.session_events),
            "healthy": mongodb_healthy and expired_sessions < 5  # Allow up to 5 expired sessions
        }
