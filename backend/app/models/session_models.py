#!/usr/bin/env python3
"""
Session Models - Data models for Deep Think research sessions

Manages session-based processing that continues even when frontend disconnects.
Provides data structures for tracking research progress, storing intermediate results,
and managing session lifecycle.

Key Features:
- Session lifecycle management (pending, active, completed, failed)
- Progress tracking with step-by-step updates
- Result storage with confidence metrics and source attribution
- Session metadata for debugging and analytics
- MongoDB-compatible serialization
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
from enum import Enum
from pydantic import BaseModel, Field
import uuid


# =============================================================================
# Session Status and Step Enums
# =============================================================================

class SessionStatus(Enum):
    """Session processing status"""
    PENDING = "pending"          # Session created but not started
    ACTIVE = "active"            # Session currently processing
    COMPLETED = "completed"      # Session finished successfully
    FAILED = "failed"            # Session failed with error
    TIMEOUT = "timeout"          # Session exceeded time limit
    CANCELLED = "cancelled"      # Session cancelled by user

class SessionStep(Enum):
    """Research processing steps for progress tracking"""
    INITIALIZING = "initializing"
    ANALYZING_QUESTION = "analyzing_question"
    GENERATING_QUERIES = "generating_queries"
    SEARCHING_WEB = "searching_web"
    EXTRACTING_CONTENT = "extracting_content"
    EVALUATING_RELEVANCE = "evaluating_relevance"
    SYNTHESIZING_ANSWER = "synthesizing_answer"
    COMPLETING = "completing"

# =============================================================================
# Core Session Models
# =============================================================================

@dataclass
class SessionProgress:
    """Session progress tracking"""
    current_step: SessionStep
    progress_percent: int  # 0-100
    description: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage"""
        return {
            'current_step': self.current_step.value,
            'progress_percent': self.progress_percent,
            'description': self.description,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionProgress':
        """Create from dictionary (MongoDB document)"""
        return cls(
            current_step=SessionStep(data['current_step']),
            progress_percent=data['progress_percent'],
            description=data['description'],
            timestamp=data['timestamp'],
            metadata=data.get('metadata', {})
        )

@dataclass
class SessionResult:
    """Research session result"""
    answer_content: str
    confidence: float
    sources: List[str]
    statistics: Dict[str, Any] = field(default_factory=dict)
    generation_time: float = 0.0
    total_duration: float = 0.0
    queries_generated: int = 0
    sources_analyzed: int = 0
    cache_hits: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage"""
        return {
            'answer_content': self.answer_content,
            'confidence': self.confidence,
            'sources': self.sources,
            'statistics': self.statistics,
            'generation_time': self.generation_time,
            'total_duration': self.total_duration,
            'queries_generated': self.queries_generated,
            'sources_analyzed': self.sources_analyzed,
            'cache_hits': self.cache_hits,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionResult':
        """Create from dictionary (MongoDB document)"""
        return cls(
            answer_content=data['answer_content'],
            confidence=data['confidence'],
            sources=data['sources'],
            statistics=data.get('statistics', {}),
            generation_time=data.get('generation_time', 0.0),
            total_duration=data.get('total_duration', 0.0),
            queries_generated=data.get('queries_generated', 0),
            sources_analyzed=data.get('sources_analyzed', 0),
            cache_hits=data.get('cache_hits', 0),
            metadata=data.get('metadata', {})
        )

@dataclass
class ProcessingSession:
    """Main processing session model for Deep Think research"""
    session_id: str
    chat_id: str
    user_id: str
    question: str
    status: SessionStatus
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    current_progress: Optional[SessionProgress] = None
    result: Optional[SessionResult] = None
    error_message: Optional[str] = None
    timeout_seconds: int = 600  # 10 minutes default
    progress_history: List[SessionProgress] = field(default_factory=list)
    session_metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create_new(cls, question: str, chat_id: str, user_id: str, 
                   timeout_seconds: int = 600) -> 'ProcessingSession':
        """Create a new processing session"""
        now = datetime.now(timezone.utc)
        session_id = str(uuid.uuid4())
        
        return cls(
            session_id=session_id,
            chat_id=chat_id,
            user_id=user_id,
            question=question,
            status=SessionStatus.PENDING,
            created_at=now,
            updated_at=now,
            timeout_seconds=timeout_seconds,
            session_metadata={
                'question_length': len(question),
                'created_timestamp': now.isoformat()
            }
        )
    
    def start_processing(self) -> None:
        """Mark session as started"""
        self.status = SessionStatus.ACTIVE
        self.started_at = datetime.now(timezone.utc)
        self.updated_at = self.started_at
    
    def update_progress(self, progress: SessionProgress) -> None:
        """Update session progress"""
        self.current_progress = progress
        self.progress_history.append(progress)
        self.updated_at = datetime.now(timezone.utc)
    
    def complete_session(self, result: SessionResult) -> None:
        """Mark session as completed with result"""
        self.status = SessionStatus.COMPLETED
        self.result = result
        self.completed_at = datetime.now(timezone.utc)
        self.updated_at = self.completed_at
    
    def fail_session(self, error_message: str) -> None:
        """Mark session as failed with error"""
        self.status = SessionStatus.FAILED
        self.error_message = error_message
        self.completed_at = datetime.now(timezone.utc)
        self.updated_at = self.completed_at
    
    def timeout_session(self) -> None:
        """Mark session as timed out"""
        self.status = SessionStatus.TIMEOUT
        self.completed_at = datetime.now(timezone.utc)
        self.updated_at = self.completed_at
        self.error_message = f"Session timed out after {self.timeout_seconds} seconds"
    
    def cancel_session(self) -> None:
        """Mark session as cancelled"""
        self.status = SessionStatus.CANCELLED
        self.completed_at = datetime.now(timezone.utc)
        self.updated_at = self.completed_at
    
    def is_expired(self) -> bool:
        """Check if session has exceeded timeout"""
        if not self.started_at:
            return False
        
        elapsed = datetime.now(timezone.utc) - self.started_at
        return elapsed.total_seconds() > self.timeout_seconds
    
    def is_active(self) -> bool:
        """Check if session is currently active"""
        return self.status == SessionStatus.ACTIVE and not self.is_expired()
    
    def is_completed(self) -> bool:
        """Check if session is completed (success or failure)"""
        return self.status in [SessionStatus.COMPLETED, SessionStatus.FAILED, 
                              SessionStatus.TIMEOUT, SessionStatus.CANCELLED]
    
    def get_duration(self) -> Optional[float]:
        """Get session duration in seconds"""
        if not self.started_at:
            return None
        
        end_time = self.completed_at or datetime.now(timezone.utc)
        return (end_time - self.started_at).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage"""
        return {
            'session_id': self.session_id,
            'chat_id': self.chat_id,
            'user_id': self.user_id,
            'question': self.question,
            'status': self.status.value,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'current_progress': self.current_progress.to_dict() if self.current_progress else None,
            'result': self.result.to_dict() if self.result else None,
            'error_message': self.error_message,
            'timeout_seconds': self.timeout_seconds,
            'progress_history': [p.to_dict() for p in self.progress_history],
            'session_metadata': self.session_metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingSession':
        """Create from dictionary (MongoDB document)"""
        # Reconstruct progress objects
        current_progress = None
        if data.get('current_progress'):
            current_progress = SessionProgress.from_dict(data['current_progress'])
        
        progress_history = []
        for p_data in data.get('progress_history', []):
            progress_history.append(SessionProgress.from_dict(p_data))
        
        # Reconstruct result object
        result = None
        if data.get('result'):
            result = SessionResult.from_dict(data['result'])
        
        return cls(
            session_id=data['session_id'],
            chat_id=data['chat_id'],
            user_id=data['user_id'],
            question=data['question'],
            status=SessionStatus(data['status']),
            created_at=data['created_at'],
            updated_at=data['updated_at'],
            started_at=data.get('started_at'),
            completed_at=data.get('completed_at'),
            current_progress=current_progress,
            result=result,
            error_message=data.get('error_message'),
            timeout_seconds=data.get('timeout_seconds', 600),
            progress_history=progress_history,
            session_metadata=data.get('session_metadata', {})
        )

# =============================================================================
# Pydantic Models for API Requests/Responses
# =============================================================================

class SessionCreateRequest(BaseModel):
    """Request to create a new processing session"""
    question: str = Field(..., description="Research question", min_length=1, max_length=1000)
    chat_id: str = Field(..., description="Chat ID for this session")
    timeout_seconds: Optional[int] = Field(600, description="Session timeout in seconds", ge=60, le=3600)

class SessionProgressResponse(BaseModel):
    """Response for session progress updates"""
    session_id: str = Field(..., description="Session identifier")
    status: str = Field(..., description="Current session status")
    current_step: Optional[str] = Field(None, description="Current processing step")
    progress_percent: Optional[int] = Field(None, description="Progress percentage 0-100")
    description: Optional[str] = Field(None, description="Progress description")
    updated_at: datetime = Field(..., description="Last update timestamp")

class SessionResultResponse(BaseModel):
    """Response for completed session result"""
    session_id: str = Field(..., description="Session identifier")
    question: str = Field(..., description="Original research question")
    answer_content: str = Field(..., description="Research answer")
    confidence: float = Field(..., description="Answer confidence 0-1")
    sources: List[str] = Field(..., description="Source URLs")
    statistics: Dict[str, Any] = Field(default_factory=dict, description="Extracted statistics")
    total_duration: float = Field(..., description="Total processing time in seconds")
    queries_generated: int = Field(..., description="Number of search queries generated")
    sources_analyzed: int = Field(..., description="Number of sources analyzed")
    cache_hits: int = Field(..., description="Number of cache hits")
    completed_at: datetime = Field(..., description="Completion timestamp")

class SessionListResponse(BaseModel):
    """Response for session list queries"""
    sessions: List[Dict[str, Any]] = Field(..., description="List of sessions")
    total_count: int = Field(..., description="Total number of sessions")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")

class SessionStatsResponse(BaseModel):
    """Response for session statistics"""
    total_sessions: int = Field(..., description="Total number of sessions")
    active_sessions: int = Field(..., description="Currently active sessions")
    completed_sessions: int = Field(..., description="Successfully completed sessions")
    failed_sessions: int = Field(..., description="Failed sessions")
    average_duration: float = Field(..., description="Average session duration in seconds")
    success_rate: float = Field(..., description="Success rate percentage")
    cache_hit_rate: float = Field(..., description="Cache hit rate percentage")

# =============================================================================
# Session Event Models
# =============================================================================

@dataclass
class SessionEvent:
    """Event for session state changes"""
    session_id: str
    event_type: str  # 'created', 'started', 'progress', 'completed', 'failed', 'timeout'
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/logging"""
        return {
            'session_id': self.session_id,
            'event_type': self.event_type,
            'timestamp': self.timestamp,
            'data': self.data
        }

# =============================================================================
# Session Cleanup Models
# =============================================================================

@dataclass
class SessionCleanupConfig:
    """Configuration for session cleanup"""
    max_age_days: int = 7  # Keep sessions for 7 days
    max_completed_sessions: int = 1000  # Keep max 1000 completed sessions per user
    cleanup_interval_hours: int = 6  # Run cleanup every 6 hours
    
class SessionCleanupResult(BaseModel):
    """Result of session cleanup operation"""
    sessions_cleaned: int = Field(..., description="Number of sessions removed")
    space_freed_mb: float = Field(..., description="Storage space freed in MB")
    oldest_remaining: Optional[datetime] = Field(None, description="Oldest remaining session")
    cleanup_duration: float = Field(..., description="Cleanup operation duration in seconds")