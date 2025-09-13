#!/usr/bin/env python3
"""
Streaming Error Handler - Simplified error handling for Deep Think streaming

Provides clean error reporting and graceful degradation for streaming operations
without complex retry logic. Focuses on clear error messages and fallback mechanisms.

Key Features:
- Simple error categorization and reporting  
- Graceful degradation strategies
- Clean error message formatting
- Connection failure handling
- No automatic retry logic (by design)
"""

import logging
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# =============================================================================
# Error Categories and Models
# =============================================================================

class ErrorCategory(Enum):
    """Categories of streaming errors"""
    CONNECTION = "connection"          # Client connection issues
    SESSION = "session"               # Session management errors
    ORCHESTRATOR = "orchestrator"     # Research orchestrator errors  
    STREAMING = "streaming"           # Streaming service errors
    TIMEOUT = "timeout"               # Timeout errors
    VALIDATION = "validation"         # Input validation errors
    SYSTEM = "system"                 # System/infrastructure errors

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"          # Non-critical, operation can continue
    MEDIUM = "medium"    # Important but recoverable
    HIGH = "high"        # Critical, operation should stop
    FATAL = "fatal"      # System-level failure

@dataclass
class StreamingError:
    """Structured streaming error"""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    technical_details: str
    user_message: str
    session_id: Optional[str] = None
    connection_id: Optional[str] = None
    timestamp: datetime = None
    error_code: Optional[str] = None
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if self.context is None:
            self.context = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'category': self.category.value,
            'severity': self.severity.value,
            'message': self.message,
            'user_message': self.user_message,
            'session_id': self.session_id,
            'connection_id': self.connection_id,
            'timestamp': self.timestamp.isoformat(),
            'error_code': self.error_code,
            'context': self.context
        }
    
    def to_sse_format(self) -> str:
        """Convert to Server-Sent Events format"""
        error_data = {
            'type': 'error',
            'category': self.category.value,
            'severity': self.severity.value,
            'message': self.user_message,
            'error_code': self.error_code,
            'session_id': self.session_id,
            'timestamp': self.timestamp.isoformat(),
            'recoverable': self.severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM]
        }
        return f"data: {json.dumps(error_data)}\n\n"

# =============================================================================
# Simplified Streaming Error Handler
# =============================================================================

class StreamingErrorHandler:
    """
    Simplified error handler for streaming operations
    
    Provides clean error categorization, user-friendly messages, and graceful
    degradation without complex retry mechanisms.
    """
    
    def __init__(self):
        self.error_count = 0
        self.errors_by_category = {category: 0 for category in ErrorCategory}
        self.last_errors = []  # Keep last 10 errors
        self.max_error_history = 10
        
        logger.info("StreamingErrorHandler initialized (no retry logic)")
    
    # =============================================================================
    # Error Creation and Categorization
    # =============================================================================
    
    def create_connection_error(self, message: str, session_id: str = None, 
                              connection_id: str = None, context: Dict[str, Any] = None) -> StreamingError:
        """Create connection-related error"""
        return StreamingError(
            category=ErrorCategory.CONNECTION,
            severity=ErrorSeverity.MEDIUM,
            message=message,
            technical_details=message,
            user_message="Connection issue occurred. Please refresh to reconnect.",
            session_id=session_id,
            connection_id=connection_id,
            error_code="CONN_ERR",
            context=context or {}
        )
    
    def create_session_error(self, message: str, session_id: str = None,
                           severity: ErrorSeverity = ErrorSeverity.HIGH,
                           context: Dict[str, Any] = None) -> StreamingError:
        """Create session-related error"""
        user_messages = {
            ErrorSeverity.LOW: "Minor session issue occurred.",
            ErrorSeverity.MEDIUM: "Session issue detected. Processing may be affected.",
            ErrorSeverity.HIGH: "Session error occurred. Please try again.",
            ErrorSeverity.FATAL: "Critical session error. Please contact support."
        }
        
        return StreamingError(
            category=ErrorCategory.SESSION,
            severity=severity,
            message=message,
            technical_details=message,
            user_message=user_messages[severity],
            session_id=session_id,
            error_code="SESS_ERR",
            context=context or {}
        )
    
    def create_orchestrator_error(self, message: str, session_id: str = None,
                                context: Dict[str, Any] = None) -> StreamingError:
        """Create orchestrator-related error"""
        return StreamingError(
            category=ErrorCategory.ORCHESTRATOR,
            severity=ErrorSeverity.HIGH,
            message=message,
            technical_details=message,
            user_message="Research processing error. The analysis could not be completed.",
            session_id=session_id,
            error_code="ORCH_ERR",
            context=context or {}
        )
    
    def create_timeout_error(self, timeout_seconds: int, session_id: str = None,
                           context: Dict[str, Any] = None) -> StreamingError:
        """Create timeout error"""
        return StreamingError(
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            message=f"Operation timed out after {timeout_seconds} seconds",
            technical_details=f"Timeout: {timeout_seconds}s",
            user_message=f"Analysis is taking longer than expected ({timeout_seconds}s). "
                        "Results will be saved when complete.",
            session_id=session_id,
            error_code="TIMEOUT",
            context=context or {'timeout_seconds': timeout_seconds}
        )
    
    def create_validation_error(self, message: str, field: str = None,
                              context: Dict[str, Any] = None) -> StreamingError:
        """Create validation error"""
        return StreamingError(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            message=message,
            technical_details=message,
            user_message="Invalid input. Please check your request and try again.",
            error_code="VALID_ERR",
            context=context or {'field': field} if field else context or {}
        )
    
    def create_system_error(self, message: str, session_id: str = None,
                          severity: ErrorSeverity = ErrorSeverity.HIGH,
                          context: Dict[str, Any] = None) -> StreamingError:
        """Create system-level error"""
        user_messages = {
            ErrorSeverity.MEDIUM: "System issue detected. Functionality may be limited.",
            ErrorSeverity.HIGH: "System error occurred. Please try again later.",
            ErrorSeverity.FATAL: "Critical system error. Please contact support."
        }
        
        return StreamingError(
            category=ErrorCategory.SYSTEM,
            severity=severity,
            message=message,
            technical_details=message,
            user_message=user_messages[severity],
            session_id=session_id,
            error_code="SYS_ERR",
            context=context or {}
        )
    
    # =============================================================================
    # Error Processing and Reporting
    # =============================================================================
    
    def handle_error(self, error: StreamingError) -> Dict[str, Any]:
        """
        Process error without retry logic
        
        Returns response information for client communication
        """
        # Track error statistics
        self.error_count += 1
        self.errors_by_category[error.category] += 1
        
        # Add to error history
        self.last_errors.append(error)
        if len(self.last_errors) > self.max_error_history:
            self.last_errors.pop(0)
        
        # Log error appropriately
        log_message = f"[{error.category.value.upper()}] {error.message}"
        if error.session_id:
            log_message += f" (session: {error.session_id})"
        
        if error.severity == ErrorSeverity.LOW:
            logger.info(log_message)
        elif error.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        elif error.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error.severity == ErrorSeverity.FATAL:
            logger.critical(log_message)
        
        # Determine response action
        response_action = self._determine_response_action(error)
        
        return {
            'error': error.to_dict(),
            'action': response_action,
            'should_disconnect': response_action in ['disconnect', 'terminate'],
            'recoverable': error.severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM]
        }
    
    def _determine_response_action(self, error: StreamingError) -> str:
        """
        Determine appropriate response action without retry logic
        
        Actions:
        - continue: Continue operation with degraded functionality
        - disconnect: Close connection gracefully
        - terminate: Stop processing immediately
        - fallback: Switch to alternative method
        """
        # Action based on severity
        if error.severity == ErrorSeverity.LOW:
            return 'continue'
        elif error.severity == ErrorSeverity.MEDIUM:
            if error.category in [ErrorCategory.CONNECTION, ErrorCategory.STREAMING]:
                return 'disconnect'
            else:
                return 'continue'
        elif error.severity == ErrorSeverity.HIGH:
            if error.category == ErrorCategory.SESSION:
                return 'terminate'
            else:
                return 'disconnect'
        elif error.severity == ErrorSeverity.FATAL:
            return 'terminate'
        
        return 'disconnect'  # Default safe action
    
    # =============================================================================
    # Error Reporting and Communication
    # =============================================================================
    
    async def send_error_to_client(self, error: StreamingError, handler) -> bool:
        """
        Send error message to client via SSE
        
        Returns True if message was sent successfully
        """
        try:
            handler.write(error.to_sse_format())
            await handler.flush()
            return True
            
        except Exception as send_error:
            logger.error(f"Failed to send error to client: {send_error}")
            return False
    
    def format_user_error_message(self, errors: List[StreamingError]) -> str:
        """Format multiple errors into user-friendly message"""
        if not errors:
            return "An unknown error occurred."
        
        if len(errors) == 1:
            return errors[0].user_message
        
        # Group errors by severity
        fatal_errors = [e for e in errors if e.severity == ErrorSeverity.FATAL]
        high_errors = [e for e in errors if e.severity == ErrorSeverity.HIGH]
        medium_errors = [e for e in errors if e.severity == ErrorSeverity.MEDIUM]
        
        # Prioritize most severe errors
        if fatal_errors:
            return "Critical system errors occurred. Please contact support."
        elif high_errors:
            return "Multiple errors occurred during processing. Please try again."
        elif medium_errors:
            return "Some issues occurred but processing may continue with limited functionality."
        else:
            return "Minor issues detected during processing."
    
    # =============================================================================
    # Graceful Degradation Strategies
    # =============================================================================
    
    def get_fallback_strategy(self, error: StreamingError) -> Dict[str, Any]:
        """
        Get fallback strategy for error without retry logic
        """
        strategies = {
            ErrorCategory.CONNECTION: {
                'strategy': 'basic_sse',
                'description': 'Fall back to basic SSE without enhanced streaming',
                'message': 'Switching to basic progress updates'
            },
            ErrorCategory.STREAMING: {
                'strategy': 'polling',
                'description': 'Fall back to polling for progress updates',
                'message': 'Progress updates available via refresh'
            },
            ErrorCategory.SESSION: {
                'strategy': 'standalone',
                'description': 'Continue without session management',
                'message': 'Processing continues without session tracking'
            },
            ErrorCategory.ORCHESTRATOR: {
                'strategy': 'simple_search',
                'description': 'Fall back to simple search instead of Deep Think',
                'message': 'Switching to simple search mode'
            },
            ErrorCategory.TIMEOUT: {
                'strategy': 'background',
                'description': 'Continue processing in background',
                'message': 'Processing continues in background'
            }
        }
        
        return strategies.get(error.category, {
            'strategy': 'none',
            'description': 'No fallback available',
            'message': 'Please try again later'
        })
    
    # =============================================================================
    # Error Statistics and Monitoring
    # =============================================================================
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring"""
        return {
            'total_errors': self.error_count,
            'errors_by_category': {cat.value: count for cat, count in self.errors_by_category.items()},
            'recent_errors': len(self.last_errors),
            'error_rate': self.error_count,  # Could be calculated per time period
            'most_common_category': max(self.errors_by_category, 
                                      key=self.errors_by_category.get).value if self.error_count > 0 else None
        }
    
    def get_recent_errors(self) -> List[Dict[str, Any]]:
        """Get recent errors for debugging"""
        return [error.to_dict() for error in self.last_errors]
    
    def clear_error_history(self) -> None:
        """Clear error history (for maintenance)"""
        self.last_errors.clear()
        logger.info("Error history cleared")
    
    # =============================================================================
    # Utility Methods  
    # =============================================================================
    
    def is_recoverable_error(self, error: StreamingError) -> bool:
        """Check if error is recoverable (no retry, just classification)"""
        return error.severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM]
    
    def should_terminate_session(self, error: StreamingError) -> bool:
        """Check if error should terminate the entire session"""
        return (error.severity == ErrorSeverity.FATAL or 
                (error.severity == ErrorSeverity.HIGH and 
                 error.category in [ErrorCategory.SESSION, ErrorCategory.ORCHESTRATOR]))
    
    def get_error_summary(self, session_id: str) -> Dict[str, Any]:
        """Get error summary for a specific session"""
        session_errors = [e for e in self.last_errors if e.session_id == session_id]
        
        if not session_errors:
            return {'session_id': session_id, 'error_count': 0}
        
        return {
            'session_id': session_id,
            'error_count': len(session_errors),
            'categories': list(set(e.category.value for e in session_errors)),
            'highest_severity': max(e.severity.value for e in session_errors),
            'latest_error': session_errors[-1].to_dict()
        }